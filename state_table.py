"""
state_table.py — полный перебор состояний агента + табличная value-функция от VLM.

Идея Кирилла: мир дискретный и раскладка фиксированная, значит можно один раз
перечислить ВСЕ позы агента ((свободные клетки + goal) x 4 направления),
отрендерить, оценить VLM'ом и сложить в словарь. Планировщик VLM больше
не зовёт вообще.

Один тонкий момент: байтовый ключ-картинка работает только для РЕАЛЬНЫХ
рендеров (они детерминированы). Декодер world model байт-в-байт не попадёт
никогда. Поэтому для воображаемых кадров — nearest-neighbor по пикселям
(avg-pool 8x8 -> L2 к эталонам, один cdist), а точный dict остаётся для
реальных кадров.

v2: РАЗМЕТКА И МАТЧИНГ ЖИВУТ В РАЗНЫХ РАЗРЕШЕНИЯХ.
Раньше кадры жались до 64px ДО подачи в VLM — доля зелёного в клетке goal
при агенте НА ней падает 0.76 -> 0.19, NEAREST-апскейл 64->448 это не
восстанавливает. Отсюда occlusion-inversion в таблице: сосед goal скорится
выше самой goal, аттрактор смещён, планировщик паркуется рядом.
Теперь enumerate_agent_states(label_hires=True) дополнительно возвращает
родные рендеры (320px) — их и кормим VLM'у; 64px остаются эталонами NN.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@torch.no_grad()
def enumerate_agent_states(env, seed=0, out_size=64, label_hires=False):
    """Все допустимые позы агента при фиксированной раскладке.
    -> frames (K, out_size, out_size, 3) uint8, states [(x, y, dir), ...].
       при label_hires=True: frames, frames_hi, states — frames_hi это
       РОДНЫЕ рендеры среды (tile_size*grid, обычно 320px) для VLM-разметки.
    Клетка допустима, если пустая или can_overlap (goal). Закрытая дверь,
    ключ и стены отсекаются автоматически."""
    env.reset(seed=seed)
    frames, frames_hi, states = [], [], []
    for x in range(1, env.width - 1):
        for y in range(1, env.height - 1):
            cell = env.grid.get(x, y)
            if cell is not None and not cell.can_overlap():
                continue
            for d in range(4):
                env.agent_pos = np.array([x, y])
                env.agent_dir = d
                rgb = env.render()                                  # родные ~320px
                frames.append(np.array(
                    Image.fromarray(rgb).resize((out_size, out_size))))
                if label_hires:
                    frames_hi.append(rgb)
                states.append((x, y, d))
    print(f"состояний: {len(states)} "
          f"({len(states) // 4} клеток x 4 направления)")
    if label_hires:
        return np.stack(frames), np.stack(frames_hi), states
    return np.stack(frames), states


@torch.no_grad()
def build_state_values(scorer, frames, prompts, batch_size=24, verbose=True):
    """Скорим каждое состояние КАЖДЫМ промптом (один проход на все задачи),
    кросс-нормируем по строке: v_task = p_task / sum_p.
    frames: uint8 (K,H,W,3) — можно и нужно подавать hi-res (см. enumerate).
    -> dict {task: tensor (K,)}."""
    raw = {}
    for task, prompt in prompts.items():
        scorer.set_goal(prompt)
        chunks = [scorer.score_rgb(list(frames[i:i + batch_size]))
                  for i in range(0, len(frames), batch_size)]
        raw[task] = torch.cat(chunks).float().cpu()
        if verbose:
            print(f"  '{task}': mean={raw[task].mean():.3f} "
                  f"std={raw[task].std():.3f} max={raw[task].max():.3f}")
    total = sum(raw.values()) + 1e-8
    return {t: raw[t] / total for t in prompts}


def pin_terminal(values, states, cells, value=None):
    """FALLBACK, не основной путь: принудительно поднять значение состояний в
    заданных клетках (все 4 направления) до value (default: max(values)+0.1).
    Вшивает ground-truth позицию цели, т.е. подрывает тезис «меняется только
    текст» — использовать, только если hi-res разметка + промпт не вытянули
    on-goal в топ. ВАЖНО: вызывать ДО постройки StateTable/objectives."""
    v = values.clone()
    if value is None:
        value = float(values.max()) + 0.1
    cells = {tuple(c) for c in cells}
    for i, (x, y, d) in enumerate(states):
        if (x, y) in cells:
            v[i] = value
    return v


class StateTable:
    """Эталонные кадры + значения одной задачи.
    score_decoded: NN-lookup для воображаемых декодов (никакого VLM).
    exact: точный словарь по байтам — для реальных кадров."""

    def __init__(self, frames_uint8, values, device, pool=8):
        self.pool = pool
        x = torch.from_numpy(frames_uint8).float().div(255).permute(0, 3, 1, 2)
        self.refs = F.adaptive_avg_pool2d(x, pool).flatten(1).to(device)   # (K,D)
        self.values = values.float().to(device)                            # (K,)
        self.exact = {frames_uint8[i].tobytes(): float(values[i])
                      for i in range(len(values))}

    @torch.no_grad()
    def score_decoded(self, frames):
        """frames (N,3,64,64) float [-0.5,0.5] -> (N,) значения ближайших эталонов.
        self.last_dist — L2 до соседа: если медиана растёт, декодер уехал с
        многообразия реальных кадров (бесплатный диагност воображения)."""
        q = F.adaptive_avg_pool2d((frames.float() + 0.5).clamp(0, 1),
                                  self.pool).flatten(1)
        dist, idx = torch.cdist(q, self.refs).min(dim=1)
        self.last_dist, self.last_idx = dist, idx
        return self.values[idx]


def table_objective(model, table, mode="sum", chunk=512):
    """Objective для планировщика: декодим воображаемые латенты и смотрим в
    таблицу. VLM-вызовов ноль, поэтому можно позволить sum по ВСЕМУ горизонту
    (плотный сигнал 'дойди и стой') вместо только последнего кадра."""
    @torch.no_grad()
    def score(hs, ss):
        H, N = hs.shape[:2]
        feats = torch.cat([hs, ss], dim=-1).reshape(H * N, -1)
        vals = torch.cat([table.score_decoded(model.decoder(feats[i:i + chunk]))
                          for i in range(0, H * N, chunk)]).reshape(H, N)
        if mode == "last":
            return vals[-1]
        if mode == "max":
            return vals.max(dim=0).values
        return vals.sum(dim=0)
    return score


def value_heatmap(states, values, size):
    """(size,size) карта: max по 4 направлениям в клетке. NaN — недоступно.
    Рисовать: plt.imshow(hm, origin='upper'); отчётная картинка ландшафта VLM."""
    hm = np.full((size, size), np.nan)
    for (x, y, d), v in zip(states, values.tolist()):
        hm[y, x] = v if np.isnan(hm[y, x]) else max(hm[y, x], v)
    return hm


def inspect_states(states, values, where):
    """Значения в конкретных клетках, напр. where=[(8,8)] — goal с агентом НА
    клетке: офлайн-проверка occlusion до всякого планирования."""
    out = []
    for (x, y, d), v in zip(states, values.tolist()):
        if (x, y) in where:
            out.append(((x, y, d), round(v, 4)))
    return out
