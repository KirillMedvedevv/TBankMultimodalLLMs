"""
fast_planner.py — быстрый планировщик + дистилляция VLM в reward-голову.

Три идеи, каждая независима (можно включать по одной):

1) CachedFrameScorer — обёртка над KaggleScorer. Дискретный мир => из N
   воображаемых кандидатов уникальных терминальных кадров мало. Квантуем
   кадр (avg-pool 8x8, 12 уровней) -> байтовый ключ -> Qwen зовём только по
   уникальным, остальным раздаём из кэша. Кэш живёт между шагами эпизода
   (и между сидами одной задачи): соседние шаги смотрят на те же клетки.
   Drop-in: vlm_objective(model, CachedFrameScorer(scorer)) работает как раньше.

2) shooting_plan / run_episode_v2 — random shooting, исправленный:
   - действия сэмплим только из MOVE_ACTIONS=(0,1,2): для задач key/door/goal
     pickup/drop/toggle/done не нужны (успех = стоять рядом/на клетке), а с
     ACTION_DIM=7 больше половины горизонта уходило в no-op'ы;
   - warm-start: лучший план прошлого шага сдвигаем на 1 и кладём в кандидаты
     вместе с его мутациями (MPC-стиль) — меньше дёрганья на плоском сигнале;
   - replan_every=k: исполняем первые k действий плана (фильтр наблюдает
     каждый реальный кадр), VLM-планирование зовём в k раз реже.

3) Дистилляция VLM -> VLMHead(h,s): один раз размечаем кадры реплей-буфера
   Qwen'ом (по уникальным кадрам — их сотни, не тысячи), учим MLP на латентах
   постериора, и планируем ЭТОЙ головой — так же дёшево, как reward-baseline.
   Симметрия для отчёта: reward-голова : среда :: VLM-голова : текстовый промпт.

Формат эпизодов — как в data.py: dict(images (T,64,64,3) uint8,
actions (T,) int64, rewards (T,) float32).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

MOVE_ACTIONS = (0, 1, 2)          # left, right, forward — достаточно для key/door/goal
TURN_ACTIONS = (0, 1)


# --------------------------------------------------------------- утилиты
def prep_frame(frame, device):
    """env.render() (H,W,3) uint8 -> (1,3,64,64) float [-0.5,0.5]."""
    img = Image.fromarray(frame).resize((64, 64))
    t = torch.from_numpy(np.array(img)).float().div(255).sub(0.5)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)


def frame_keys(frames, pool=8, levels=12):
    """(N,3,64,64) float [-0.5,0.5] -> список байтовых ключей.
    Даунсемпл + грубая квантовка: почти одинаковые декоды -> один ключ."""
    x = (frames.detach().float() + 0.5).clamp(0, 1)
    x = F.adaptive_avg_pool2d(x, pool)
    q = (x * (levels - 1)).round().to(torch.uint8).cpu().numpy()
    return [q[i].tobytes() for i in range(q.shape[0])]


def anti_spin_penalty(actsHN, w_turn=0.02, w_spin=0.03):
    """(H,N) индексы действий -> (N,) штраф за вращение на месте."""
    a = actsHN.T
    is_turn = (a == TURN_ACTIONS[0]) | (a == TURN_ACTIONS[1])
    turn_frac = is_turn.float().mean(dim=1)
    consec = (is_turn[:, :-1] & is_turn[:, 1:]).float().sum(dim=1)
    return w_turn * turn_frac + w_spin * consec


# ------------------------------------------------- 1) кэширующий скорер
class CachedFrameScorer:
    """Обёртка над scorer'ом с .score(frames)->(N,): одинаковые (после
    квантования) кадры скорим один раз. set_goal чистит кэш (смена задачи)."""

    def __init__(self, scorer, pool=8, levels=12):
        self.scorer, self.pool, self.levels = scorer, pool, levels
        self.cache = {}
        self.requests = 0        # сколько кадров попросили
        self.forwards = 0        # сколько реально ушло в VLM

    def set_goal(self, *a, **k):
        self.cache.clear()
        return self.scorer.set_goal(*a, **k)

    def stats(self):
        saved = 0 if self.requests == 0 else 1 - self.forwards / self.requests
        return dict(requests=self.requests, vlm_forwards=self.forwards,
                    saved_frac=round(saved, 3), cache_size=len(self.cache))

    @torch.no_grad()
    def score(self, frames):
        keys = frame_keys(frames, self.pool, self.levels)
        self.requests += len(keys)
        first_row = {}
        for i, k in enumerate(keys):
            first_row.setdefault(k, i)
        missing = [k for k in first_row if k not in self.cache]
        if missing:
            rows = torch.tensor([first_row[k] for k in missing])
            vals = self.scorer.score(frames[rows])
            for k, v in zip(missing, vals.tolist()):
                self.cache[k] = float(v)
            self.forwards += len(missing)
        return torch.tensor([self.cache[k] for k in keys])


def vlm_objective(model, scorer, mode="last"):
    """Как в ноутбуке; передавай сюда CachedFrameScorer."""
    def score(hs, ss):
        H, N = hs.shape[:2]
        if mode == "last":
            frames = model.decoder(torch.cat([hs[-1], ss[-1]], dim=-1))
            return scorer.score(frames)
        frames = model.decoder(torch.cat([hs, ss], dim=-1).reshape(H * N, -1))
        return scorer.score(frames).reshape(H, N).mean(dim=0)
    return score


def reward_objective(model):
    def score(hs, ss):
        return model.reward(hs, ss).sum(dim=0)
    return score


# ------------------------------------------------- 2) планировщик v2
@torch.no_grad()
def shooting_plan(model, h, s, objective, horizon, n_candidates, device,
                  action_dim=7, actions=MOVE_ACTIONS, prev_plan=None,
                  n_warm=8, mut_p=0.3, penalty=anti_spin_penalty):
    """Random shooting: действия из подмножества `actions`, warm-start прошлым
    планом + его мутациями. Возвращает ЛУЧШИЙ ПЛАН целиком (H,) cpu int64."""
    H, N = horizon, n_candidates
    lut = torch.tensor(actions, device=device)
    acts = lut[torch.randint(0, len(actions), (H, N), device=device)]   # (H,N) env-id

    if prev_plan is not None and len(prev_plan) > 0:
        prev = prev_plan.to(device)[:H]
        if len(prev) < H:
            pad = lut[torch.randint(0, len(actions), (H - len(prev),), device=device)]
            prev = torch.cat([prev, pad])
        acts[:, 0] = prev                                   # план прошлого шага
        m = min(n_warm - 1, N - 1)
        if m > 0:                                           # его мутации
            mask = torch.rand(H, m, device=device) < mut_p
            rnd = lut[torch.randint(0, len(actions), (H, m), device=device)]
            acts[:, 1:1 + m] = torch.where(mask, rnd, prev.unsqueeze(1).expand(H, m))

    oh = F.one_hot(acts, action_dim).float()
    hh = h.expand(N, -1).contiguous()
    sc = s.expand(N, -1).contiguous()
    hs, sl = [], []
    for t in range(H):
        hh, sc = model.imagine_step(hh, sc, oh[t])
        hs.append(hh); sl.append(sc)
    scores = objective(torch.stack(hs), torch.stack(sl)).to(device).float()
    if penalty is not None:
        scores = scores - penalty(acts).to(scores)
    return acts[:, int(scores.argmax())].detach().cpu()


@torch.no_grad()
def run_episode_v2(env, model, objective, task, seed, horizon, max_steps, device,
                   n_candidates=256, replan_every=1, actions=MOVE_ACTIONS,
                   action_dim=7, penalty=anti_spin_penalty, record=False):
    """Эпизод с warm-start'ом и исполнением replan_every действий за план.
    Фильтр наблюдает КАЖДЫЙ реальный кадр.
    Возвращает (успех, шаги), с record=True — (успех, шаги, кадры)."""
    env.reset(seed=seed)
    h, s = model.initial(1, device)
    prev_a = torch.zeros(1, action_dim, device=device)
    frames = [env.render()] if record else None
    plan, steps = None, 0

    while steps < max_steps and not env.task_success(task):
        h, s = model.observe_step(h, s, prev_a, prep_frame(env.render(), device))
        plan = shooting_plan(model, h, s, objective, horizon, n_candidates, device,
                             action_dim=action_dim, actions=actions,
                             prev_plan=plan, penalty=penalty)
        k = 0
        while k < replan_every and steps < max_steps and not env.task_success(task):
            a = int(plan[k])
            env.step(a)
            prev_a = F.one_hot(torch.tensor([a], device=device), action_dim).float()
            if record:
                frames.append(env.render())
            steps += 1; k += 1
            more = k < replan_every and steps < max_steps and not env.task_success(task)
            if more:
                h, s = model.observe_step(h, s, prev_a, prep_frame(env.render(), device))
        plan = plan[k:]

    ok = bool(env.task_success(task))
    return (ok, steps, frames) if record else (ok, steps)


# ------------------------------------------------- 3) дистилляция VLM
class VLMHead(nn.Module):
    """MLP (h,s) -> скаляр, как RewardModel, но учится на метках VLM."""

    def __init__(self, stoch=30, deter=200, hidden=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(deter + stoch, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h, s):
        return self.net(torch.cat([h, s], dim=-1)).squeeze(-1)


@torch.no_grad()
def episodes_to_latents(model, episodes, device, max_episodes=None):
    """Прогоняем observe-фильтр по эпизодам буфера.
    -> feats (M, deter+stoch) float32 cpu, frames (M,64,64,3) uint8.
    Латент (h_t,s_t) соответствует кадру images[t] (кадр ДО действия t)."""
    feats, frames = [], []
    eps = episodes[:max_episodes] if max_episodes else episodes
    for e in eps:
        imgs, acts = e["images"], e["actions"]
        T = len(acts)
        h, s = model.initial(1, device)
        prev_a = torch.zeros(1, model.action_dim, device=device)
        for t in range(T):
            x = torch.from_numpy(imgs[t]).float().div(255).sub(0.5)
            x = x.permute(2, 0, 1).unsqueeze(0).to(device)
            h, s = model.observe_step(h, s, prev_a, x)
            feats.append(torch.cat([h, s], dim=-1).squeeze(0).cpu())
            frames.append(imgs[t])
            prev_a = F.one_hot(torch.tensor([int(acts[t])], device=device),
                               model.action_dim).float()
    return torch.stack(feats), np.stack(frames)


@torch.no_grad()
def vlm_label_frames(scorer, frames_uint8, task, prompts, batch_size=24,
                     cross_normalize=True, verbose=True):
    """Метки VLM для (M,64,64,3) uint8. Дедуп по точному хэшу кадра: рендер
    детерминирован, дискретных состояний сотни -> Qwen зовём только по ним.
    cross_normalize: скорим уникальные кадры ВСЕМИ промптами и берём
    p_task / sum_p (row-softmax дебиасинг) — офлайн это дёшево.
    -> labels (M,) float32."""
    keys = [frames_uint8[i].tobytes() for i in range(len(frames_uint8))]
    first_row = {}
    for i, k in enumerate(keys):
        first_row.setdefault(k, i)
    uniq_rows = list(first_row.values())
    uniq = [frames_uint8[i] for i in uniq_rows]
    if verbose:
        print(f"кадров {len(keys)} -> уникальных {len(uniq)}")

    task_list = list(prompts) if cross_normalize else [task]
    per_task = {}
    for t in task_list:
        scorer.set_goal(prompts[t])
        out = [scorer.score_rgb(uniq[i:i + batch_size])
               for i in range(0, len(uniq), batch_size)]
        per_task[t] = torch.cat(out).float()
        if verbose:
            print(f"  '{t}' размечен: mean={per_task[t].mean():.3f} "
                  f"std={per_task[t].std():.3f}")

    p = per_task[task]
    if cross_normalize:
        total = sum(per_task.values()) + 1e-8
        p = p / total
    lab = {k: float(v) for k, v in zip(first_row.keys(), p)}
    return torch.tensor([lab[k] for k in keys], dtype=torch.float32)


def train_vlm_head(feats, labels, stoch=30, deter=200, device="cpu",
                   epochs=300, lr=1e-3, batch=1024, verbose=True):
    """MSE-регрессия меток VLM на латентах постериора. Минуты даже на CPU."""
    head = VLMHead(stoch=stoch, deter=deter).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    X, y = feats.to(device), labels.to(device)
    n, d = len(X), deter
    for ep in range(epochs):
        idx = torch.randperm(n, device=device)[:batch]
        pred = head(X[idx, :d], X[idx, d:])
        loss = F.mse_loss(pred, y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and ep % max(1, epochs // 5) == 0:
            print(f"  epoch {ep:4d} | mse {loss.item():.5f}")
    head.eval()
    return head


def distilled_objective(head, mode="sum"):
    """Планирование дистиллированной головой — по цене reward-baseline."""
    @torch.no_grad()
    def score(hs, ss):
        v = head(hs, ss)                       # (H,N)
        if mode == "last":
            return v[-1]
        if mode == "max":
            return v.max(dim=0).values
        return v.sum(dim=0)
    return score


# ------------------------------------------------- диагностика воображения
@torch.no_grad()
def probe_imagination(model, env, device, seed=0, pos=None, direction=None,
                      warmup=3, n_steps=6, action=2, path="imagine_probe.png"):
    """Ставим агента (опционально pos/direction, напр. лицом в стену), греем
    фильтр warmup реальными кадрами (действие done=6 — no-op), затем воображаем
    n_steps раз `action` и декодим; ниже — что сделала бы РЕАЛЬНАЯ среда.
    Если в воображении агент проходит сквозь стену, а в реальности упирается —
    вот источник «ходит в стену»."""
    env.reset(seed=seed)
    if pos is not None:
        env.agent_pos = np.array(pos)
    if direction is not None:
        env.agent_dir = direction

    h, s = model.initial(1, device)
    prev_a = torch.zeros(1, model.action_dim, device=device)
    for _ in range(warmup):
        h, s = model.observe_step(h, s, prev_a, prep_frame(env.render(), device))
        env.step(6)                                           # no-op
        prev_a = F.one_hot(torch.tensor([6], device=device),
                           model.action_dim).float()

    def to64(fr):
        return np.array(Image.fromarray(fr).resize((64, 64)))

    start = to64(env.render())
    oh = F.one_hot(torch.tensor([action], device=device), model.action_dim).float()
    hh, ss = h, s
    imagined = []
    for _ in range(n_steps):
        hh, ss = model.imagine_step(hh, ss, oh)
        d = model.decoder(torch.cat([hh, ss], dim=-1))[0]
        imagined.append(((d + 0.5).clamp(0, 1) * 255).byte()
                        .permute(1, 2, 0).cpu().numpy())
    real = []
    for _ in range(n_steps):
        env.step(action)
        real.append(to64(env.render()))

    row_i = np.concatenate([start] + imagined, axis=1)
    row_r = np.concatenate([start] + real, axis=1)
    strip = np.concatenate([row_i, row_r], axis=0)            # верх: сон, низ: реальность
    Image.fromarray(strip).resize((strip.shape[1] * 3, strip.shape[0] * 3),
                                  Image.NEAREST).save(path)
    print(f"верхний ряд — воображение, нижний — среда (action={action}) -> {path}")
    return path
