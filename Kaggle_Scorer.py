"""
kaggle_scorer.py

VLM scorer (yes/no logit style) for the planner, meant to run on a Kaggle GPU
notebook. Swappable backend: pick a model with the `model=` argument and the same
`.score()` / `.set_goal()` interface works for either family.

Why yes/no instead of CLIP:
  CLIP / SigLIP score by cosine similarity to a text embedding -> fuzzy on our
  tiny grid frames (confuses "blue door" with "green square"), slow on CPU.
  A generative VLM is ASKED "is the goal achieved?" and we read P(Yes) straight
  from the next-token logits. That P in [0,1] is the score -> discriminative by
  construction, no negative-prompt hack needed.

Supported backends (auto-detected from the model name, or force with backend=):
  qwen     -> Qwen2-VL-2B / Qwen2.5-VL-3B / 7B   (same code, process_vision_info)
  smolvlm  -> SmolVLM / SmolVLM2                 (AutoModelForImageTextToText)

Big models on Kaggle (2 x T4, 16GB each):
  7B fp16 (~16.6GB весов) в одну T4 не лезет — шардим на обе:
    KaggleScorer("qwen2.5-7b", device_map="auto",
                 max_memory={0: "13GiB", 1: "13GiB"})
  либо 4-бит NF4 на ОДНОЙ T4 (~5.5GB, чуть шумнее логиты, ранжирование обычно ок):
    KaggleScorer("qwen2.5-7b", load_4bit=True)      # pip install bitsandbytes

Drop-in interface (same shape contract as SigLIPScorer / CLIPScorer):
    scorer = KaggleScorer(model="qwen2.5-3b")
    scorer.set_goal("the red agent is on the green goal square")
    scores = scorer.score(frames)                    # (N,3,64,64) in [-0.5,0.5] -> (N,)

Kaggle setup (once):
    !pip install -U "transformers>=4.50.0" accelerate
    !pip install -U qwen-vl-utils          # only needed for the qwen backend
    !pip install -U bitsandbytes           # only for load_4bit=True
"""

import inspect

import numpy as np
import torch
from PIL import Image

# NB: токена здесь больше нет и быть не должно. Qwen публичный, login() не нужен.
# Старый hf_... токен из этого файла — ОТОЗВАТЬ на huggingface.co/settings/tokens.

try:
    from qwen_vl_utils import process_vision_info
    _HAS_QWEN_UTILS = True
except Exception:
    _HAS_QWEN_UTILS = False


PRESETS = {
    "qwen2.5-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
}


def _detect_backend(model_name):
    n = model_name.lower()
    if "smolvlm" in n:
        return "smolvlm"
    if "qwen" in n:
        return "qwen"
    raise ValueError(f"can't infer backend from {model_name!r}; pass backend='qwen' or 'smolvlm'")


class KaggleScorer:
    def __init__(self, model="qwen2.5-3b", device=None, dtype=None,
                 upscale=448, max_pixels=512 * 512, backend=None,
                 device_map=None, max_memory=None, load_4bit=False):
        """device_map="auto" — шардинг весов по всем видимым GPU (для 7B на 2xT4).
        max_memory={0:"13GiB",1:"13GiB"} — потолки на карту, оставляем запас
        под активации. load_4bit=True — NF4-квантизация (одна GPU, bitsandbytes).
        При шардинге .to(device) на модель звать нельзя — accelerate сам
        раскладывает и гоняет активации между картами; входы кладём на cuda:0."""
        self.model_name = PRESETS.get(model, model)
        self.backend = backend or _detect_backend(self.model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            if "cuda" in str(self.device):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype
        self.upscale = upscale

        if self.backend == "qwen":
            self._load_qwen(self.model_name, max_pixels, device_map, max_memory, load_4bit)
        elif self.backend == "smolvlm":
            self._load_smolvlm(self.model_name)
        else:
            raise ValueError(f"unknown backend {self.backend!r}")

        self.tok = self.processor.tokenizer
        self.tok.padding_side = "left"        # so logits[:, -1] is the real last token per row

        # only ask for the last-position logits if this transformers version supports it
        params = inspect.signature(self.model.forward).parameters
        if "logits_to_keep" in params:
            self._fwd_kw = {"logits_to_keep": 1}
        elif "num_logits_to_keep" in params:
            self._fwd_kw = {"num_logits_to_keep": 1}
        else:
            self._fwd_kw = {}

        self.system = ("You are a precise visual grader for a top-down grid-world game. "
                       "Judge only what is visible in the image and answer with one word.")
        self.question = None
        self._yes_ids = self._token_ids(["Yes", "yes", "YES", " Yes"])
        self._no_ids = self._token_ids(["No", "no", "NO", " No"])

    # ---------------- backend loading ----------------
    def _load_qwen(self, model_name, max_pixels, device_map, max_memory, load_4bit):
        from transformers import AutoProcessor
        if "2.5" in model_name or "2_5" in model_name:
            from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        else:
            from transformers import Qwen2VLForConditionalGeneration as Cls

        kw = dict(torch_dtype=self.dtype, low_cpu_mem_usage=True)
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(torch.float16 if self.dtype == torch.float16
                                        else torch.bfloat16))
            kw["device_map"] = device_map or {"": 0}
        elif device_map is not None:
            kw["device_map"] = device_map
            if max_memory is not None:
                kw["max_memory"] = max_memory

        if "device_map" in kw:
            self.model = Cls.from_pretrained(model_name, **kw).eval()
            self.device = "cuda:0"    # входы на первый шард, дальше accelerate сам
            if hasattr(self.model, "hf_device_map"):
                gpus = {v for v in self.model.hf_device_map.values()}
                print(f"[scorer] шардинг по устройствам: {sorted(map(str, gpus))}")
        else:
            self.model = Cls.from_pretrained(model_name, **kw).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(model_name, max_pixels=max_pixels)

    def _load_smolvlm(self, model_name):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=self.dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        # our frames are tiny; disable tiling so each is ONE small image -> far fewer
        # vision tokens, much faster
        try:
            self.processor.image_processor.do_image_splitting = False
        except Exception:
            pass

    # ---------------- goal / prompt ----------------
    def set_goal(self, goal_text, negative_texts=None):
        """goal_text: a STATEMENT of the achieved goal, phrased so yes/no fits,
        e.g. 'the red agent is on the green goal square' or 'the blue door is open'.
        negative_texts accepted for drop-in compat with SigLIPScorer but ignored."""
        self.question = (f"Is the following true in this image: {goal_text}? "
                         f"Answer with only 'Yes' or 'No'.")

    # ---------------- public scoring ----------------
    @torch.no_grad()
    def score(self, frames, batch_size=16):
        """frames: (N,3,64,64) float in [-0.5,0.5] (RSSM decoder output).
        returns: (N,) tensor of P(Yes) in [0,1] on CPU."""
        return self._score_pils(self._frames_to_pil(frames), batch_size)

    @torch.no_grad()
    def score_rgb(self, rgb_list, batch_size=16):
        """rgb_list: iterable of (H,W,3) uint8 env renders -> (N,) P(Yes)."""
        pil = [self._prep_pil(Image.fromarray(np.asarray(f).astype(np.uint8)))
               for f in rgb_list]
        return self._score_pils(pil, batch_size)

    # ---------------- internals ----------------
    def _token_ids(self, words):
        ids = set()
        for w in words:
            toks = self.tok(w, add_special_tokens=False).input_ids
            if toks:
                ids.add(toks[0])
        return sorted(ids)

    def _prep_pil(self, img):
        # crisp NEAREST upscale so the little agent / door / key stay legible
        return img.convert("RGB").resize((self.upscale, self.upscale), Image.NEAREST)

    def _frames_to_pil(self, frames):
        if not torch.is_tensor(frames):
            frames = torch.as_tensor(frames)
        x = (frames.detach().float() + 0.5).clamp(0, 1)          # [-0.5,0.5] -> [0,1]
        x = (x * 255).round().to(torch.uint8).cpu()              # (N,3,64,64)
        return [self._prep_pil(Image.fromarray(f.permute(1, 2, 0).numpy())) for f in x]

    def _score_pils(self, pil_list, batch_size):
        assert self.question is not None, "call set_goal(...) first"
        out = [self._batch_pyes(pil_list[i:i + batch_size])
               for i in range(0, len(pil_list), batch_size)]
        return torch.cat(out)

    def _build_inputs(self, images):
        if self.backend == "qwen":
            messages = [[
                {"role": "system", "content": self.system},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": self.question},
                ]},
            ] for img in images]
            texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                     for m in messages]
            image_inputs = process_vision_info(messages)[0] if _HAS_QWEN_UTILS else images
            return self.processor(text=texts, images=image_inputs,
                                  padding=True, return_tensors="pt").to(self.device)

        # smolvlm: image placeholder carries no data; images passed separately as
        # a list-of-lists (one list per sample). system folded into the user text.
        messages = [[
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": self.system + "\n" + self.question},
            ]},
        ] for _ in images]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in messages]
        return self.processor(text=texts, images=[[img] for img in images],
                              padding=True, return_tensors="pt").to(self.device)

    def _batch_pyes(self, images):
        inputs = self._build_inputs(images)
        logits = self.model(**inputs, **self._fwd_kw).logits[:, -1, :].float()   # (B, vocab)
        logp = torch.log_softmax(logits, dim=-1)
        p_yes = logp[:, self._yes_ids].logsumexp(dim=-1).exp()
        p_no = logp[:, self._no_ids].logsumexp(dim=-1).exp()
        return (p_yes / (p_yes + p_no + 1e-8)).cpu()


# ============================= TESTER =============================
def run_frame_test(scorer, goal_text, show_png=None):
    """Place the agent at telling spots, render clean frames, print P(Yes) for the
    goal. If the model understands the scene, 'on goal' should top the list and
    door/key/start sit clearly lower."""
    from env_tasks import MultiObjEnv          # поправь импорт если нужно

    scorer.set_goal(goal_text)
    env = MultiObjEnv(size=10, render_mode="rgb_array", highlight=False)
    env.reset(seed=0)
    gx, gy = env.goal_pos

    positions = {
        "start":       (1, 1),
        "on goal":     (gx, gy),
        "near goal x": (gx - 1, gy),
        "near goal y": (gx, gy - 1),
        "door":        (4, 2),
        "key":         (1, gy),
    }
    rgb_frames, names = [], []
    for name, (x, y) in positions.items():
        env.reset(seed=0)
        env.agent_pos = np.array([x, y])
        env.agent_dir = 0
        rgb_frames.append(env.render())
        names.append(name)

    sc = scorer.score_rgb(rgb_frames).tolist()
    print(f"goal pos {(gx, gy)} | goal {goal_text!r}\n")
    for name, v in sorted(zip(names, sc), key=lambda z: -z[1]):
        print(f"{v:.3f} {'#' * int(round(v * 30)):<30} {name}")

    if show_png:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for ax, name, img, v in zip(axes.flatten(), names, rgb_frames, sc):
            ax.imshow(img)
            ax.set_title(f"{name}\nP(yes)={v:.3f}", fontsize=10)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(show_png, dpi=110)
        print(f"\nsaved -> {show_png}")
    return dict(zip(names, sc))


def compare_models(models, goal_text, **kwargs):
    """Score the same frames with several models back-to-back, freeing each before
    the next so a Kaggle GPU doesn't OOM. models: list of presets or HF ids."""
    results = {}
    for m in models:
        print("=" * 64, f"\nMODEL: {m}")
        scorer = KaggleScorer(model=m, **kwargs)
        print(f"backend={scorer.backend} device={scorer.device} dtype={scorer.dtype} "
              f"fwd_kw={scorer._fwd_kw}\n")
        results[m] = run_frame_test(scorer, goal_text)
        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    import sys
    MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-3b"
    GOALS = ["the red triangular agent is standing on the green goal square",
             "the red triangular agent next to yellow key",
             "the red triangular agent next to blue door"]

    scorer = KaggleScorer(model=MODEL)
    k = 0
    for i in GOALS:
        k += 1
        print("Test", k)
        print("backend:", scorer.backend, "| model:", scorer.model_name,
              "| device:", scorer.device, "| dtype:", scorer.dtype, "| fwd_kw:", scorer._fwd_kw)
        run_frame_test(scorer, i, show_png=f"kaggle_scorer_test_{k}.png")
