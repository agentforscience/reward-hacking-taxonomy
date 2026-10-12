"""Shared paths, seeding and the reward-model wrapper."""
import os, random, glob
import numpy as np
import torch
import torch.nn as nn

HUB = os.path.join(os.environ.get("HF_HOME", "/data/transformers/"), "hub")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
# Large artefacts (adapters) live outside the workspace: /data is almost full.
SCRATCH = os.environ.get("RH_SCRATCH", "/tmp/rh_taxonomy")

RM_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
JUDGES = {"qwen14b": "Qwen/Qwen2.5-14B-Instruct",
          "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
          "llama8b": "unsloth/Meta-Llama-3.1-8B-Instruct",
          "smoke": "Qwen/Qwen2.5-0.5B-Instruct"}   # code-path test only, never used for results


def snapshot(repo, kind="models"):
    """Resolve a cached hub snapshot directory (no network access)."""
    pat = os.path.join(HUB, f"{kind}--{repo.replace('/', '--')}", "snapshots", "*")
    cands = [d for d in sorted(glob.glob(pat))
             if kind != "models" or os.path.exists(os.path.join(d, "config.json"))]
    assert cands, f"{repo} not found in {HUB}"
    return cands[-1]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class RewardModel(nn.Module):
    """Causal LM backbone (+LoRA) with a scalar head on the final token."""
    MAX_PROMPT_TOK, MAX_RESP_TOK = 256, 512

    def __init__(self, device, lora=True):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = snapshot(RM_BASE)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"   # so position -1 is always the real last token
        self.backbone = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
        if lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                             "gate_proj", "up_proj", "down_proj"])
            self.backbone = get_peft_model(self.backbone, cfg)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
        self.device = device
        self.to(device)

    def _clip(self, text, n):
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return text if len(ids) <= n else self.tok.decode(ids[:n])

    def format(self, prompt, response):
        msgs = [{"role": "user", "content": self._clip(prompt, self.MAX_PROMPT_TOK)},
                {"role": "assistant", "content": self._clip(response, self.MAX_RESP_TOK)}]
        return self.tok.apply_chat_template(msgs, tokenize=False)

    def forward(self, prompts, responses):
        texts = [self.format(p, r) for p, r in zip(prompts, responses)]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        # Call the decoder stack directly: skips the LM head (vocab-sized logits are not needed).
        lm = self.backbone.base_model.model if hasattr(self.backbone, "peft_config") else self.backbone
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = lm.model(**enc)
        h = out.last_hidden_state[:, -1].float()
        return self.head(h).squeeze(-1)

    @torch.no_grad()
    def score(self, prompts, responses, bs=32):
        self.eval()
        out = []
        for i in range(0, len(prompts), bs):
            out.append(self(prompts[i:i + bs], responses[i:i + bs]).cpu())
        return torch.cat(out).tolist()

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.backbone.save_pretrained(path)
        torch.save(self.head.state_dict(), os.path.join(path, "head.pt"))

    @classmethod
    def load(cls, path, device):
        from peft import PeftModel
        m = cls(device, lora=False)
        # frozen at use time -> merge LoRA and keep weights in bf16 (forward already ran under bf16 autocast)
        m.backbone = PeftModel.from_pretrained(m.backbone, path).merge_and_unload().to(device, dtype=torch.bfloat16)
        m.head.load_state_dict(torch.load(os.path.join(path, "head.pt")))
        return m.eval()
