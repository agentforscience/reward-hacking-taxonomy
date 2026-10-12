"""RLHF: policy-gradient fine-tuning of the policy against the learned reward model.

Objective per batch:  -E[ A(x,y) * log pi(y|x) ] + beta * KL(pi || pi_init)
REINFORCE with a per-prompt baseline (several responses are sampled per prompt and the
advantage is the reward minus that prompt's mean reward) and an exact per-token KL penalty
to the frozen initial policy, i.e. the usual RLHF objective without PPO clipping / value
head. pi is the temperature-tempered policy that is also used at evaluation time.
At every checkpoint the fixed eval prompts are answered and scored by the RM.
"""
import argparse, json, os, time, random
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from common import RewardModel, set_seed, snapshot, DATA, RESULTS, SCRATCH

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--beta", type=float, default=0.05)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--temp", type=float, default=0.7, help="policy temperature (same for training samples and eval)")
ap.add_argument("--n_samples", type=int, default=2, help="responses sampled per prompt; their mean reward is the baseline")
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--n_ckpt", type=int, default=10, help="checkpoints after step 0")
ap.add_argument("--eval_limit", type=int, default=0, help="pilot only: evaluate on every k-th eval prompt")
ap.add_argument("--max_new", type=int, default=256, help="token cap during training")
ap.add_argument("--max_new_eval", type=int, default=384, help="token cap at evaluation")
ap.add_argument("--policy", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--chunk", type=int, default=4)
ap.add_argument("--tag", default=None)
ap.add_argument("--no_eval", action="store_true")
args = ap.parse_args()
tag = args.tag or f"seed{args.seed}_beta{args.beta}"
out_dir = os.path.join(RESULTS, "rlhf", tag); os.makedirs(out_dir, exist_ok=True)
set_seed(args.seed)
dev = "cuda"
while True:                       # GPUs are shared: wait until there is room to load the models
    try:
        if torch.cuda.mem_get_info()[0] > 7 * 2**30:
            break
    except Exception:
        pass
    time.sleep(20)

tok = AutoTokenizer.from_pretrained(snapshot(args.policy)); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(snapshot(args.policy), torch_dtype=torch.float32).to(dev)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
rm = RewardModel.load(os.path.join(SCRATCH, "rm"), dev)
for p in rm.parameters():
    p.requires_grad_(False)
EOS = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]

train_prompts = json.load(open(f"{DATA}/train_prompts.json"))
eval_set = json.load(open(f"{DATA}/eval_prompts.json"))
if args.eval_limit:
    eval_set = eval_set[::args.eval_limit]
rng = random.Random(args.seed)


def retry_oom(fn, *a, **k):
    """Run fn; if a neighbouring job has taken the GPU memory, back off and retry."""
    for attempt in range(400):
        try:
            return fn(*a, **k)
        except (torch.OutOfMemoryError, torch.AcceleratorError) as e:
            if "out of memory" not in str(e).lower():
                raise
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            print(f"[oom] attempt {attempt}, waiting", flush=True); time.sleep(15)
    raise RuntimeError("GPU memory never became available")


def encode(prompts):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True) for p in prompts]
    return tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=400).to(dev)


@torch.no_grad()
def generate(prompts, temperature, top_p, max_new=None):
    """Sample responses. Returns full token matrix, prompt length, response mask, texts."""
    model.eval()
    enc = encode(prompts)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        seq = model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p, top_k=0,
                             max_new_tokens=max_new or args.max_new, pad_token_id=tok.pad_token_id, eos_token_id=EOS)
    P = enc.input_ids.shape[1]
    resp = seq[:, P:]
    is_eos = torch.zeros_like(resp, dtype=torch.bool)
    for e in EOS:
        is_eos |= resp == e
    # mask covers tokens up to and including the first EOS
    before = (is_eos.cumsum(1) - is_eos.long()) == 0
    n_tok = before.sum(1)
    truncated = ~is_eos.any(1)
    texts = [tok.decode(r[:n], skip_special_tokens=True).strip() for r, n in zip(resp, n_tok)]
    return enc, seq, P, before, texts, n_tok.tolist(), truncated.tolist()


def evaluate(step):
    """Answer the fixed eval prompts with the current policy and score them with the RM."""
    cpu_state, gpu_state = torch.get_rng_state(), torch.cuda.get_rng_state()
    torch.manual_seed(10_000 + args.seed); torch.cuda.manual_seed(10_000 + args.seed)
    order = sorted(range(len(eval_set)), key=lambda i: len(eval_set[i]["prompt"]))
    rows = []
    for i in range(0, len(order), 48):
        idx = order[i:i + 48]
        ps = [eval_set[j]["prompt"] for j in idx]
        _, _, _, _, texts, n_tok, trunc = retry_oom(generate, ps, args.temp, 0.9, args.max_new_eval)
        scores = retry_oom(rm.score, ps, texts)
        for j, t, n, tr, s in zip(idx, texts, n_tok, trunc, scores):
            rows.append(dict(tag=tag, seed=args.seed, beta=args.beta, step=step,
                             frac=step / args.steps, pid=eval_set[j]["pid"], kind=eval_set[j]["kind"],
                             response=t, n_tok=n, truncated=tr, rm=s))
    with open(os.path.join(out_dir, "eval_generations.jsonl"), "a") as f:
        for r in sorted(rows, key=lambda r: r["pid"]):
            f.write(json.dumps(r) + "\n")
    torch.set_rng_state(cpu_state); torch.cuda.set_rng_state(gpu_state)
    print(f"[eval step {step}] rm={np.mean([r['rm'] for r in rows]):.3f} "
          f"len={np.mean([r['n_tok'] for r in rows]):.0f} trunc={np.mean([r['truncated'] for r in rows]):.2f}",
          flush=True)


def train_step(prompts):
    enc, seq, P, mask, texts, n_tok, trunc = generate(prompts, args.temp, 1.0)
    r = torch.tensor(rm.score(prompts, texts), device=dev)
    # per-prompt baseline removes between-prompt reward variance (prompts are tiled n_samples times)
    adv = r.view(args.n_samples, -1) - r.view(args.n_samples, -1).mean(0, keepdim=True)
    adv = (adv / (adv.std() + 1e-6)).reshape(-1)
    attn = torch.cat([enc.attention_mask, mask.long()], 1)
    mean_len = mask.sum(1).float().mean()
    model.train(); opt.zero_grad()
    kl_tot = 0.0
    CH = args.chunk
    for c in range(0, len(prompts), CH):
        sl = slice(c, c + CH)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=seq[sl], attention_mask=attn[sl]).logits[:, P - 1:-1]
            with torch.no_grad(), model.disable_adapter():
                ref = model(input_ids=seq[sl], attention_mask=attn[sl]).logits[:, P - 1:-1]
        logp = F.log_softmax(logits.float() / args.temp, -1)
        ref_logp = F.log_softmax(ref.float() / args.temp, -1)
        m = mask[sl].float()
        tok_logp = logp.gather(-1, seq[sl, P:].unsqueeze(-1)).squeeze(-1)
        kl = (logp.exp() * (logp - ref_logp)).sum(-1)                 # exact KL(pi || pi_init) per position
        pg = -(adv[sl] * (tok_logp * m).sum(1)).sum() / (mean_len * len(prompts))
        kl_term = (kl * m).sum() / (mean_len * len(prompts))          # mean KL per token
        (pg + args.beta * kl_term).backward()
        kl_tot += kl_term.item()
        del logits, ref, logp, ref_logp, kl
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
    opt.step()
    return r, kl_tot, n_tok, trunc, gn


json.dump(vars(args), open(os.path.join(out_dir, "config.json"), "w"), indent=1)
open(os.path.join(out_dir, "eval_generations.jsonl"), "w").close()
ckpt_steps = {round(args.steps * k / args.n_ckpt) for k in range(args.n_ckpt + 1)}
train_log = []; t0 = time.time()
if not args.no_eval:
    evaluate(0)
n_prompts = args.batch // args.n_samples
n_arena = n_prompts * 3 // 4
for step in range(1, args.steps + 1):
    prompts = (rng.sample(train_prompts["arena"], n_arena) + rng.sample(train_prompts["syco"], n_prompts - n_arena)) * args.n_samples
    r, kl_tot, n_tok, trunc, gn = retry_oom(train_step, prompts)
    train_log.append(dict(step=step, reward=r.mean().item(), reward_std=r.std().item(), kl_per_tok=kl_tot,
                          len=float(np.mean(n_tok)), trunc=float(np.mean(trunc)), grad_norm=gn,
                          time=time.time() - t0))
    if step % 5 == 0:
        l = train_log[-5:]
        print(f"step {step} R={np.mean([x['reward'] for x in l]):.3f} KL/tok={np.mean([x['kl_per_tok'] for x in l]):.4f} "
              f"len={np.mean([x['len'] for x in l]):.0f} trunc={np.mean([x['trunc'] for x in l]):.2f} "
              f"gn={gn:.2f} t={time.time()-t0:.0f}s", flush=True)
        json.dump(train_log, open(os.path.join(out_dir, "train_log.json"), "w"))
    if step in ckpt_steps and not args.no_eval:
        evaluate(step)
json.dump(train_log, open(os.path.join(out_dir, "train_log.json"), "w"))
model.save_pretrained(os.path.join(SCRATCH, "policy", tag))
print("done", time.time() - t0)
