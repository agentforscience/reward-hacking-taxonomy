"""Train the reward model on human pairwise votes with a Bradley-Terry loss."""
import json, os, sys, time, random
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy.stats import spearmanr
from common import RewardModel, set_seed, DATA, RESULTS, SCRATCH

set_seed(42)
dev = "cuda"
N_TRAIN = int(os.environ.get("RM_N_TRAIN", 20000)); BS = 8; LR = 1e-4
train = pd.read_json(f"{DATA}/rm_train.jsonl", lines=True).iloc[:N_TRAIN]
val = pd.read_json(f"{DATA}/rm_val.jsonl", lines=True)
rm = RewardModel(dev)
params = [p for p in rm.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
steps = len(train) // BS
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)


def evaluate():
    sa = np.array(rm.score(val.prompt.tolist(), val.a.tolist()))
    sb = np.array(rm.score(val.prompt.tolist(), val.b.tolist()))
    acc = float(((sa > sb) == (val.winner == "a").values).mean())
    la, lb = val.a.str.len().values, val.b.str.len().values
    return dict(val_acc=acc,
                # how often "prefer the longer response" agrees with humans / with the RM
                human_prefers_longer=float(((la > lb) == (val.winner == "a").values).mean()),
                rm_prefers_longer=float(((la > lb) == (sa > sb)).mean()),
                spearman_score_len=float(spearmanr(np.r_[sa, sb], np.r_[la, lb])[0]),
                score_mean=float(np.r_[sa, sb].mean()), score_std=float(np.r_[sa, sb].std()))


log = []; t0 = time.time()
for s in range(steps):
    rm.train()
    b = train.iloc[s * BS:(s + 1) * BS]
    win = [r.a if r.winner == "a" else r.b for r in b.itertuples()]
    lose = [r.b if r.winner == "a" else r.a for r in b.itertuples()]
    sw, sl = rm(b.prompt.tolist(), win), rm(b.prompt.tolist(), lose)
    loss = -F.logsigmoid(sw - sl).mean() + 1e-3 * (sw ** 2 + sl ** 2).mean()  # small L2 keeps scale bounded
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
    log.append(loss.item())
    if (s + 1) % 100 == 0:
        print(f"step {s+1}/{steps} loss {np.mean(log[-100:]):.4f} {time.time()-t0:.0f}s", flush=True)
res = evaluate()
res.update(n_train=len(train), steps=steps, lr=LR, batch=BS, final_train_loss=float(np.mean(log[-100:])))
print(res)
rm.save(os.path.join(SCRATCH, "rm"))
json.dump(res, open(f"{RESULTS}/rm_eval.json", "w"), indent=1)
