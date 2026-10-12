# Reward Hacking Taxonomy in RLHF-Trained Language Models

A small end-to-end RLHF experiment run locally: a reward model trained on real human preference votes (LMSYS Arena), a Qwen2.5-0.5B policy optimised against it (3 seeds + a strong-KL ablation, 11 checkpoints each), and every checkpoint's outputs labelled with a six-category reward-hacking taxonomy by blinded LLM judges plus judge-free surface metrics. Full details: [REPORT.md](REPORT.md).

## Key findings

- **Verbosity + sycophancy share of reward-hacking instances: 49.9% (95% CI 45.3-54.3), not > 60%.** Nearly all of it is verbosity; sycophancy is under 1%. The figure is judge-dependent (80.7% with a second judge that calls almost everything verbose).
- **Verbosity emerges by 20% of training** (+22 tokens, significant) but plateaus after 50-60% instead of rising monotonically. **Sycophancy did not emerge** on any of three measures.
- **Format changes come first, not later**: markdown use rises by 10%, peaks at 50%, then falls.
- **RM score and judge preference diverge detectably at about 40%** of steps (60% on one of four judge metrics); judges never clearly preferred the RLHF policy before that.
- **Fabrication** is the label most enriched among reward-hacking instances (34.5% vs 11.5% at step 0).

Caveats: 0.5B policy, LLM judges instead of humans, poor inter-judge agreement on most taxonomy labels, run length chosen arbitrarily. See REPORT.md section 6.

## Reproduce

```bash
uv venv && source .venv/bin/activate && uv sync      # torch, transformers, peft, pandas, scipy, matplotlib
cd src
python prepare_data.py                                # builds data/ from the HF cache
python train_rm.py                                    # reward model -> $RH_SCRATCH/rm (default /tmp/rh_taxonomy)
for s in 0 1 2; do python rlhf.py --seed $s --beta 0.05 --steps 160 --lr 3e-5 --chunk 2; done
python rlhf.py --seed 0 --beta 0.5 --steps 160 --lr 3e-5 --chunk 2
python judge.py --judge qwen7b  --tags seed0_beta0.05,seed1_beta0.05,seed2_beta0.05
python judge.py --judge qwen7b  --tags seed0_beta0.5 --steps 0,20,40,60,80,100
python judge.py --judge llama8b --tags seed0_beta0.05,seed1_beta0.05,seed2_beta0.05 --steps 0,20,40,60,80,100
python analyze.py                                     # results/analysis.json, results/tables, figures/
```

Models and datasets are read from the local Hugging Face cache (`HF_HOME`); no API keys are used. One 24 GB GPU is enough. `analyze.py` is deterministic; training is seeded but not bitwise reproducible. `judge_loop.sh` is the helper used to run judges on shared GPUs while training was still in progress.

## Layout

| Path | Content |
|---|---|
| `planning.md` | Pre-specified plan, definitions and the amendments made during execution |
| `REPORT.md` | Full report |
| `src/` | `common.py`, `prepare_data.py`, `train_rm.py`, `rlhf.py`, `judge.py`, `judge_loop.sh`, `analyze.py` |
| `data/` | RM pairs, training prompts, fixed eval prompts |
| `results/` | Raw generations, judge outputs, tables, `analysis.json`, discarded first attempt and pilots |
| `figures/` | Six figures |
| `logs/` | Run logs |
