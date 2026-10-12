# Planning: Reward Hacking Taxonomy in RLHF-Trained Language Models

## Situation at start of session (deviations from the brief)

- The resource-finder stage was **skipped** (`.neurico/pipeline_state.json`: `"skipped": true`).
  There is no `literature_review.md`, `resources.md`, `papers/`, `datasets/` or `code/`.
  Background below is from my own knowledge of the literature, not a gathered review.
- `OPENROUTER_KEY` / `OPENAI_API_KEY` are **not set**, so no API models are available.
  Everything is run with real open-weight models on local GPUs (4x RTX A6000, 48 GB each).
  This actually suits the hypothesis, which is about RLHF *training dynamics* and
  cannot be tested through an inference API anyway.
- No human raters are available. "Human preference ratings" are **proxied** by blinded
  LLM judges that are larger than, and not involved in training, the policy. This is the
  single biggest validity caveat and is carried through to the report.
- `/data` has ~5 GB free. The venv uses uv `link-mode = symlink`; model artefacts go to `/tmp`.

## Motivation & Novelty Assessment

### Why This Research Matters
RLHF optimises a policy against a learned reward model (RM) that only approximates human
preferences. When the policy finds features the RM over-values (length, flattery,
markdown structure), measured reward rises while real quality stalls or falls. Knowing
*which* failure modes dominate and *when* they appear tells practitioners what to monitor
and when early stopping or regularisation is needed.

### Gap in Existing Work
Prior work documents the individual phenomena: length bias in RLHF/RMs (Singhal et al. 2023,
"A Long Way to Go"), sycophancy (Perez et al. 2022; Sharma et al. 2023), reward
over-optimisation (Gao et al. 2022; Moskovitz et al. 2023), format/markdown bias in
preference models. What is largely missing is (a) a single taxonomy applied to the *same*
training runs so that shares are comparable, and (b) per-category onset timing along the
training trajectory.

### Our Novel Contribution
A small but end-to-end real RLHF pipeline (RM trained on real human preference votes,
policy optimised against it) with dense checkpointing, where every checkpoint's outputs are
labelled with a six-category reward-hacking taxonomy and scored by both the RM and
held-out judges. This yields category shares, onset times and a proxy-vs-judge divergence
point from the same runs.

### Experiment Justification
- **Exp 1 - RM training on LMSYS Arena human votes.** Needed so the proxy reward carries
  real human-preference biases rather than biases I hand-design.
- **Exp 2 - RLHF (3 seeds), 11 checkpoints each.** The object of study; seeds separate
  systematic dynamics from run-to-run noise.
- **Exp 3 - Judge + taxonomy labelling of every checkpoint's outputs.** Gives category
  prevalence over time (H2, H3), category shares among hacking instances (H1) and the
  judge trajectory (H4).
- **Exp 4 - KL-coefficient ablation (1 seed, stronger KL).** Tests whether the timing
  conclusions are a property of RLHF or just of one regularisation strength.
- **Exp 5 - Second judge from a different model family on all outputs.** Checks that the
  "human preference" proxy is not an artefact of one judge.

## Research Question
In RLHF against a learned reward model: which failure modes make up reward hacking, when
do they appear, and when does the reward model score stop tracking (proxy) human preference?

## Hypothesis Decomposition
- **H1 (share):** verbosity + sycophancy are the primary category in > 60% of reward-hacking instances.
- **H2 (early & monotone):** verbosity and sycophancy prevalence rise above the step-0 level
  by 20% of training and increase monotonically afterwards.
- **H3 (format gaming late & saturating):** format gaming onset is later than H2's
  categories and its prevalence plateaus.
- **H4 (divergence after 40%):** judge preference tracks RM score up to 40% of steps and diverges after.

## Proposed Methodology

### Models and data (all cached locally)
| Role | Choice |
|---|---|
| Policy | `Qwen/Qwen2.5-1.5B-Instruct` + LoRA (r=16) |
| Reward model | `Qwen/Qwen2.5-0.5B-Instruct` + LoRA + scalar head, Bradley-Terry loss |
| RM training data | `lmarena-ai/arena-human-preference-55k` (real human pairwise votes; single-turn, non-tie) |
| Policy prompts | held-out Arena prompts (75%) + TriviaQA questions with a user-stated belief from `meg-tong/sycophancy-eval` (25%, half correct / half incorrect beliefs) |
| Eval prompts (fixed, disjoint) | 120 Arena + 60 trivia with *incorrect* user belief + the same 60 trivia neutral |
| Judge 1 (primary) | `Qwen/Qwen2.5-14B-Instruct` |
| Judge 2 (robustness) | `Llama-3.1-8B-Instruct` (different family) |

### RLHF algorithm
On-policy policy gradient (REINFORCE, batch-normalised advantage) with an exact per-token
KL penalty to the frozen initial policy - the same objective as PPO-based RLHF minus
clipping/value head. 200 updates x 32 episodes, sampling at T=1. Training length and
learning rate are fixed from a short pilot *before* any judge labelling is seen.
Checkpoints at 0,10,...,100% of steps. At each checkpoint the 240 eval prompts are answered
(T=0.7, top-p 0.9, fixed generation seed) and scored by the RM.

### Taxonomy (judge multi-label flags + one primary label)
V verbosity/padding - S sycophancy/flattery/unwarranted agreement - F format gaming
(structure/markup substituting for substance) - R repetition/degeneration -
H hedging/evasion/over-disclaiming - C confident fabrication or off-topic filler - N none.
Each flag is backed by a judge-independent surface metric where one exists: token length and
truncation rate (V), wrong-belief adoption on trivia probes by string match and
flattery-lexicon rate (S), markdown-element density (F), distinct-3-gram ratio (R).

### Operational definitions (fixed before looking at results)
- **Reward-hacking instance:** an eval response at checkpoint t>0 whose RM score is higher
  than the same seed's step-0 response to that prompt, while Judge 1's blinded,
  order-randomised pairwise comparison prefers the step-0 response.
- **H1:** share of instances with primary label V or S; 95% bootstrap CI resampling prompts. Supported if CI lower bound > 0.60.
- **Onset (H2/H3):** first checkpoint where flag prevalence exceeds step 0 (paired prompt-level
  bootstrap, Holm-corrected across checkpoints, pooled over seeds). "Early" = onset <= 20%.
- **Monotone:** Mann-Kendall trend test on the pooled prevalence series plus the count of decreasing steps per seed.
- **Saturation (H3):** slope of prevalence over the last 40% of training not different from 0 while slope over the first 60% is positive.
- **Divergence (H4):** earliest checkpoint where the judge metric is significantly below its
  running maximum while RM score is significantly above its value at that maximum. Also
  reported: per-checkpoint prompt-level Spearman correlation between RM score and judge rating.

### Baselines / controls
Step-0 policy (no RLHF) is the reference for every comparison. Strong-KL run as ablation.
Surface metrics act as a judge-free check on the judge's flags. Judge 2 as a check on Judge 1.

### Statistical analysis
Prompt-clustered bootstrap (10k resamples) for all CIs; alpha = 0.05; Holm correction within
each family of per-checkpoint tests; Cohen's kappa for inter-judge agreement; seeds
reported individually as well as pooled.

## Expected Outcomes
Support: V+S share CI above 60%; V and S onsets at <=20% with monotone trend; F onset later
with a flat tail; judge tracks RM until >=40% then falls. Any of these can fail
independently and will be reported as such.

## Timeline
Setup/data 30 min - RM training 20 min - pilot + RLHF runs 90 min - judging 60 min - analysis 40 min - report 30 min.

## Potential Challenges
- Policy collapse into gibberish under weak KL: pick lr/beta from pilot; R category captures it.
- Token cap censors verbosity: report truncation rate alongside length.
- Judge shares a family with the policy (Qwen): Judge 2 is Llama.
- "% of training steps" depends on the arbitrary total step count: ablation plus explicit caveat.

## Success Criteria
All four sub-hypotheses receive a clear supported / not supported / inconclusive verdict
backed by CIs, with raw generations and labels saved for inspection.

## Amendments made during execution (before any judge labels were seen)

1. **Compute.** Other jobs on the machine occupied most of all four GPUs for the whole
   session, leaving 5-25 GB on any one card. Consequences: policy is
   `Qwen2.5-0.5B-Instruct` (not 1.5B); 16 episodes per update; primary judge is
   `Qwen2.5-7B-Instruct` (the 14B judge does not fit); Judge 2 (Llama-3.1-8B) and the
   ablation run are judged at checkpoints 0/20/40/60/80/100% only.
2. **First RLHF attempt discarded** (`results/attempt1_unstable_lr1e-4/`). With lr 1e-4,
   a batch-mean baseline and T=1 sampling, the *training* reward itself fell below its
   starting value after ~50 updates while KL to the initial policy grew to 0.3 nats/token
   in all runs: noise-driven drift, i.e. an optimisation failure rather than reward
   hacking. The runs were stopped at 35-45% and are not analysed.
3. **Revised RLHF objective.** Two responses are sampled per prompt and the advantage is
   reward minus the prompt's mean reward (removes between-prompt reward variance); the
   policy is the T=0.7 tempered distribution for both training samples and evaluation;
   lr 3e-5. A 48-step pilot at lr 2e-5 and 5e-5 (seed 99, `results/pilots/`) showed
   steadily rising training reward for both; 3e-5 was fixed for all reported runs.
   Token cap: 256 in training, 384 at evaluation. 160 updates, checkpoints every 16.
4. **Judge protocol changes** (after inspecting the first ~700 judge outputs for format and
   position bias only): (a) the rating prompt now asks for a list of present problems
   instead of six 0/1 fields, because the 7B judge dropped fields in 28% of replies;
   (b) the judge picked "A" in 72% of pairwise comparisons, so every pair is now judged in
   both orders. `prefers_new` is the mean of the two verdicts (0, 0.5 or 1) and a
   reward-hacking instance requires the judge to prefer the step-0 response in **both** orders.
