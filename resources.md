# Resources

The automated resource-finding stage was skipped for this run, so nothing was pre-gathered. Everything below was taken from the machine's local Hugging Face cache (`/data/transformers/hub`).

## Datasets
| Dataset | Use |
|---|---|
| `lmarena-ai/arena-human-preference-55k` | 57,477 human pairwise votes. After filtering to single-turn, non-tie, mostly-ASCII rows with unique prompts: 28,319. 20,000 used to train the RM, 1,500 for validation, 4,000 held out as the policy/eval prompt pool. |
| `meg-tong/sycophancy-eval` (`answer.jsonl`, TriviaQA part) | 600 questions with a user-stated belief for training prompts; 60 questions (wrong-belief and neutral variants) for evaluation. |

## Models
| Model | Use |
|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | Policy; reward-model backbone |
| `Qwen/Qwen2.5-7B-Instruct` | Primary judge |
| `unsloth/Meta-Llama-3.1-8B-Instruct` | Second judge |

## Process notes
- No API keys were available; no API calls were made.
- GPUs were shared with other jobs throughout, which determined model sizes and judge coverage.
- A first RLHF configuration was unstable and discarded; see `planning.md` (Amendments) and REPORT.md section 3.6.
