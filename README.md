# Belief-Anchored Decoding reduces deception in Large Language Models

Inference-time interventions that reduce **self-initiated deception** in open-weight LLMs, evaluated on the Contact Searching Question (CSQ) framework of Wu et al. (ICLR 2026).

This repository builds directly on top of:

> Z. Wu, M. Du, S.-K. Ng, B. He.
> *Beyond Prompt-Induced Lies: Investigating LLM Deception on Benign Prompts*.
> ICLR 2026.
> Code: <https://github.com/Xtra-Computing/LLM-Deception>

We extend their evaluation framework with **four inference-time mitigation strategies** — three prompt-level (Forced Enumeration, Sample-and-Vote, Belief-First) and one logit-level (**Belief-Anchored Decoding, BAD**) — across six open-weight LLMs (2B–32B). All evaluation uses Wu et al.'s exact protocol: bias-corrected deception scores δ (Wu Eq. 2) and ρ (Wu Eq. 1).

## Headline contribution: Belief-Anchored Decoding (BAD)

For a complex Broken-list question, BAD first extracts the model's belief over Yes/No for each consecutive edge of the implied chain (one forward pass per edge). During greedy decoding of the long question, a custom `LogitsProcessor` modifies next-token logits by adding a **product-of-experts** of the edge beliefs:

```
Logit_align(y) = Logit_expr(y)  +  λ · Σ_i log P_belief^(i)(y)
```

A single broken edge with `P_belief(Yes) ≈ 0` produces a strong negative penalty on the "Yes" token, pulling expression toward belief. λ is selected per model to minimise δ subject to accuracy ≥ baseline − 5%. The implementation is polarity-aware (handles "can"/"cannot" phrasings), so the same processor works across all four CSQ question categories.

## Repository layout

```
.
├── main.py                       # generate questions, run baselines/interventions, analyze
├── configs/models.yaml           # model registry (HF IDs, vLLM ports)
├── inference/                    # vLLM client + prompt-level interventions
├── methods/                      # logit-level methods (BAD lives here)
│   ├── method3_bad.py            # Method 3: Belief-Anchored Decoding
│   └── common.py                 # shared utilities, model registry, metrics
├── runs/                         # remaining experiments for the paper (this work)
│   ├── run_bad_remaining_qtypes.py
│   ├── run_intervention_c_linked.py
│   ├── run_compute_matched_voting.py
│   ├── run_sycophancy_prompt.py
│   ├── run_sycophancy_bad.py
│   ├── compute_bootstrap_cis.py
│   ├── run_all.sh                # sequential master pipeline (single GPU)
│   └── run_parallel_gpu1.sh      # second-GPU helper for parallelization
├── analysis/                     # δ/ρ metrics, plotting
├── questions/generate.py         # CSQ question generator (imports from paper_repo/)
├── paper/, proposal.tex          # LaTeX source
├── launch_servers.sh             # vLLM launcher (one model at a time)
└── requirements.txt
```

## Reproducing the experiments

The full pipeline is designed for a **single 32 GiB GPU** (RTX 5090 in the paper). Two GPUs roughly halve wall-clock time.

### 0. Prerequisites

- Linux + CUDA 12.x
- Python **3.10**
- ≥ 1 NVIDIA GPU with **≥ 32 GiB VRAM** (RTX 5090, A100-40G, RTX 6000 Ada, etc.)
- ~ 250 GiB free disk for HF weights cache + experimental results
- A Hugging Face account with access to gated models (Gemma-2, Llama-3.x). Run `huggingface-cli login` once.

### 1. Clone

```bash
git clone https://github.com/Mahyar-GH79/Belief-Anchor_Decoding-reduces-deception-in-Large-Language-Models.git
cd Belief-Anchor_Decoding-reduces-deception-in-Large-Language-Models

# CSQ question generation imports from Wu et al.'s repo. Clone it
# alongside the codebase as `paper_repo/`:
git clone https://github.com/Xtra-Computing/LLM-Deception.git paper_repo
```

### 2. Create the virtual environment

```bash
python3.10 -m venv env
source env/bin/activate
pip install --upgrade pip

# Core analysis / inference deps
pip install -r requirements.txt

# Heavy deps (not in requirements.txt because they vary by CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes
pip install vllm  # 0.17.x or later
```

Verify:

```bash
python -c "import vllm, torch, transformers; \
           print('vllm', vllm.__version__, '| torch', torch.__version__, \
                 '| cuda', torch.cuda.is_available())"
```

### 3. Generate the CSQ question dataset

```bash
python main.py generate --n-questions 200 --seed 42
```

This creates `questions/{Linked,LinkedReverse,Broken,BrokenReverse}_n{5,10,20,40,80}.json` — 200 questions per (qtype, n) cell, 4 000 total.

### 4. Baseline + prompt-level interventions on each of the 6 models

The reported model fleet is:

| Model | HF ID | vLLM port |
|---|---|---|
| `gemma-2-9b-it` | `google/gemma-2-9b-it` | 8002 |
| `Qwen2.5-7B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` | 8008 |
| `Qwen2.5-32B-Instruct` (AWQ for vLLM) | `Qwen/Qwen2.5-32B-Instruct-AWQ` | 8003 |
| `Meta-Llama-3.1-8B-Instruct` | `meta-llama/Llama-3.1-8B-Instruct` | 8001 |
| `Mistral-Nemo-Instruct-2407` | `mistralai/Mistral-Nemo-Instruct-2407` | 8004 |
| `Phi-4-mini-instruct` | `microsoft/Phi-4-mini-instruct` | 8010 |

For each model, launch vLLM in one terminal and run the 5 conditions (`baseline`, `intervention_a`, `intervention_b`, `intervention_c`, `intervention_ab`) in another:

```bash
# Terminal A: launch vLLM (one model at a time on a 32 GiB GPU)
bash launch_servers.sh gemma         # then qwen7b, llama, mistral, phi4mini, qwen32b

# Terminal B: drive the experiments
python main.py run --model gemma-2-9b-it --condition all
```

Outputs land in `results/<model>/<condition>_<qtype>_n<n>.json`.

### 5. Method 3 (BAD): initial λ-sweep on Broken

This step does the per-model λ-selection on Broken-list questions (Wu Eq. 2 δ_pos) and writes `plots/methods/method3_bad/summary.json` with the chosen λ per model. **No vLLM is used here** — BAD relies on a custom `LogitsProcessor`, run via Hugging Face Transformers. Make sure the GPU is free.

```bash
python -m methods.method3_bad --device 0 --models \
    gemma-2-9b-it Qwen2.5-7B-Instruct Qwen2.5-32B-Instruct \
    Meta-Llama-3.1-8B-Instruct Mistral-Nemo-Instruct-2407 Phi-4-mini-instruct
```

### 6. Remaining experiments (this work)

After step 5, the following protocol gaps remain:
- BAD on **LinkedReverse + BrokenReverse** — needed for bias-corrected δ (Wu Eq. 2)
- BAD on **Linked** — sanity check (BAD must not collapse correct-Yes accuracy)
- Intervention C on **Linked + LinkedReverse** — Wu's belief-first protocol does not naturally apply to Linked questions, so we synthesize a single-edge probe matched to the question's phrasing
- **Compute-matched** Sample-and-Vote (k = n+1) — isolates BAD's algorithmic contribution from its compute budget
- **Sycophancy / incentivizing-prompt robustness** (Wu §5.4) on baseline + C + BAD

These are filled in by the `runs/` driver:

```bash
bash runs/run_all.sh
```

`run_all.sh` is **idempotent** — every Python script skips a (model, qtype, n) cell whose output JSON already exists. Safe to interrupt and restart.

For two-GPU parallelization, in a second terminal on a free GPU:

```bash
bash runs/run_parallel_gpu1.sh   # picks up the slowest models first
```

Per-script invocation is also supported:

```bash
python -m runs.run_intervention_c_linked   --model Phi-4-mini-instruct --port 8010
python -m runs.run_compute_matched_voting  --model Phi-4-mini-instruct --port 8010
python -m runs.run_sycophancy_prompt       --model Phi-4-mini-instruct --port 8010
python -m runs.run_bad_remaining_qtypes    --model Phi-4-mini-instruct --device 0
python -m runs.run_sycophancy_bad          --model Phi-4-mini-instruct --device 0
```

### 7. Analysis: bias-corrected δ and ρ with bootstrap 95% CIs

```bash
python -m runs.compute_bootstrap_cis
```

Produces `analysis/bootstrap_cis.json` with 1000-sample bootstrap mean and 95% CI for both Wu Eq. 1 (ρ) and Wu Eq. 2 (δ) for every (model, condition, n) cell.

### 8. (Optional) Generate paper figures

```bash
python plot_final_paper.py
python main.py analyze --target-model Phi-4-mini-instruct
```

## Hardware and timing notes

A single RTX 5090 (32 GiB) is sufficient for every step:

- Phase A (vLLM-batched, prompt-level interventions): throughput-bound, fast. ~1–4 h per model for the full grid.
- Phase B (HF + custom `LogitsProcessor` for BAD): single-stream, ~3–8 h per medium model for the full λ-sweep + remaining qtypes.

End-to-end on one GPU: roughly 4–6 days continuous. With a second GPU running `runs/run_parallel_gpu1.sh`, ~2.5–3.5 days.

Qwen-2.5-32B specifics:
- vLLM uses AWQ 4-bit (HF ID `Qwen/Qwen2.5-32B-Instruct-AWQ`).
- BAD scripts use bitsandbytes NF4 4-bit on the base model (`Qwen/Qwen2.5-32B-Instruct`); both fit in 32 GiB.

## Sanity check after BAD finishes

The polarity-aware BAD processor is the most novel piece of new logic. After it produces results, verify it does not collapse Linked accuracy or BrokenReverse accuracy:

```bash
python3 -c "
import json, os
for m in ['gemma-2-9b-it','Qwen2.5-7B-Instruct','Qwen2.5-32B-Instruct',
         'Meta-Llama-3.1-8B-Instruct','Mistral-Nemo-Instruct-2407','Phi-4-mini-instruct']:
    print(f'\n{m}')
    for qt in ['Linked','LinkedReverse','Broken','BrokenReverse']:
        accs = []
        for n in [5,10,20,40,80]:
            p = f'results_methods/method3_bad/{m}/best_lambda_{qt}_n{n}.json'
            if not os.path.exists(p): continue
            r = json.load(open(p))
            v = [x for x in r if x.get('initial_answer')]
            if v:
                a = sum(1 for x in v if x['initial_is_correct'])/len(v)
                accs.append(f'n={n}:{a:.2f}')
        print(f'  BAD {qt:14s} {\" \".join(accs)}')
"
```

What you want: BAD-Linked accuracy stays high (≥ 0.85 for most models) and BAD-LinkedReverse accuracy is comparable to the LinkedReverse baseline. Sharp drops on Linked or LinkedReverse indicate the polarity logic is inverted.

## Citation

If you use this code, please cite the parent paper:

```bibtex
@inproceedings{wu2026beyond,
    title = {Beyond Prompt-Induced Lies: Investigating LLM Deception on Benign Prompts},
    author = {Wu, Zhaomin and Du, Mingzhe and Ng, See-Kiong and He, Bingsheng},
    booktitle = {International Conference on Learning Representations},
    year = {2026}
}
```
