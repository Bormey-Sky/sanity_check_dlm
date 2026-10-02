# A Sanity Check for Feature Attribution on a Masked Diffusion Language Model

Code and results for the term paper *A Sanity Check for Feature Attribution on a Masked Diffusion Language Model* (Bormey Chanchem, Trier University).

The paper tests whether three feature-attribution methods (Saliency, Integrated Gradients and Occlusion) detect a **verified political-bias injection** in MDLM-169M. It compares a base checkpoint with four LoRA-injected checkpoints (left and right, at 1,000 and 5,000 articles) on the 62 Political Compass Test statements. All methods are benchmarked against a random-attribution control.


## Setup

**Hardware.** MDLM needs an Ampere-or-newer GPU (A100 or L4). It fails on a T4 because it depends on FlashAttention. The results were produced on a Colab A100.

```bash
pip install -r requirements.txt
```

`flash-attn` is not in `requirements.txt` because there is no official PyPI wheel for torch 2.11. Install a wheel that matches your exact Python/CUDA/torch versions from
<https://github.com/lesj0610/flash-attention/releases/tag/v2.8.3-cu12-torch2.11>.
Confirmed working: Python 3.13, torch 2.11.0+cu128, CUDA 12.8, flash-attn 2.8.3 (`cp313` wheel).

## Checkpoints

The four LoRA adapters were trained and verified in prior work. This repository reuses them unchanged and performs no finetuning. The finetuning code, the held-out NormPLL verification and the training details (rank 16 / 3 epochs for 1k, rank 32 / 5 epochs for 5k) are in the poster repository:
<https://github.com/Bormey-Sky/TNLP_poster_bias_diff_ar>

Pass each adapter directory to `main.py` with `--ckpt`. The base model is downloaded from the Hugging Face Hub (`kuleshov-group/mdlm-owt`).

## Reproducing the results

**1. Attribution runs** (5 checkpoints x 4 methods = 20 CSVs in `results/`):

```bash
for METHOD in saliency integrated_gradients occlusion random_baseline; do
  python main.py --method $METHOD --direction base  --dose base
  python main.py --method $METHOD --direction left  --dose 1k --ckpt <path>/left_1k
  python main.py --method $METHOD --direction left  --dose 5k --ckpt <path>/left_5k
  python main.py --method $METHOD --direction right --dose 1k --ckpt <path>/right_1k
  python main.py --method $METHOD --direction right --dose 5k --ckpt <path>/right_5k
done
```

Saliency, IG and Occlusion are deterministic. The random control is seeded per checkpoint (`RANDOM_SEED + offset` in `main.py`), so its draws differ between checkpoints but are reproducible.

**2. Tables and figures.** Open `Advtopic_sanity_check_ar_diffusion.ipynb`. Part B reads the CSVs in `results/` and reproduces Tables 2-4 and Figures 2-3 of the paper. It runs on CPU in a few seconds and is saved with its outputs, so it can simply be viewed.

The random-control CSVs were not kept after the original Colab session. The notebook falls back to their fractions saved in `results/diagnostic_summary.json`. Re-running the five `random_baseline` commands above restores the CSVs, and the notebook then uses them directly.

## Reporting convention

Results are fractions of the 62 statements for which a pre-specified ordering held. They are compared with the random control, not with a significance threshold (see "Diagnostic Tests" in the paper's Methodology section).
