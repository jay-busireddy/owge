# OWGE POMDP + Information Theory + LoRA Validation Package

This package is designed for a laptop/CPU-first empirical sister paper to the published OWGE conceptual preprint.

## What is included

- `owge_experiment.py` — complete experiment runner
- `requirements.txt` — Python dependencies
- `EXPERIMENT_DESIGN.md` — scientific rationale and hypothesis mapping
- `sample_test_set/` — 12 fixed held-out-style episodes (72 PNG frames) plus `test_manifest.csv`
- `run_smoke_windows.bat` — minimal Windows verification run
- `run_laptop_windows.bat` — intended laptop experiment

The Python program automatically generates train/validation/test episodes, runs the five observers, produces CSV results, performs paired hypothesis tests, and generates **18 separate plots** when all sweeps are enabled.

## 1. Create a Python environment on Windows

Open PowerShell or Command Prompt in this folder.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, either use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

or run the `.venv\Scripts\python.exe` executable directly.

## 2. First run the smoke preset

```powershell
python owge_experiment.py --preset smoke --output owge_smoke --device auto
```

The smoke preset is only a code/system check. **Do not use smoke statistics in a paper.** Its seed count and training budget are intentionally tiny.

Expected folders:

```text
owge_smoke/
  data_preview/
  plots/
  results/
  run_config.json
```

## 3. Run the laptop experiment

```powershell
python owge_experiment.py --preset laptop --output owge_laptop --device auto
```

`--device auto` uses CUDA if PyTorch can see an NVIDIA GPU, Apple MPS on supported Macs, otherwise CPU.

The laptop preset uses five matched main seeds. Treat that as an exploratory/pilot study. The final paper should report enough independent seeds to stabilize confidence intervals/effect sizes; the `paper` preset currently uses 20 matched seeds.

## 4. Stronger run for paper-quality statistics

```powershell
python owge_experiment.py --preset paper --output owge_paper --device auto
```

You can also change the number of main seeds without editing the file:

```powershell
python owge_experiment.py --preset paper --seed-count 10 --output owge_10seeds --device auto
```

## 5. If you only want the main five-observer run

```powershell
python owge_experiment.py --preset laptop --skip-sweeps --output owge_main_only --device auto
```

The distractor evaluation is still generated, but the information sweep and reversal training are skipped.

## Base model

The default model is intentionally much smaller than a normal pretrained vision model. The verified code reports about **12,074 parameters** before freezing. Only low-rank adapters are trainable for each observer.

This is preferable for the first causal experiment because every model is completely controlled and pretraining leakage is impossible.

If a later external-validity study needs a public pretrained backbone, TorchVision MobileNetV3-Small is a reasonable laptop-class comparison, but it is **not required for this first study**.

## Generated plots

With sweeps enabled the code creates these separate plots:

1. held-out accuracy by observer
2. reward by observer
3. OWGE+ by observer
4. attention calibration
5. delayed-cue retention
6. resource cost
7. LoRA update norm
8. validation learning curves
9. OWGE vs held-out transfer
10. controlled peripheral information
11. accuracy vs conditional peripheral information
12. O3-O1 crossover vs information
13. distractor robustness
14. O2 exhaustive-observer efficiency vs distractors
15. reversal recovery O3 vs O4
16. LoRA update cosine-similarity heat map
17. false-positive/false-negative composition
18. POMDP vs fully observable MDP control

## Files to send back to ChatGPT

After the run, zip the entire output directory, for example on PowerShell:

```powershell
Compress-Archive -Path owge_laptop\* -DestinationPath owge_laptop_results.zip
```

Send the ZIP back. The most important files are:

- `results/metrics.csv`
- `results/hypothesis_tests.csv`
- `results/information_theory.csv`
- `results/mi_sweep.csv`
- `results/distractor_sweep.csv`
- `results/reversal.csv`
- `results/mdp_control_training.csv`
- `results/learning_curves.csv`
- `results/lora_geometry.csv`
- all files under `plots/`
- `run_config.json`

Those results can then be audited before any claim is written, and the sister-paper LaTeX can be generated from the actual measured results.

## Important interpretation rule

A different LoRA update is **not** itself evidence of higher smartness. Null hypotheses are tested with matched held-out behavior/OWGE/resource measures. LoRA geometry is secondary evidence about how adaptation differed internally.
