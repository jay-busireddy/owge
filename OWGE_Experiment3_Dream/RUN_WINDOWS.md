# Windows laptop run instructions

## 1. Preserve previous experiments

Do not overwrite your old folders such as:
- owge_rho_tuning
- owge_confirmatory_full

Experiment 3 must use a new output directory.

## 2. Create a clean Python environment

Open a normal PowerShell (preferably not Anaconda Prompt) in this package directory.

```powershell
py -m venv .venv_owge3
.\.venv_owge3\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Test the OpenMP/import environment:

```powershell
python -c "import torch,numpy,scipy,pandas,matplotlib; print('environment OK', torch.__version__)"
```

Do not use KMP_DUPLICATE_LIB_OK=TRUE for scientific runs.

## 3. Syntax check

```powershell
python -m py_compile owge_dream_experiment.py
```

No output means success.

## 4. Optional smoke run

This verifies code only; never use its p-values as evidence.

```powershell
python owge_dream_experiment.py --preset smoke --output owge_dream_smoke --device auto
```

Expected outputs include results CSVs and >20 plots.

## 5. Final confirmatory Experiment 3

Run exactly:

```powershell
python owge_dream_experiment.py --preset confirmatory --output owge_dream_confirmatory_final --device auto
```

The default confirmatory peripheral reserve is rho=0.50 and is frozen in this package. Do not change rho after seeing results.

Do not stop the run based on intermediate results or p-values.

## 6. If the process is interrupted

The script saves incremental results and observer checkpoints. Resume with the SAME output directory:

```powershell
python owge_dream_experiment.py --preset confirmatory --output owge_dream_confirmatory_final --device auto --resume
```

Use --resume only to finish the exact prespecified run, not to alter settings.

## 7. What to return

When complete, verify these exist:

- owge_dream_confirmatory_final/run_config.json
- owge_dream_confirmatory_final/results/metrics.csv
- owge_dream_confirmatory_final/results/observer_summary.csv
- owge_dream_confirmatory_final/results/hypothesis_tests.csv
- owge_dream_confirmatory_final/results/information_theory.csv
- owge_dream_confirmatory_final/results/lora_geometry.csv
- owge_dream_confirmatory_final/results/learning_curves.csv
- owge_dream_confirmatory_final/results/SUMMARY.txt
- owge_dream_confirmatory_final/plots/ (more than 20 plots)

Zip the entire folder:

```powershell
Compress-Archive -Path .\owge_dream_confirmatory_final\* -DestinationPath .\owge_dream_confirmatory_final.zip
```

Upload that ZIP to ChatGPT. Do not edit/filter the CSV files before upload.

## 8. Statistical interpretation

Primary test:
H0: mean Phase-C balanced accuracy(O4D) <= mean Phase-C balanced accuracy(O1).
HA: mean Phase-C balanced accuracy(O4D) > mean Phase-C balanced accuracy(O1).

- If one-sided p < 0.05 and the effect is in the expected direction: reject H0.
- If p >= 0.05: fail to reject H0. This does not prove H0.
- We will also inspect the paired 95% CI and paired effect size.

Secondary hypotheses are Holm-corrected.

Do not rerun with changed epochs, rho, memory weight, dream frequency, or seeds merely because the p-value is unfavorable. A changed configuration would be a new experiment.
