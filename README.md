# Review Artifact for COMNET-D-26-02608

This repository is a lightweight review artifact for the revised Computer Networks submission. It contains the scripts, aggregate outputs, generated LaTeX tables, and figure sources used to support the major-revision experiments.

The artifact is intended for peer review. It avoids manuscript metadata, reviewer files, and large raw packet/window traces. The included scripts can regenerate the omitted raw traces from the stated seed lists.

## Contents

- `scripts/`: DES simulation, aggregation, learned tail-scale calibration, revision audits, coefficient sensitivity, observable-descriptor audit, frequency-identifiability audit, non-stationary shift audit, and figure generation scripts.
- `scripts/ns3/`: NS-3 runner, scratch scenario, packet-trace postprocessing, held-out evaluation, and seed-pair audit scripts.
- `outputs/des_20seed_summary/`: 20-seed DES aggregate CSV files and generated table snippets.
- `outputs/revision_audits/`: statistical, sensitivity, observable-descriptor, frequency-identifiability, risk-weight, and non-stationary audit outputs.
- `outputs/ns3_10seed_summary/`: 10-seed NS-3 window-level summaries, held-out metrics, and seed-fold metrics.
- `tables/`: LaTeX table snippets used in the revised manuscript.
- `figures/`: PGFPlots/TikZ figure sources used to generate the revised manuscript figures.

## Environment

The DES and audit scripts use Python 3.10+ and the Python standard library. No Python package installation is required for the included DES/audit scripts.

Optional components:

- NS-3.47 is required to regenerate packet traces.
- XeLaTeX with PGFPlots is required to compile the figure sources into PDFs.

## Seed Lists

DES main and audit seeds:

```text
7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83
```

NS-3 packet-trace seeds:

```text
7, 11, 13, 17, 19, 23, 29, 31, 37, 41
```

## Reproducing the DES Summaries

From the repository root, run the following commands. The full 20-seed DES run can take time because it generates per-window predictions for every seed.

PowerShell example:

```powershell
$seeds = @(7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83)
foreach ($seed in $seeds) {
  python scripts\run_simulation.py --preset main --seed $seed --margin-scale 2.2 --include-ablations --outdir "outputs\des_seed${seed}_margin22"
}
python scripts\aggregate_results.py --pattern "outputs/des_seed*_margin22" --outdir outputs/des_20seed_summary
python scripts\revision_audits.py --seed-pattern "outputs/des_seed*_margin22" --outdir outputs/revision_audits --table-dir tables/des
```

Additional revision audits:

```powershell
$seedText = "7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83"
python scripts\coefficient_sensitivity.py --seeds $seedText --outdir outputs/revision_audits --table-dir tables/des
python scripts\model_blind_descriptors.py --seeds $seedText --outdir outputs/revision_audits --table-dir tables/des
python scripts\identifiability_audit.py --seeds $seedText --outdir outputs/revision_audits --table-dir tables/des
python scripts\nonstationarity_shift_audit.py --seeds $seedText --outdir outputs/revision_audits --table-dir tables/des
```

## Reproducing the NS-3 Summaries

The included NS-3 scenario file is `scripts/ns3/uav_vehicular_vacation.cc`. To regenerate packet traces, copy it into the NS-3.47 `scratch/` directory or configure the runner to use an NS-3 tree containing that scratch file.

PowerShell example, assuming `NS3_ROOT` points to the NS-3.47 root:

```powershell
$seeds = @(7,11,13,17,19,23,29,31,37,41)
$scenarios = @("overall","load_high","vacation_high","drift_strong","traffic_mix_video_heavy","traffic_mix_c2_heavy")
powershell -ExecutionPolicy Bypass -File scripts\ns3\run_ns3_experiment.ps1 -Ns3Root $env:NS3_ROOT -Seeds $seeds -Scenarios $scenarios -Target video -Duration 120 -OutDir outputs\ns3_validation_extended
python scripts\ns3\postprocess_ns3.py --packet-glob "outputs/ns3_validation_extended/packets/*_packets.csv" --outdir outputs/ns3_validation_extended
python scripts\ns3\evaluate_ns3_heldout.py --window outputs/ns3_validation_extended/ns3_window_metrics.csv --outdir outputs/ns3_validation_extended --train-seeds 7,11,13,17,19,23,29,31 --test-seeds 37,41
python scripts\ns3\evaluate_ns3_seedfold.py --window outputs/ns3_validation_extended/ns3_window_metrics.csv --outdir outputs/ns3_validation_extended --table-dir tables/ns3
```

## Raw Trace Policy

The lightweight artifact includes aggregate CSVs, audit CSVs, generated table snippets, and NS-3 window-level outputs. Large raw DES per-window traces and raw NS-3 packet traces are omitted from the GitHub-ready folder to keep the review artifact compact. They can be regenerated using the scripts and commands above, or provided as a separate archive/release if requested by the editor or reviewers.

## Suggested Citation in the Manuscript

Use the statement in `REVIEW_ARTIFACT_STATEMENT.md` for manuscript and response-letter wording.
