# Paper 1 Simulation Experiments

This directory contains the first reproducible experiment harness for the
learning-augmented queueing paper.

The initial runner is intentionally dependency-free. It uses only the Python
standard library and produces CSV/JSON outputs that can later be consumed by a
plotting script or the paper draft.

## Quick smoke run

```powershell
python .\experiments\run_simulation.py --preset quick --seed 7 --outdir .\paper1_draft\experiment_outputs\smoke_001
```

## Larger run

```powershell
python .\experiments\run_simulation.py --preset main --seed 7 --margin-scale 2.2 --outdir .\paper1_draft\experiment_outputs\main_seed7_margin22
```

## Multi-seed run

```powershell
foreach ($s in 7,11,13,17,19) {
  python .\experiments\run_simulation.py --preset main --seed $s --margin-scale 2.2 --outdir ".\paper1_draft\experiment_outputs\main_seed${s}_margin22"
}
```

Aggregate the resulting summaries:

```powershell
python .\experiments\aggregate_results.py --pattern "paper1_draft/experiment_outputs/main_seed*_margin22" --outdir .\paper1_draft\experiment_outputs\multiseed_margin22
```

## Ablation run

```powershell
foreach ($s in 7,11,13,17,19) {
  python .\experiments\run_simulation.py --preset main --seed $s --margin-scale 2.2 --include-ablations --outdir ".\paper1_draft\experiment_outputs\ablation_seed${s}_margin22"
}
```

Aggregate the ablation summaries:

```powershell
python .\experiments\aggregate_results.py --pattern "paper1_draft/experiment_outputs/ablation_seed*_margin22" --outdir .\paper1_draft\experiment_outputs\ablation_multiseed_margin22
```

## Outputs

- `summary_metrics.csv`: aggregate metrics by scenario and method.
- `window_predictions.csv`: per-window empirical metrics and predictions.
- `run_config.json`: arguments and calibration values for the run.
- `aggregate_summary.csv`: mean/std metrics across multiple seeds.
- `main_results_table.tex`: LaTeX table snippet generated from multi-seed results.
- `ablation_results_table.tex`: LaTeX table snippet generated when ablation methods are present.

The current runner is a reproducible experimental scaffold. It is useful for
checking the full evaluation pipeline while keeping calibration choices and any
stronger estimator variants traceable to generated outputs.

## NS-3 supplemental validation

An NS-3 validation layer is available under `experiments/ns3/`. It provides a
scratch program for a UAV-assisted vehicular Wi-Fi link, a PowerShell runner,
and a postprocessor that produces window metrics and a LaTeX table.

```powershell
.\experiments\ns3\run_ns3_experiment.ps1 -Ns3Root ".\tools\ns3\ns-allinone-3.47\ns-3.47"
```

See `experiments/ns3/README.md` for details. This workspace now has a local
NS-3.47 source tree and MSYS2/MinGW toolchain under `tools/ns3/`; the runner
will use that local toolchain when present.
