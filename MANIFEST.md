# Manifest

This file summarizes the review artifact contents and how each group maps to the revised manuscript.

## Scripts

- `scripts/run_simulation.py`: DES simulator and model/baseline evaluation.
- `scripts/aggregate_results.py`: DES multi-seed aggregation and table generation.
- `scripts/revision_audits.py`: seed-level bootstrap/Wilcoxon audit and risk-weight audit.
- `scripts/coefficient_sensitivity.py`: correction-coefficient sensitivity audit.
- `scripts/model_blind_descriptors.py`: observable-descriptor audit.
- `scripts/identifiability_audit.py`: controlled frequency-identifiability audit for unavailable-interval descriptors.
- `scripts/nonstationarity_shift_audit.py`: mid-trace non-stationary shift audit.
- `scripts/make_figures.py`: PGFPlots/TikZ source generation for manuscript figures.
- `scripts/ns3/run_ns3_experiment.ps1`: NS-3 batch runner.
- `scripts/ns3/uav_vehicular_vacation.cc`: NS-3 scratch scenario.
- `scripts/ns3/postprocess_ns3.py`: packet-trace to window-metric postprocessing.
- `scripts/ns3/evaluate_ns3_heldout.py`: representative held-out NS-3 evaluation.
- `scripts/ns3/evaluate_ns3_seedfold.py`: five-fold seed-pair NS-3 audit.

## Outputs

- `outputs/des_20seed_summary/aggregate_summary.csv`: main 20-seed DES aggregate metrics.
- `outputs/des_20seed_summary/control_policy_summary.csv`: same-budget action-selection summary.
- `outputs/revision_audits/statistical_audit_summary.csv`: seed-level paired audit.
- `outputs/revision_audits/risk_weight_sensitivity_summary.csv`: risk-weight robustness audit.
- `outputs/revision_audits/coefficient_sensitivity_*.csv`: correction-coefficient sensitivity outputs.
- `outputs/revision_audits/model_blind_descriptor_*.csv`: observable-descriptor audit outputs.
- `outputs/revision_audits/identifiability_*.csv`: controlled frequency-identifiability audit outputs.
- `outputs/revision_audits/nonstationarity_shift_*.csv`: non-stationary shift audit outputs.
- `outputs/ns3_10seed_summary/ns3_summary.csv`: 10-seed NS-3 packet-level summary.
- `outputs/ns3_10seed_summary/ns3_shadow_prediction_metrics.csv`: representative held-out NS-3 metrics.
- `outputs/ns3_10seed_summary/ns3_seedfold_metrics_summary.csv`: five-fold seed-pair NS-3 metrics.
- `outputs/ns3_10seed_summary/ns3_window_metrics.csv`: window-level NS-3 metrics derived from packet traces.

## Tables and Figures

- `tables/des/`: DES table snippets included in the revised manuscript.
- `tables/ns3/`: NS-3 table snippets included in the revised manuscript.
- `figures/`: PGFPlots/TikZ sources for revised manuscript figures.

## Omitted Large Files

The following files are intentionally omitted from this lightweight GitHub-ready artifact:

- Per-seed DES `window_predictions.csv` files.
- Raw NS-3 packet CSV traces.
- LaTeX build intermediates and generated PDFs.

These files can be regenerated using the scripts and seed lists documented in `README.md`.
