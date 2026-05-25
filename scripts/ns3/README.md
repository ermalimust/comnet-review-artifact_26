# NS-3 Packet-Level Interruption Case Study

This directory adds a portable NS-3 packet-level case-study layer for the paper.

The goal is not to replace the dependency-free DES experiment. The NS-3 run is
intended as conservative packet-level plausibility evidence for the Computer
Networks positioning: heterogeneous wireless traffic, rule-driven background
transmissions, shared wireless contention, mobility/channel variation, and
target-flow delay risk.

The scratch program supports two topology modes:

- `single_sta` (default): target and rule-driven background flows originate from
  the UAV STA and share the AP medium.
- `two_sta_contention`: the target flow originates from the UAV STA, while
  Remote-ID, telemetry, and C2 background flows originate from a second STA that
  contends for the same AP.

## Files

- `uav_vehicular_vacation.cc`: NS-3 scratch program.
- `run_ns3_experiment.ps1`: copies the scratch program into an NS-3 tree and
  runs multi-seed scenarios.
- `postprocess_ns3.py`: converts packet traces into window-level metrics and a
  LaTeX table snippet.

## Requirements

This workspace now includes a local NS-3.47 source tree and MinGW/MSYS2 build
environment under:

```text
tools/ns3/ns-allinone-3.47/ns-3.47
tools/ns3/msys64
```

The runner will automatically use `tools/ns3/msys64/usr/bin/bash.exe` when it is
present. You can also point `-Ns3Root` to another working NS-3 source tree.

The runner supports both the newer `ns3` frontend and legacy `waf` frontend:

- `<ns3-root>/ns3`
- `<ns3-root>/waf`

## Run

From the repository root:

```powershell
.\scripts\ns3\run_ns3_experiment.ps1 -Ns3Root ".\tools\ns3\ns-allinone-3.47\ns-3.47"
```

Useful options:

```powershell
.\scripts\ns3\run_ns3_experiment.ps1 `
  -Ns3Root ".\tools\ns3\ns-allinone-3.47\ns-3.47" `
  -Seeds @(7,11,13,17,19) `
  -Scenarios @("overall","load_high","vacation_high","drift_strong") `
  -Target video `
  -Duration 120 `
  -OutDir "outputs/ns3_validation"
```

The runner writes packet traces under:

```text
outputs/ns3_validation/packets/
```

Then it writes:

- `ns3_window_metrics.csv`
- `ns3_summary.csv`
- `ns3_validation_table.tex`

Two-STA topology probe:

```powershell
.\scripts\ns3\run_ns3_experiment.ps1 `
  -Ns3Root ".\tools\ns3\ns-allinone-3.47\ns-3.47" `
  -Seeds @(7,11,13) `
  -Scenarios @("overall","vacation_high","drift_strong") `
  -Target video `
  -Duration 120 `
  -Topology two_sta_contention `
  -OutDir "outputs/ns3_topology_probe"
python .\scripts\ns3\evaluate_ns3_heldout.py --window ".\outputs\ns3_topology_probe\ns3_window_metrics.csv" --outdir ".\outputs\ns3_topology_probe" --train-seeds 7,11 --test-seeds 13
```

The default NS-3 deadline threshold is `0.01` s. The postprocessor reports
received-packet delay statistics separately from a loss-inclusive deadline
violation rate: any target packet with delay above the threshold, or any target
packet not received by the sink, is counted as a violation.

## Manual NS-3 Run

If you prefer to run from inside the NS-3 tree:

```powershell
copy .\scripts\ns3\uav_vehicular_vacation.cc <path-to-ns-3.47>\scratch\
cd <path-to-ns-3.47>
python .\ns3 run "uav_vehicular_vacation --scenario=overall --seed=7 --out=D:/tmp/ns3_packets.csv"
```

Postprocess manually:

```powershell
cd <review-artifact-root>
python .\scripts\ns3\postprocess_ns3.py --input "D:\tmp\ns3_packets.csv" --outdir .\outputs\ns3_validation_extended
```

## Scenarios

The scratch program currently supports:

- `overall`
- `load_high`
- `vacation_high`
- `drift_strong`
- `traffic_mix_video_heavy`
- `traffic_mix_c2_heavy`

Each scenario adjusts target load, rule-driven background occupation, mobility,
and channel drift. The target flow can be `video` or `event_c2`. The topology can
be `single_sta` or `two_sta_contention`.

## Interpretation

This NS-3 layer should be reported as a packet-level interruption case study.
It is best used to support claims such as:

- rule-driven background traffic creates target-flow delay inflation;
- high load and strong drift remain the hardest cases;
- vacation-like service unavailability is observable in a packet-level wireless
  simulator.
- moving background flows to a separate contending STA preserves the expected
  high-vacation delay and strong-drift loss signatures in a second small
  topology.

It should not be described as a full hardware, field-trace, C-V2X, NR-V2X, or
deployment validation unless additional trace-driven calibration is added.
