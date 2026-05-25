# Suggested Review Artifact Wording

## Data and Code Availability

A review artifact has been prepared for this revision and is available at <https://github.com/ermalimust/comnet-review-artifact_26>. The artifact contains the DES simulation scripts, learned tail-scale calibration protocol, NS-3 scratch scenario and runner, calibration and aggregation scripts, 20-seed DES aggregate outputs, NS-3 10-seed held-out and seed-fold outputs, revision-audit CSV files, the frequency-identifiability audit, generated LaTeX table snippets, and figure-generation sources. Large raw per-window DES traces and raw NS-3 packet traces are omitted from the lightweight repository to keep the artifact compact, but they can be regenerated from the included scripts and seed lists or supplied as a separate archive upon request. A public archival version of the repository can be released after acceptance.

## Response Letter Wording

To strengthen reproducibility during review, we have prepared a public review artifact at <https://github.com/ermalimust/comnet-review-artifact_26>. The artifact includes the simulation, calibration, aggregation, NS-3, and audit scripts used to generate the revised tables and figures, together with aggregate CSV outputs and LaTeX table sources. This allows reviewers to inspect the seed lists, calibration protocol, coefficient and identifiability audits, robustness audits, and NS-3 held-out split/seed-fold procedures directly. Large raw traces are excluded from the lightweight GitHub repository but are reproducible from the included scripts and can be provided separately if requested.

## GitHub Upload Steps

1. Upload the contents of this `review_artifact_github` folder as the repository root.
2. Do not upload reviewer letters, author-identifying cover letters, local build logs, or manuscript PDFs unless the journal explicitly allows it.
3. Insert the repository URL into the manuscript and response letter wording above.
4. After acceptance, replace the repository URL with a public archival DOI if required.
