# Suggested Review Artifact Wording

## Data and Code Availability

An anonymized review artifact has been prepared for this revision and will be made available to the editor and reviewers via a GitHub repository: `[anonymous GitHub URL to be inserted before submission]`. The artifact contains the DES simulation scripts, NS-3 scratch scenario and runner, calibration and aggregation scripts, 20-seed DES aggregate outputs, NS-3 10-seed held-out and seed-fold outputs, revision-audit CSV files, generated LaTeX table snippets, and figure-generation sources. Large raw per-window DES traces and raw NS-3 packet traces are omitted from the lightweight repository to keep the artifact compact, but they can be regenerated from the included scripts and seed lists or supplied as a separate archive upon request. A public archival version of the repository can be released after acceptance.

## Response Letter Wording

To strengthen reproducibility during review, we have prepared an anonymized review artifact at `[anonymous GitHub URL]`. The artifact includes the simulation, calibration, aggregation, NS-3, and audit scripts used to generate the revised tables and figures, together with aggregate CSV outputs and LaTeX table sources. This allows reviewers to inspect the seed lists, calibration protocol, robustness audits, and NS-3 held-out split/seed-fold procedures directly. Large raw traces are excluded from the lightweight GitHub repository but are reproducible from the included scripts and can be provided separately if requested.

## GitHub Upload Steps

1. Create a new anonymous repository, for example `comnet-review-artifact`.
2. Upload the contents of this `review_artifact_github` folder as the repository root.
3. Do not upload reviewer letters, author-identifying cover letters, local build logs, or manuscript PDFs unless the journal explicitly allows it.
4. Insert the anonymous repository URL into the manuscript and response letter wording above.
5. After acceptance, replace the anonymous URL with a public archival URL or DOI if required.

