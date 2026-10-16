# Code and Data Supplement

This directory provides a compact CPU validation path for EKT-Router.
The records in `sample_data/` are synthetic schema examples. They do not contain
benchmark questions, media, predictions, private storage paths, or model output
from the submitted experiments.

Run the CPU smoke workflow from the repository root:

```bash
bash supplement/run_smoke.sh
```

The workflow performs three checks:

1. It derives all three expert signatures from a complete diagnostic grid.
2. It trains the factorized joint utility model and packet verifier on tiny
   illustrative records.
3. It confirms that the bounded inference demo still executes.

The generated checkpoint is written to a temporary directory and removed when
the check finishes. It is not a trained research artifact and must not be used
for benchmark evaluation.

## Included schemas

| File | Schema exercised |
|---|---|
| `diagnostic_outcomes.jsonl` | Expert-by-capability outcome grid |
| `route_pairs.jsonl` | Pairwise complete-route preference records |
| `verifier.jsonl` | Post-acquisition packet quality and gain records |
| `router_smoke.yaml` | Minimal CPU training configuration |

Model and data repositories currently contain Coming soon cards only; no release date is promised.
Raw benchmarks must be downloaded under their original licenses.
