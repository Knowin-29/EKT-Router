from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.protocol_manifest import (
    EvaluationManifest,
    EvaluationProtocol,
    ResultProvenance,
    SourceType,
    aggregate_eligible,
)


def protocol() -> EvaluationProtocol:
    return EvaluationProtocol(
        benchmark="Where2Place",
        split="test",
        prompt_hash="prompt-sha256",
        decode_hash="decode-sha256",
        media_budget={"max_frames": 32, "max_pixels": 1003520},
        scorer_name="where2place-native",
        scorer_revision="v1.0",
    )


class EvaluationManifestTest(unittest.TestCase):
    def test_reproduced_manifest_round_trip(self):
        manifest = EvaluationManifest(
            protocol=protocol(),
            result=ResultProvenance(
                model_name="EKT-Router",
                source_type=SourceType.REPRODUCED,
                sample_count=100,
                unique_sample_count=100,
                model_revision="router-v10",
                checkpoint_hash="checkpoint-sha256",
                prediction_hash="prediction-sha256",
            ),
        )
        self.assertEqual(manifest.validation_errors(), ())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "manifest.json"
            manifest.write_json(target)
            row = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(row["schema_version"], "ekt-eval-manifest/1.0")
        self.assertEqual(row["result"]["source_type"], "reproduced")
        self.assertEqual(row["protocol_hash"], manifest.protocol.protocol_hash)

    def test_incomplete_or_unhashed_run_fails_closed(self):
        manifest = EvaluationManifest(
            protocol=protocol(),
            result=ResultProvenance(
                model_name="partial-run",
                source_type=SourceType.REPRODUCED,
                sample_count=100,
                unique_sample_count=99,
                model_revision="rev",
            ),
        )
        errors = manifest.validation_errors()
        self.assertIn("sample_ids_not_unique_or_incomplete", errors)
        self.assertIn("missing_checkpoint_hash", errors)
        self.assertIn("missing_prediction_hash", errors)

    def test_reported_rows_cannot_form_matched_aggregate(self):
        reproduced = EvaluationManifest(
            protocol=protocol(),
            result=ResultProvenance(
                model_name="local",
                source_type=SourceType.REPRODUCED,
                sample_count=100,
                unique_sample_count=100,
                model_revision="rev",
                checkpoint_hash="ckpt",
                prediction_hash="pred",
            ),
        )
        reported = EvaluationManifest(
            protocol=protocol(),
            result=ResultProvenance(
                model_name="paper-only",
                source_type=SourceType.REPORTED,
                sample_count=100,
                unique_sample_count=100,
                source_locator="doi:example/table-1",
            ),
        )
        self.assertFalse(aggregate_eligible([reproduced, reported]))


if __name__ == "__main__":
    unittest.main()
