import argparse
import json
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from train.train_utility_router import train


class UtilityTrainerTest(unittest.TestCase):
    def test_cpu_smoke_training_writes_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "train_path": str(root / "train.jsonl"),
                "output_dir": str(root / "output"),
                "query_dim": 3,
                "signature_dim": 3,
                "hidden_dim": 8,
                "interaction_rank": 4,
                "verifier_hidden_dim": 8,
                "batch_size": 2,
                "epochs": 1,
                "seed": 7,
            }
            (root / "config.yaml").write_text(
                yaml.safe_dump(config), encoding="utf-8"
            )
            signatures = {
                "general": [0.8, 0.4, 0.3],
                "spatial_reasoning": [0.5, 0.9, 0.4],
                "spatiotemporal": [0.5, 0.5, 0.9],
            }
            (root / "signatures.json").write_text(
                json.dumps(signatures), encoding="utf-8"
            )
            rows = [
                {
                    "query_embedding": [0.1, 0.2, 0.3],
                    "left": {
                        "expert": "spatial_reasoning",
                        "action": {"knowledge": "template", "tool": "ground"},
                    },
                    "right": {
                        "expert": "general",
                        "action": {"knowledge": "none", "tool": "none"},
                    },
                    "preference": 1,
                    "target_gain": 0.4,
                },
                {
                    "query_embedding": [0.3, 0.2, 0.1],
                    "left": {
                        "expert": "general",
                        "action": {"knowledge": "none", "tool": "none"},
                    },
                    "right": {
                        "expert": "spatiotemporal",
                        "action": {"knowledge": "rag", "tool": "depth"},
                    },
                    "preference": -1,
                    "target_gain": -0.2,
                },
            ]
            (root / "train.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                config=str(root / "config.yaml"),
                train_path="",
                verifier_path="",
                signatures=str(root / "signatures.json"),
                output_dir="",
                device="cpu",
                cost_weight=1.0,
                calibration_weight=0.1,
            )
            train(args)
            checkpoint_path = root / "output" / "knowinrouter.pt"
            self.assertTrue(checkpoint_path.exists())
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.assertIn("utility_model", checkpoint)
            self.assertIn("packet_verifier", checkpoint)


if __name__ == "__main__":
    unittest.main()
