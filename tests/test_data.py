"""Check that the published inputs remain the actual full homework dataset."""
import hashlib
import json
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class SourceDataTests(unittest.TestCase):
    def test_published_bytes_match_recorded_source_hashes(self):
        manifest = json.loads((ROOT / 'data/source_manifest.json').read_text())
        for name in ('churn_features.parquet', 'churn_truth.csv'):
            with self.subTest(name=name):
                path = ROOT / 'data' / name
                self.assertEqual(path.stat().st_size, manifest[name]['bytes'])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 manifest[name]['sha256'])

    def test_full_population_and_label_coverage(self):
        features = pd.read_parquet(ROOT / 'data/churn_features.parquet')
        truth = pd.read_csv(ROOT / 'data/churn_truth.csv')
        self.assertEqual(features.shape, (198578, 11))
        self.assertEqual(len(truth), 200000)
        for frame in (features, truth):
            self.assertFalse(frame.subscriber_key.isna().any())
            self.assertTrue(frame.subscriber_key.is_unique)
        self.assertTrue(set(features.subscriber_key) <= set(truth.subscriber_key))
        self.assertEqual(set(truth.churned), {False, True})
        self.assertNotIn('churned', features.columns)
        self.assertNotIn('churn_day_index', features.columns)
        self.assertNotIn('retention_contact', features.columns)


if __name__ == '__main__':
    unittest.main()
