"""분할과 필수/선택 데이터 경로의 회귀 방지 검사."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.features import delta_q
from src.preprocess import BATCH_FILES, available_batches, load_cache
from src.train import INNER_FOLDS, N_FOLDS, WeightedMedian, fit_model, mape, report_table, ridge, split_b1


class PipelineTests(unittest.TestCase):
    def test_optional_b3_and_required_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            with self.assertRaises(FileNotFoundError):
                available_batches(p)
            for b in ("B1", "B2"):
                (p / BATCH_FILES[b]).touch()
            self.assertEqual(list(available_batches(p)), ["B1", "B2"])
            with self.assertRaises(FileNotFoundError):
                available_batches(p, ["B1", "B2", "B3"])
            (p / BATCH_FILES["B3"]).touch()
            self.assertEqual(list(available_batches(p)), ["B1", "B2", "B3"])
            self.assertEqual(list(available_batches(p, ["B1", "B2"])), ["B1", "B2"])

    def test_cell_and_policy_separation_in_every_level(self):
        df = pd.DataFrame({"batch": "B1", "near_eol": True,
                           "policy_key": np.repeat(np.arange(20), 2),
                           "log_var_dQ": np.linspace(-4.5, -3, 40),
                           "cycle_life": np.linspace(1200, 500, 40)})
        train, valid = split_b1(df)
        self.assertFalse(set(train.policy_key) & set(valid.policy_key))
        for tr, va in GroupKFold(N_FOLDS).split(train, groups=train.policy_key):
            part = train.iloc[tr]
            self.assertFalse(set(part.policy_key) & set(train.iloc[va].policy_key))
            for it, iv in GroupKFold(INNER_FOLDS).split(part, groups=part.policy_key):
                self.assertFalse(set(part.iloc[it].policy_key) & set(part.iloc[iv].policy_key))
        model = fit_model(ridge(False), train, ["log_var_dQ"])
        self.assertIsInstance(model.cv, GroupKFold)
        np.testing.assert_allclose(model.best_estimator_["standardscaler"].mean_,
                                   [train.log_var_dQ.mean()])

    def test_incomplete_cache_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "cells.parquet").touch()
            with patch("src.preprocess.build_cache", return_value="rebuilt") as rebuild:
                self.assertEqual(load_cache(p), "rebuilt")
                rebuild.assert_called_once_with(cache_dir=p)

    def test_delta_uses_cycle_10_and_100(self):
        data = {"B1_c00|9": np.ones(1000), "B1_c00|99": np.ones(1000) * 3,
                "B1_c00|10": np.ones(1000) * 100, "B1_c00|100": np.ones(1000) * 1000}
        np.testing.assert_array_equal(delta_q(data, "B1_c00"), np.full(1000, 2))
        with self.assertRaises(ValueError):
            delta_q({}, "B1_c00")
        data["B1_c00|99"][0] = np.nan
        with self.assertRaises(ValueError):
            delta_q(data, "B1_c00")

    def test_weighted_baseline_matches_mape_objective(self):
        y = np.array([1., 10., 10.])
        model = WeightedMedian().fit(np.zeros((3, 1)), y)
        self.assertEqual(model.value_, 1.)
        self.assertLess(mape(y, model.predict(y)), mape(y, np.full(3, y.mean())))

    def test_mape_rejects_missing_or_invalid_targets(self):
        for y, p in (([], []), ([0], [1]), ([1], [np.nan]), ([1, 2], [1])):
            with self.assertRaises(ValueError):
                mape(y, p)

    def test_report_omits_b3_when_not_evaluated(self):
        table = report_table(10., 12., 30.)
        self.assertEqual(len(table), 6)
        self.assertFalse(table["구분"].str.contains("Batch 3").any())
        self.assertEqual(table.loc[5, "MAPE (%)"], 20.9)
        full = report_table(10., 12., 30., 15.)
        self.assertEqual(len(full), 9)
        self.assertEqual(full.loc[7, "MAPE (%)"], -15.)


if __name__ == "__main__":
    unittest.main()
