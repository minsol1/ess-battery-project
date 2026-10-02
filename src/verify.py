"""현재 캐시·코드·저장 결과의 일치 여부를 검증한다: python -m src.verify."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.features import FEATURE_SETS, build_features
from src.preprocess import CACHE_DIR
from src.train import (INNER_FOLDS, N_FOLDS, RESULTS_DIR, ROOT, breakdown, fit_model,
                       make_model, mape, report_table, select_final, split_b1)


def verify(cache_dir=CACHE_DIR, results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)
    metadata = json.loads((results_dir / "run_metadata.json").read_text())
    for name, digest in metadata["source_sha256"].items():
        assert hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() == digest, f"코드가 실행 후 변경됨: {name}"
    feat = build_features(cache_dir)
    train, valid = split_b1(feat)
    assert set(train.cell_id).isdisjoint(valid.cell_id)
    assert set(train.policy_key).isdisjoint(valid.policy_key)
    for tr, va in GroupKFold(N_FOLDS).split(train, groups=train.policy_key):
        part = train.iloc[tr]
        assert set(part.policy_key).isdisjoint(train.iloc[va].policy_key)
        for it, iv in GroupKFold(INNER_FOLDS).split(part, groups=part.policy_key):
            assert set(part.iloc[it].policy_key).isdisjoint(part.iloc[iv].policy_key)

    manifest = pd.read_csv(results_dir / "split_manifest.csv")
    assert manifest.cell_id.is_unique and set(manifest.cell_id) == set(feat.cell_id)
    assert set(manifest.loc[manifest.split == "train", "cell_id"]) == set(train.cell_id)
    assert set(manifest.loc[manifest.split == "valid", "cell_id"]) == set(valid.cell_id)
    for k, (_, va) in enumerate(GroupKFold(N_FOLDS).split(train, groups=train.policy_key)):
        assert set(manifest.loc[manifest.cv_fold == k, "cell_id"]) == set(train.iloc[va].cell_id)

    cv = pd.read_csv(results_dir / "cv_comparison.csv")
    cv["folds"] = cv["folds"].apply(lambda s: np.array(json.loads(s)))
    for _, row in cv.iterrows():
        np.testing.assert_allclose(row.cv_mape, row.folds.mean(), rtol=1e-12)
        np.testing.assert_allclose(row.cv_std, row.folds.std(), rtol=1e-12)
    best = cv[cv.model == "Ridge"].sort_values("cv_mape").iloc[0].to_dict()
    selected = select_final(cv, best)
    assert (selected["model"], selected["features"], selected["target"]) == (
        metadata["model"], metadata["feature_set"], metadata["target"])

    pred = pd.read_csv(results_dir / "predictions.csv")
    expected = set(valid.cell_id) | set(feat.loc[feat.batch.isin(["B2", "B3"]), "cell_id"])
    assert pred.cell_id.is_unique and set(pred.cell_id) == expected
    assert set(pred.loc[pred.split == "valid", "cell_id"]) == set(valid.cell_id)
    cols = FEATURE_SETS[metadata["feature_set"]]
    assert cols == metadata["features"]
    assert set(cols).isdisjoint({"cycle_life", "last_QD", "near_eol", "batch", "max_time_gap_hours"})
    expected_features = feat.set_index("cell_id").loc[pred.cell_id, cols + ["cycle_life"]]
    np.testing.assert_allclose(pred[cols + ["cycle_life"]], expected_features, rtol=1e-12)
    model = fit_model(make_model(metadata["model"], metadata["target"]), train, cols)
    np.testing.assert_allclose(model.predict(pred[cols]), pred.pred, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(pred["ape_%"], (pred.pred - pred.cycle_life).abs() / pred.cycle_life * 100)
    score = lambda d: mape(d.cycle_life, d.pred)
    b2, b3, holdout = (pred[pred.batch == b] for b in ("B2", "B3", "B1"))
    table = report_table(metadata["cv_mape"], score(holdout), score(b2), score(b3) if len(b3) else None,
                         n_valid=len(holdout), n_b2=len(b2), n_b3=len(b3))
    pd.testing.assert_frame_equal(table, pd.read_csv(results_dir / "model_performance.csv"))
    pd.testing.assert_frame_equal(breakdown(pred, train, cols), pd.read_csv(results_dir / "test_breakdown.csv"),
                                  check_exact=False, rtol=1e-10, atol=1e-10)
    print(f"PASS: 정책 분리(홀드아웃·외부 CV·내부 CV), 후보 선택, {len(pred)}셀 재학습 예측, 성능표·세부 지표·코드 해시 일치")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    verify(args.cache_dir, args.results_dir)
