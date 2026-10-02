"""분할 · CV 모델 비교 · 홀드아웃 · B2(·B3) 평가 → results/ 저장.

Day 1 설계(3-3 ~ 3-7):
  1. B1 종료 근접 36셀 → 정책 단위 분할 (seed 20261001) : 학습 28 / 홀드아웃 8
  2. 학습 28셀 정책 그룹 4-fold CV, 원 단위 MAPE
       기준선(1/y 가중 중앙값) · Ridge × 입력 5구성 × 타깃 2종(원 수명 / log 수명)
       → 최선 Ridge 대비 Elastic Net · 얕은 회귀트리 (채택: MAPE 1%p↓ + 4개 중 3개 폴드 개선)
         EN은 정책 단독·결합 입력 모두 비교, 트리는 최선 입력·최선 다변수 입력만 탐색 비교
  3. 선택 고정 → 학습 28셀 재학습 → 홀드아웃 8셀 → 같은 모델로 B2 (·B3)
  4. 세부 성능: 실험집단 / 입력 범위 안·밖 / 학습 수명 범위 밖 / 원래 정책별
대치·표준화는 내부·외부 정책 그룹 CV 모두 Pipeline 안에서 각 학습 폴드에만 적합한다.

실행: python -m src.train
"""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.features import FEATURE_SETS, build_features
from src.preprocess import CACHE_DIR

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

SEED = 20261001                   # Day 1 3-4 분할 seed
N_FOLDS = 4                       # 정책 그룹 CV 폴드 수 (학습 28셀 → 폴드당 학습 약 21셀)
TARGET_MAPE = 9.1                 # 과제 지정 기준: 논문 초록의 대표 성능 (동일 분할 재현값 아님)
ALPHAS = np.logspace(-3, 3, 25)
INNER_FOLDS = 3                   # 하이퍼파라미터 선택도 정책 단위로 분리


# ---------------------------------------------------------------- metrics
def mape(y, p):
    """평균 절대 백분율 오차 (%). 원 단위(사이클)에서 계산."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.size == 0 or y.shape != p.shape or not np.isfinite([y, p]).all() or (y <= 0).any():
        raise ValueError("MAPE에는 같은 크기의 유한한 예측과 양수 레이블이 필요합니다.")
    return float(np.mean(np.abs(p - y) / y) * 100)


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def over_rate(y, p):
    """과대 예측률 (%): 수명을 길게 예측하면 점검이 늦어진다."""
    return float(np.mean(np.asarray(p) > np.asarray(y)) * 100)


# ---------------------------------------------------------------- models
class WeightedMedian(BaseEstimator, RegressorMixin):
    """기준선: 1/y 가중 중앙값 (상수 예측 중 MAPE 최소)."""

    def fit(self, X, y):
        # MAPE = Σ|c - y|/y 를 최소화하는 상수 c 는 가중치 1/y 인 가중 중앙값
        y = np.asarray(y, float)
        order = np.argsort(y)
        w = (1 / y)[order]
        cw = np.cumsum(w)
        self.value_ = y[order][np.searchsorted(cw, cw[-1] / 2)]   # 누적 가중치가 절반을 넘는 지점
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


def _linear(reg, log_target):
    """결측 대치 → 표준화 → 회귀 Pipeline. 전처리는 fit 시점의 학습 데이터에만 적합되어 누수가 없다."""
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), reg)
    if log_target:
        # log10(수명)으로 학습하고, 예측은 10**z 로 원 단위(사이클)로 복원
        return TransformedTargetRegressor(pipe, func=np.log10, inverse_func=lambda z: 10 ** z)
    return pipe


def ridge(log_target):
    """전처리를 포함한 Pipeline 전체를 내부 정책 그룹 CV로 선택한다."""
    prefix = "regressor__" if log_target else ""
    return GridSearchCV(_linear(Ridge(), log_target), {prefix + "ridge__alpha": ALPHAS},
                        scoring=make_scorer(mape, greater_is_better=False),
                        cv=GroupKFold(INNER_FOLDS), error_score="raise")


def elastic_net(log_target):
    """L1+L2 후보: 내부 정책 그룹 CV에서 alpha·l1_ratio를 선택한다."""
    prefix = "regressor__" if log_target else ""
    alphas = np.logspace(-5, 1, 25) if log_target else ALPHAS
    return GridSearchCV(_linear(ElasticNet(max_iter=50_000), log_target),
                        {prefix + "elasticnet__alpha": alphas,
                         prefix + "elasticnet__l1_ratio": [0.2, 0.5, 0.8]},
                        scoring=make_scorer(mape, greater_is_better=False),
                        cv=GroupKFold(INNER_FOLDS), error_score="raise")


def fit_model(model, train, cols):
    """내부 탐색에 현재 학습 부분집합의 정책 그룹만 전달한다."""
    kwargs = {"groups": train["policy_key"]} if isinstance(model, GridSearchCV) else {}
    return model.fit(train[cols], train["cycle_life"], **kwargs)


def shallow_tree(log_target):
    """조건부 후보: 구간별 관계·상호작용을 단순 분기로 표현 (깊이 2, 리프 최소 5셀)."""
    tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=5, random_state=SEED)
    return _linear(tree, log_target)


# ---------------------------------------------------------------- split / CV
def split_b1(df):
    """B1 종료 근접 36셀 → 학습 28 / 홀드아웃 8. 같은 정책 셀은 같은 쪽에만 들어간다."""
    b1 = df[(df["batch"] == "B1") & df["near_eol"]].reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)   # 정책 그룹의 약 20%
    tr, ho = next(gss.split(b1, groups=b1["policy_key"]))
    return b1.iloc[tr].reset_index(drop=True), b1.iloc[ho].reset_index(drop=True)


def cv_mape(model, train, cols):
    """정책 그룹 4-fold CV → 폴드별 MAPE 배열."""
    X, y, g = train[cols], train["cycle_life"], train["policy_key"]
    scores = []
    # GroupKFold: 검증 폴드의 정책은 학습 폴드에 없음 → '새 정책' 에 대한 일반화 성능
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(X, y, g):
        m = fit_model(clone(model), train.iloc[tr], cols)
        scores.append(mape(y.iloc[va], m.predict(X.iloc[va])))
    return np.array(scores)


def compare_models(train):
    """후보 모델 × 입력 구성 × 타깃 변환을 같은 CV 로 비교."""
    rows = []

    def add(name, model, fs, target):
        cols = FEATURE_SETS[fs] if fs else ["log_var_dQ"]   # 기준선은 입력을 쓰지 않음
        s = cv_mape(model, train, cols)
        rows.append({"model": name, "features": fs or "-", "target": target,
                     "cv_mape": s.mean(), "cv_std": s.std(), "folds": s})

    # 1) 기준선
    add("Baseline", WeightedMedian(), None, "raw")
    # 2) Ridge: 입력 5구성 × 타깃 2종 = 10개
    for fs in FEATURE_SETS:
        for log_t in (False, True):
            add("Ridge", ridge(log_t), fs, "log" if log_t else "raw")

    # 3) Day 1의 정책 단독·결합 입력을 모두 Elastic Net으로 확인한다.
    ridge_rows = [r for r in rows if r["model"] == "Ridge"]
    best = min(ridge_rows, key=lambda r: r["cv_mape"])
    for fs in FEATURE_SETS:
        for log_t in (False, True):
            add("ElasticNet", elastic_net(log_t), fs, "log" if log_t else "raw")

    # 4) 트리는 최선 입력·최선 다변수 입력에만 제한해 탐색 비교한다.
    # Day 1의 조건부 후보를 진단 비교까지 확장했으며, 채택 기준은 그대로 유지한다.
    multi = min((r for r in ridge_rows if len(FEATURE_SETS[r["features"]]) > 1), key=lambda r: r["cv_mape"])
    seen = set()
    for candidate in (best, multi):
        key = (candidate["features"], candidate["target"])
        if key not in seen:
            add("Tree(d2,leaf5)", shallow_tree(key[1] == "log"), *key)
            seen.add(key)

    res = pd.DataFrame(rows)
    # 최선 Ridge 대비 MAPE 차이 (음수 = 더 좋음) · 폴드별로 최선 Ridge 보다 나은 폴드 수
    res["vs_best_ridge_%p"] = res["cv_mape"] - best["cv_mape"]
    res["folds_improved"] = [int((r["folds"] < best["folds"]).sum()) for r in rows]
    return res, best, multi


def _last_step(model):
    """Pipeline(또는 로그 타깃 래퍼) 안의 마지막 회귀기."""
    model = model.best_estimator_ if isinstance(model, GridSearchCV) else model
    pipe = model.regressor_ if isinstance(model, TransformedTargetRegressor) else model
    return pipe[-1]


def en_selection(train, fs, target):
    """Elastic Net 폴드별 계수 → 0으로 줄어든 입력과 선택의 일관성 확인 (Day 1 3-7)."""
    cols = FEATURE_SETS[fs]
    X, y, g = train[cols], train["cycle_life"], train["policy_key"]
    rows = []
    for k, (tr, _) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, y, g)):
        m = fit_model(elastic_net(target == "log"), train.iloc[tr], cols)
        rows += [{"check": f"EN 계수 ({fs}, {target})", "fold": k, "item": c, "value": coef}
                 for c, coef in zip(cols, _last_step(m).coef_)]
    return pd.DataFrame(rows)


def residual_curvature(model, train, cols):
    """검증 폴드 부호 오차(%)를 log var(ΔQ)의 2차식으로 적합 → 2차 계수의 부호가 폴드마다 반복되는지 확인.

    Day 1 3-7: 선형 모델의 검증 잔차에 곡률이 반복될 때만 얕은 트리를 조건부로 채택 검토한다.
    """
    X, y, g = train[cols], train["cycle_life"], train["policy_key"]
    rows = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, y, g)):
        m = fit_model(clone(model), train.iloc[tr], cols)
        err = (m.predict(X.iloc[va]) - y.iloc[va]) / y.iloc[va] * 100
        quad = np.polyfit(train["log_var_dQ"].iloc[va], err, 2)[0]
        rows.append({"check": "잔차 2차 계수 (최선 Ridge)", "fold": k, "item": "log_var_dQ", "value": quad})
    return pd.DataFrame(rows)


def select_final(res, best):
    """Ridge 최선 대비 MAPE 1%p 이상 감소 + 4개 중 3개 폴드 개선 시에만 대체 후보 채택."""
    final = best
    for _, r in res[res["model"].isin(["ElasticNet", "Tree(d2,leaf5)"])].iterrows():
        if r["vs_best_ridge_%p"] <= -1.0 and r["folds_improved"] >= 3:
            if r["cv_mape"] < final["cv_mape"]:
                final = r.to_dict()
    return final


def make_model(name, target):
    """CV 비교표의 (모델명, 타깃) → 새 모델 객체."""
    log_t = target == "log"
    return {"Ridge": ridge, "ElasticNet": elastic_net, "Tree(d2,leaf5)": shallow_tree}[name](log_t)


# ---------------------------------------------------------------- evaluation
def in_range(df, train, cols):
    """모든 입력이 학습셀의 [최솟값, 최댓값] 안에 있으면 True (외삽 여부 판정)."""
    lo, hi = train[cols].min(), train[cols].max()
    return ((df[cols] >= lo) & (df[cols] <= hi)).all(axis=1)


def score(df):
    """셀 집합 하나의 성능 요약."""
    return {"n": len(df), "MAPE": mape(df["cycle_life"], df["pred"]),
            "MAE": mae(df["cycle_life"], df["pred"]), "over_%": over_rate(df["cycle_life"], df["pred"])}


def breakdown(pred, train, cols):
    """홀드아웃·B2·B3 세부: 전체 / 실험집단별 / 입력 범위 안·밖 / 학습 수명 범위 밖 / 원래 정책별."""
    lo, hi = train["cycle_life"].min(), train["cycle_life"].max()
    pred = pred.assign(
        in_range=in_range(pred, train, cols),
        # 홀드아웃은 B1 안의 '새 정책' 평가라 B2·B3 와 구분해 집계
        set=np.where(pred["split"] == "valid", "B1 holdout", pred["batch"]),
    )
    rows = []
    for name, g in pred.groupby("set", sort=False):
        rows.append({"batch": name, "subset": "all", **score(g)})
        for grp, gg in g.groupby("group"):
            rows.append({"batch": name, "subset": f"group={grp}", **score(gg)})
        for flag, gg in g.groupby("in_range"):
            rows.append({"batch": name, "subset": "in_range" if flag else "out_of_range", **score(gg)})
        # 학습 수명 범위(최솟값~최댓값) 밖 셀: 짧은 수명 / 긴 수명 쪽 오차를 따로 확인
        for label, mask in [(f"life<{lo:.0f}", g["cycle_life"] < lo), (f"life>{hi:.0f}", g["cycle_life"] > hi)]:
            if mask.any():
                rows.append({"batch": name, "subset": label, **score(g[mask])})
        # Day 1 3-5: 오차는 원래 충전 정책별로도 본다
        for pol, gg in g.groupby("policy_key"):
            rows.append({"batch": name, "subset": f"policy={pol}", **score(gg)})
    return pd.DataFrame(rows)


def report_table(cv, valid, b2, b3=None, *, n_valid=8, n_b2=39, n_b3=44):
    """과제 Reporting format (Regression). Gap 은 모두 '뒤 - 앞' → (+) 이면 성능 저하."""
    rows = [
        ("Train (Batch 1 CV)", cv, f"{N_FOLDS}-fold 정책 그룹 CV 평균"),
        ("Valid (Batch 1 Hold-out)", valid, f"정책 단위 홀드아웃 {n_valid}셀"),
        ("Test (Batch 2)", b2, f"레이블 보유 {n_b2}셀"),
        ("Gap (Train-Valid)", valid - cv, "(+) : 과적합 의심"),
        ("Gap (Valid-Test)", b2 - valid, "(+) : 배치간 일반화 저하 의심"),
        ("Gap (Target-Test)", b2 - TARGET_MAPE, f"Target : 원논문 {TARGET_MAPE}%"),
    ]
    if b3 is not None:
        rows += [
            ("Test (Batch 3)", b3, f"레이블 보유 {n_b3}셀"),
            ("Gap (Batch2-Batch3)", b3 - b2, "Test 성능 간 비교 · (+) : B3에서 저하"),
            ("Gap (Target-Test, Batch 3)", b3 - TARGET_MAPE, "Batch 3 기준, 원논문 성능 비교"),
        ]
    return pd.DataFrame(rows, columns=["구분", "MAPE (%)", "비고"]).round({"MAPE (%)": 2})


def main(cache_dir=CACHE_DIR, results_dir=RESULTS_DIR):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df = build_features(cache_dir)
    if not {"B1", "B2"}.issubset(set(df["batch"])):
        raise ValueError("B1 학습·B2 평가 데이터가 모두 필요합니다.")
    train, holdout = split_b1(df)   # 학습 28 / 홀드아웃 8

    # 1) 학습 28셀 안에서만 CV 비교 → 최종 모델 선택 (홀드아웃·B2 는 선택에 쓰지 않음)
    res, best, multi = compare_models(train)
    final = select_final(res, best)
    cols = FEATURE_SETS[final["features"]]
    print(res.drop(columns="folds").round(2).to_string(index=False))
    print(f"\n최종: {final['model']} · {final['features']} · target={final['target']}")

    # 1-1) 설계 점검: EN 폴드별 변수 선택 · 최선 Ridge 검증 잔차의 곡률 반복 여부
    diag = pd.concat([
        en_selection(train, multi["features"], multi["target"]),
        residual_curvature(make_model("Ridge", best["target"]), train, FEATURE_SETS[best["features"]]),
    ], ignore_index=True)
    print("\n" + diag.pivot_table(index=["check", "item"], columns="fold", values="value").round(3).to_string())

    # 2) 선택 고정 → 학습 28셀 전체로 재학습. B2 평가에도 같은 모델을 그대로 사용
    model = fit_model(make_model(final["model"], final["target"]), train, cols)

    # 3) 홀드아웃(valid) · B2/B3(test) 예측
    test = df[df["batch"].isin(["B2", "B3"])]
    pred = pd.concat([holdout.assign(split="valid"), test.assign(split="test")], ignore_index=True)
    pred["pred"] = model.predict(pred[cols])
    pred["ape_%"] = (pred["pred"] - pred["cycle_life"]).abs() / pred["cycle_life"] * 100   # 셀별 백분율 오차

    # 4) 성능 집계
    valid = mape(holdout["cycle_life"], pred.loc[pred["split"] == "valid", "pred"])
    b2 = pred[pred["batch"] == "B2"]
    b3 = pred[pred["batch"] == "B3"]
    table = report_table(final["cv_mape"], valid, mape(b2["cycle_life"], b2["pred"]),
                         mape(b3["cycle_life"], b3["pred"]) if len(b3) else None,
                         n_valid=len(holdout), n_b2=len(b2), n_b3=len(b3))
    detail = breakdown(pred, train, cols)

    # 5) 저장 (folds 배열은 CSV 에 넣기 위해 리스트로 변환)
    res.assign(folds=res["folds"].apply(lambda a: json.dumps(a.tolist()))).to_csv(
        results_dir / "cv_comparison.csv", index=False)
    table.to_csv(results_dir / "model_performance.csv", index=False)
    detail.to_csv(results_dir / "test_breakdown.csv", index=False)
    diag.to_csv(results_dir / "cv_diagnostics.csv", index=False)
    pred.to_csv(results_dir / "predictions.csv", index=False)

    manifest = df[["cell_id", "batch", "policy_key", "cycle_life", "last_QD", "near_eol"]].copy()
    manifest["split"] = np.where(manifest["batch"] == "B1", "excluded", "test")
    manifest.loc[manifest.cell_id.isin(train.cell_id), "split"] = "train"
    manifest.loc[manifest.cell_id.isin(holdout.cell_id), "split"] = "valid"
    manifest["cv_fold"] = -1
    for k, (_, va) in enumerate(GroupKFold(N_FOLDS).split(train, groups=train.policy_key)):
        manifest.loc[manifest.cell_id.isin(train.iloc[va].cell_id), "cv_fold"] = k
    manifest.to_csv(results_dir / "split_manifest.csv", index=False)
    metadata = {
        "seed": SEED, "outer_folds": N_FOLDS, "inner_folds": INNER_FOLDS,
        "model": final["model"], "feature_set": final["features"], "features": cols,
        "target": final["target"], "target_mape": TARGET_MAPE,
        "cv_mape": float(final["cv_mape"]), "cv_std": float(final["cv_std"]),
        "best_params": model.best_params_ if isinstance(model, GridSearchCV) else {},
        "n_train": len(train), "n_valid": len(holdout), "n_b2": len(b2), "n_b3": len(b3),
        "input_bounds": {c: [float(train[c].min()), float(train[c].max())] for c in cols},
        "life_bounds": [float(train.cycle_life.min()), float(train.cycle_life.max())],
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "h5py", "pyarrow", "matplotlib")},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "src").glob("*.py"))},
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    from src.plotting import save_figures
    save_figures(train, pred, model, cols, results_dir)

    print("\n" + table.to_string(index=False))
    print("\n" + detail.round(2).to_string(index=False))
    return {"train": train, "holdout": holdout, "cv": res, "final": final, "diag": diag,
            "model": model, "pred": pred, "table": table, "detail": detail}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    main(args.cache_dir, args.results_dir)
