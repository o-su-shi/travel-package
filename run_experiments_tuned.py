"""
Optunaチューニング込みで OOF AUC を測り直す（実際の提出フローを再現）。
固定パラメータのablation（TEの寄与確認）も併記して比較図を更新する。
出力: assets/method_comparison.png を上書き
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import optuna
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

optuna.logging.set_verbosity(optuna.logging.WARNING)
RS = 42
N = 5
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=RS)

from src.preprocessing.cleaning import run_cleaning
from src.preprocessing.imputation import MissingValueImputer
from src.features.basic_features import FeatureEngineer
from src.features.categorical_features import CategoricalEncoder
from src.features.advanced_features import PostEncodingEngineer

print("Building features...")
df = run_cleaning(pd.read_csv("data/raw/train.csv"))
df = MissingValueImputer().fit_transform(df)
df = FeatureEngineer().fit_transform(df)
df = CategoricalEncoder(te_columns=["Product_Clean", "Convert_Designation", "MaritalStatus"],
                        target_col="ProdTaken").fit_transform(df)
df = PostEncodingEngineer().fit_transform(df)
y = df["ProdTaken"].reset_index(drop=True)
X = df.drop(columns=["ProdTaken"]).reset_index(drop=True)
cat = X.select_dtypes(include=["category", "object"]).columns.tolist()
bte = [c for c in X.columns if c.endswith("_bte")]
X_note = X.drop(columns=bte)
base_cols = [c for c in ["CityTier", "NumberOfPersonVisiting", "NumberOfFollowups",
                         "PreferredPropertyStar", "Passport", "PitchSatisfactionScore"] if c in X.columns]


# ---------- 固定パラメータ LGBM（ablation用）----------
def oof_lgb_fixed(Xin):
    c = Xin.select_dtypes(include=["category", "object"]).columns.tolist()
    oof = np.zeros(len(Xin))
    p = {"objective": "binary", "metric": "auc", "verbosity": -1, "seed": RS,
         "learning_rate": 0.05, "num_leaves": 31, "max_depth": 3}
    for tr, va in skf.split(Xin, y):
        dtr = lgb.Dataset(Xin.iloc[tr], label=y.iloc[tr], categorical_feature=c, free_raw_data=False)
        dva = lgb.Dataset(Xin.iloc[va], label=y.iloc[va], reference=dtr, categorical_feature=c, free_raw_data=False)
        m = lgb.train(p, dtr, num_boost_round=1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(Xin.iloc[va], num_iteration=m.best_iteration)
    return roc_auc_score(y, oof)


print("Ablation (fixed params)...")
auc_base = oof_lgb_fixed(X[base_cols])
auc_note = oof_lgb_fixed(X_note)
auc_te = oof_lgb_fixed(X)
print(f"  baseline={auc_base:.4f}  no-TE={auc_note:.4f}  +TE={auc_te:.4f}")


# ---------- Optunaチューニング各モデル（OOF予測を返す）----------
def tune_lgb():
    def obj(t):
        p = {"objective": "binary", "metric": "auc", "verbosity": -1, "seed": RS,
             "learning_rate": t.suggest_float("learning_rate", 1e-3, 0.1, log=True),
             "num_leaves": t.suggest_int("num_leaves", 16, 128, step=16),
             "max_depth": t.suggest_int("max_depth", 1, 4),
             "min_data_in_leaf": t.suggest_int("min_data_in_leaf", 10, 100, step=10),
             "feature_fraction": t.suggest_float("feature_fraction", 0.3, 1.0, step=0.1),
             "bagging_fraction": t.suggest_float("bagging_fraction", 0.6, 1.0, step=0.1),
             "bagging_freq": t.suggest_int("bagging_freq", 1, 10),
             "lambda_l1": t.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
             "lambda_l2": t.suggest_float("lambda_l2", 1e-8, 10.0, log=True)}
        s = []
        for tr, va in skf.split(X, y):
            dtr = lgb.Dataset(X.iloc[tr], label=y.iloc[tr], categorical_feature=cat, free_raw_data=False)
            dva = lgb.Dataset(X.iloc[va], label=y.iloc[va], reference=dtr, categorical_feature=cat, free_raw_data=False)
            m = lgb.train(p, dtr, num_boost_round=2000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            s.append(roc_auc_score(y.iloc[va], m.predict(X.iloc[va], num_iteration=m.best_iteration)))
        return np.mean(s)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RS))
    st.optimize(obj, n_trials=40)
    bp = st.best_params; bp.update({"objective": "binary", "metric": "auc", "verbosity": -1, "seed": RS})
    oof = np.zeros(len(X))
    for tr, va in skf.split(X, y):
        dtr = lgb.Dataset(X.iloc[tr], label=y.iloc[tr], categorical_feature=cat, free_raw_data=False)
        dva = lgb.Dataset(X.iloc[va], label=y.iloc[va], reference=dtr, categorical_feature=cat, free_raw_data=False)
        m = lgb.train(bp, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    return oof


def tune_cat():
    Xc = X.copy()
    for c in cat:
        Xc[c] = Xc[c].astype(str).fillna("missing")
    def obj(t):
        p = {"iterations": t.suggest_int("iterations", 300, 1500, step=100),
             "learning_rate": t.suggest_float("learning_rate", 1e-2, 0.3, log=True),
             "depth": t.suggest_int("depth", 4, 8),
             "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
             "eval_metric": "AUC", "random_seed": RS, "verbose": False,
             "early_stopping_rounds": 50}
        s = []
        for tr, va in skf.split(Xc, y):
            m = CatBoostClassifier(**p)
            m.fit(Xc.iloc[tr], y.iloc[tr], cat_features=cat,
                  eval_set=(Xc.iloc[va], y.iloc[va]), verbose=False)
            s.append(roc_auc_score(y.iloc[va], m.predict_proba(Xc.iloc[va])[:, 1]))
        return np.mean(s)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RS))
    st.optimize(obj, n_trials=25)
    bp = st.best_params; bp.update({"eval_metric": "AUC", "random_seed": RS, "verbose": False,
                                    "early_stopping_rounds": 50})
    oof = np.zeros(len(X))
    for tr, va in skf.split(Xc, y):
        m = CatBoostClassifier(**bp)
        m.fit(Xc.iloc[tr], y.iloc[tr], cat_features=cat,
              eval_set=(Xc.iloc[va], y.iloc[va]), verbose=False)
        oof[va] = m.predict_proba(Xc.iloc[va])[:, 1]
    return oof


def tune_xgb():
    def obj(t):
        p = {"n_estimators": t.suggest_int("n_estimators", 300, 1500, step=100),
             "learning_rate": t.suggest_float("learning_rate", 1e-2, 0.3, log=True),
             "max_depth": t.suggest_int("max_depth", 3, 8),
             "subsample": t.suggest_float("subsample", 0.6, 1.0),
             "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
             "reg_alpha": t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
             "reg_lambda": t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
             "eval_metric": "auc", "random_state": RS, "enable_categorical": True,
             "early_stopping_rounds": 50}
        s = []
        for tr, va in skf.split(X, y):
            m = XGBClassifier(**p)
            m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
            s.append(roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
        return np.mean(s)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RS))
    st.optimize(obj, n_trials=30)
    bp = st.best_params; bp.update({"eval_metric": "auc", "random_state": RS,
                                    "enable_categorical": True, "early_stopping_rounds": 50})
    oof = np.zeros(len(X))
    for tr, va in skf.split(X, y):
        m = XGBClassifier(**bp)
        m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


print("Tuning LGBM (Optuna)..."); oof_l = tune_lgb(); a_l = roc_auc_score(y, oof_l); print(f"  {a_l:.4f}")
print("Tuning CatBoost (Optuna)..."); oof_c = tune_cat(); a_c = roc_auc_score(y, oof_c); print(f"  {a_c:.4f}")
print("Tuning XGBoost (Optuna)..."); oof_x = tune_xgb(); a_x = roc_auc_score(y, oof_x); print(f"  {a_x:.4f}")

oofs = [oof_l, oof_c, oof_x]
def neg(w):
    w = np.array(w) / np.sum(w)
    return -roc_auc_score(y, sum(wi * o for wi, o in zip(w, oofs)))
res = minimize(neg, [1/3]*3, method="Nelder-Mead", bounds=[(0, 1)]*3)
w = res.x / res.x.sum()
a_ens = roc_auc_score(y, sum(wi * o for wi, o in zip(w, oofs)))

rows = [
    ("Baseline (raw numeric only, fixed)", auc_base, "#b0b0b0"),
    ("Full features, no Target Enc. (fixed)", auc_note, "#9ec3e8"),
    ("Full + Target Enc. (fixed)", auc_te, "#9ec3e8"),
    ("LGBM tuned (Optuna)", a_l, "#3b6fb5"),
    ("CatBoost tuned (Optuna)", a_c, "#3b6fb5"),
    ("XGBoost tuned (Optuna)", a_x, "#3b6fb5"),
    ("Ensemble tuned (final submission)", a_ens, "#e08a2e"),
]
print("\n==== 5-fold OOF AUC ====")
for n, a, _ in rows:
    print(f"  {a:.4f}  {n}")
print(f"\nEnsemble weights LGBM/CatBoost/XGB = {w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}")

# 図
labels = [r[0] for r in rows]; vals = [r[1] for r in rows]; cols = [r[2] for r in rows]
plt.figure(figsize=(10, 5.2))
yp = np.arange(len(labels))[::-1]
plt.barh(yp, vals, color=cols)
plt.yticks(yp, labels)
plt.xlim(0.6, max(vals) + 0.04)
plt.xlabel("5-fold OOF AUC")
plt.title("Method comparison: fixed-param ablation vs Optuna-tuned models & ensemble")
for p, v in zip(yp, vals):
    plt.text(v + 0.003, p, f"{v:.3f}", va="center", fontweight="bold")
plt.tight_layout()
plt.savefig("assets/method_comparison.png", bbox_inches="tight", dpi=130)
print("Saved assets/method_comparison.png")
