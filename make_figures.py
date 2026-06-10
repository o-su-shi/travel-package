"""
ポートフォリオ用の図を4枚生成するスクリプト。
出力先: assets/
1. cleaning_before_after.png  … ProductPitched のクレンジング前後の水準数
2. target_distribution.png    … 成約/非成約での収入・年齢の分布
3. feature_importance.png     … 全特徴量で学習したLGBMの重要度 top15
4. score_improvement.png      … ベースライン特徴量 vs 全特徴量 のOOF AUC
全ラベルは日本語フォント欠け回避のため英語表記。
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.preprocessing.cleaning import run_cleaning, convert_product
from src.preprocessing.imputation import MissingValueImputer
from src.features.basic_features import FeatureEngineer
from src.features.categorical_features import CategoricalEncoder
from src.features.advanced_features import PostEncodingEngineer

os.makedirs("assets", exist_ok=True)
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 11
BLUE, ORANGE = "#3b6fb5", "#e08a2e"

print("Loading data...")
df_raw = pd.read_csv("data/raw/train.csv")

# ============================================================
# 図1: クレンジング前後（ProductPitched）
# ============================================================
print("Figure 1: cleaning before/after")
raw_vc = df_raw["ProductPitched"].value_counts()
clean_vc = df_raw["ProductPitched"].map(convert_product).value_counts()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].bar(range(len(raw_vc)), raw_vc.values, color=ORANGE)
axes[0].set_title(f"Before cleaning: {len(raw_vc)} distinct values\n(split by confusable chars)")
axes[0].set_xlabel("distinct category values")
axes[0].set_ylabel("count")
axes[0].set_xticks([])

axes[1].bar(clean_vc.index, clean_vc.values, color=BLUE)
axes[1].set_title(f"After cleaning: {len(clean_vc)} canonical categories")
axes[1].set_xlabel("ProductPitched")
axes[1].set_ylabel("count")
plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right")
plt.tight_layout()
plt.savefig("assets/cleaning_before_after.png", bbox_inches="tight")
plt.close()
print(f"   before={len(raw_vc)} values -> after={len(clean_vc)} categories")

# ============================================================
# パイプライン適用（図2,3,4で共通利用）
# ============================================================
print("Running full pipeline...")
df = run_cleaning(df_raw)
imputer = MissingValueImputer()
df = imputer.fit_transform(df)
fe = FeatureEngineer()
df = fe.fit_transform(df)
encoder = CategoricalEncoder(
    te_columns=["Product_Clean", "Convert_Designation", "MaritalStatus"],
    target_col="ProdTaken",
)
df = encoder.fit_transform(df)
post_fe = PostEncodingEngineer()
df = post_fe.fit_transform(df)

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]
cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()

# ============================================================
# 図2: ターゲット別の分布
# ============================================================
print("Figure 2: target-wise distribution")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for label, color, name in [(0, ORANGE, "Not purchased"), (1, BLUE, "Purchased")]:
    axes[0].hist(df.loc[y == label, "Convert_Income"].dropna(), bins=30,
                 alpha=0.6, color=color, label=name, density=True)
    axes[1].hist(df.loc[y == label, "Age_Clean"].dropna(), bins=20,
                 alpha=0.6, color=color, label=name, density=True)
axes[0].set_title("Monthly income by outcome")
axes[0].set_xlabel("MonthlyIncome")
axes[0].set_ylabel("density")
axes[0].legend()
axes[1].set_title("Age by outcome")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("density")
axes[1].legend()
plt.tight_layout()
plt.savefig("assets/target_distribution.png", bbox_inches="tight")
plt.close()

# ============================================================
# 図4の前に: OOF AUC を計算（ベースライン vs 全特徴量）
# ============================================================
def oof_auc(X_in, y_in, cat_in):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X_in))
    params = {"objective": "binary", "metric": "auc", "verbosity": -1,
              "seed": 42, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 3}
    for tr, va in skf.split(X_in, y_in):
        dtr = lgb.Dataset(X_in.iloc[tr], label=y_in.iloc[tr],
                          categorical_feature=cat_in, free_raw_data=False)
        dva = lgb.Dataset(X_in.iloc[va], label=y_in.iloc[va], reference=dtr,
                          categorical_feature=cat_in, free_raw_data=False)
        m = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(X_in.iloc[va], num_iteration=m.best_iteration)
    return roc_auc_score(y_in, oof)

print("Computing OOF AUC (this trains a few quick models)...")
# ベースライン: クレンジング不要だった生の数値列のみ
baseline_cols = ["CityTier", "NumberOfPersonVisiting", "NumberOfFollowups",
                 "PreferredPropertyStar", "Passport", "PitchSatisfactionScore"]
baseline_cols = [c for c in baseline_cols if c in X.columns]
auc_baseline = oof_auc(X[baseline_cols], y, [])
auc_full = oof_auc(X, y, cat_cols)
print(f"   baseline AUC = {auc_baseline:.4f}")
print(f"   full     AUC = {auc_full:.4f}")
print(f"   lift     = +{auc_full - auc_baseline:.4f}")

# ============================================================
# 図3: 特徴量重要度 top15（全特徴量モデルから）
# ============================================================
print("Figure 3: feature importance")
dall = lgb.Dataset(X, label=y, categorical_feature=cat_cols, free_raw_data=False)
model = lgb.train({"objective": "binary", "metric": "auc", "verbosity": -1,
                   "seed": 42, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 3},
                  dall, num_boost_round=300)
imp = pd.DataFrame({"feature": X.columns,
                    "importance": model.feature_importance(importance_type="gain")})
imp = imp.sort_values("importance", ascending=True).tail(15)
plt.figure(figsize=(8, 6))
plt.barh(imp["feature"], imp["importance"], color=BLUE)
plt.title("Feature importance (gain) - top 15")
plt.xlabel("gain")
plt.tight_layout()
plt.savefig("assets/feature_importance.png", bbox_inches="tight")
plt.close()

# ============================================================
# 図4: スコア改善
# ============================================================
print("Figure 4: score improvement")
plt.figure(figsize=(6, 4.5))
bars = plt.bar(["Baseline\n(raw numeric only)", "Full pipeline\n(cleaning + FE)"],
               [auc_baseline, auc_full], color=[ORANGE, BLUE], width=0.55)
plt.ylim(0.5, max(auc_full + 0.05, 0.8))
plt.ylabel("OOF AUC (5-fold)")
plt.title(f"Feature engineering lift: +{auc_full - auc_baseline:.3f}")
for b, v in zip(bars, [auc_baseline, auc_full]):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
             ha="center", va="bottom", fontweight="bold")
plt.tight_layout()
plt.savefig("assets/score_improvement.png", bbox_inches="tight")
plt.close()

print("\nDone. Saved 4 figures to assets/")
print(f"SUMMARY: baseline={auc_baseline:.4f}, full={auc_full:.4f}, lift=+{auc_full-auc_baseline:.4f}")
