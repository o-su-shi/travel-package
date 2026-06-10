"""図2（ターゲット別分布）のみ再生成。x軸の重なりを解消。"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from src.preprocessing.cleaning import run_cleaning
from src.preprocessing.imputation import MissingValueImputer

BLUE, ORANGE = "#3b6fb5", "#e08a2e"
plt.rcParams["font.size"] = 11

df = run_cleaning(pd.read_csv("data/raw/train.csv"))
df = MissingValueImputer().fit_transform(df)
y = df["ProdTaken"]

inc = df["Convert_Income"].dropna()
lo, hi = inc.quantile(0.01), inc.quantile(0.99)  # 極端な外れ値で軸が伸びるのを防ぐ

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for label, color, name in [(0, ORANGE, "Not purchased"), (1, BLUE, "Purchased")]:
    sub = df.loc[y == label, "Convert_Income"].dropna()
    sub = sub[(sub >= lo) & (sub <= hi)]
    axes[0].hist(sub, bins=30, alpha=0.6, color=color, label=name, density=True)
    axes[1].hist(df.loc[y == label, "Age_Clean"].dropna(), bins=20,
                 alpha=0.6, color=color, label=name, density=True)

# 収入軸: 千円単位(k)表記 + 目盛り数を抑制 + 軽い回転
axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
axes[0].xaxis.set_major_locator(MaxNLocator(nbins=6))
axes[0].tick_params(axis="x", rotation=0)
axes[0].set_title("Monthly income by outcome")
axes[0].set_xlabel("MonthlyIncome (k = thousand)")
axes[0].set_ylabel("density")
axes[0].legend()
axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

axes[1].set_title("Age by outcome")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("density")
axes[1].legend()

plt.tight_layout()
plt.savefig("assets/target_distribution.png", bbox_inches="tight", dpi=130)
print("Saved assets/target_distribution.png  (income clipped to 1-99%:",
      f"{lo:.0f}-{hi:.0f})")
