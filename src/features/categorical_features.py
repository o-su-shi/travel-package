import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import List

class CategoricalEncoder:
    """
    カテゴリ変数のエンコーディング（Bayesian Target Encoding & Label Encoding）を行うクラス。
    学習時に算出されたマッピング辞書（状態）を保持する。
    """
    def __init__(self, te_columns: List[str], target_col: str, prior_weight: float = 100, n_splits: int = 5):
        self.te_columns = te_columns
        self.target_col = target_col
        self.prior_weight = prior_weight
        self.n_splits = n_splits
        
        # 推論時に使用するグローバル平均のマッピングを保持する辞書
        self.global_mapping_dict = {}
        self.global_mean_all = None

        # 順序カテゴリの定義
        self.rank_map = {'Executive': 4, 'VP': 3, 'AVP': 2, 'Senior Manager': 1, 'Manager': 0}
        self.age_order = ['18-24', '25-34', '35-44', '45-54', '55-60']
        self.income_order = ['Low', 'MidLow', 'MidHigh', 'High']
        self.dur_order = ['0-5', '6-10', '11-15', '16-30', '31-60']

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        学習データ用：StratifiedKFoldを用いてOOFのTE値を算出しつつ、
        テスト・推論データ用のグローバル平均値を計算して保持する。
        """
        df_out = df.copy()
        self.global_mean_all = df_out[self.target_col].mean()

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for col in self.te_columns:
            bte_vals = np.full(df_out.shape[0], np.nan, dtype=float)
            
            # 1. OOF (Out-of-Fold) エンコーディング
            for train_idx, val_idx in skf.split(df_out, df_out[self.target_col]):
                fold_tr = df_out.iloc[train_idx]
                fold_va = df_out.iloc[val_idx]
                
                local_mean = fold_tr[self.target_col].mean()
                cat_stats = fold_tr.groupby(col)[self.target_col].agg(["mean", "count"])
                smooth = (cat_stats["mean"] * cat_stats["count"] + local_mean * self.prior_weight) / (cat_stats["count"] + self.prior_weight)
                
                bte_vals[val_idx] = fold_va[col].map(smooth).fillna(local_mean).values

            df_out[f"{col}_bte"] = bte_vals

            # 2. 推論用のグローバル統計量を計算・保存
            all_stats = df_out.groupby(col)[self.target_col].agg(["mean", "count"])
            global_smooth = (all_stats["mean"] * all_stats["count"] + self.global_mean_all * self.prior_weight) / (all_stats["count"] + self.prior_weight)
            self.global_mapping_dict[col] = global_smooth.to_dict()

        return self._apply_label_encoding(df_out)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        テスト/推論データ用：fit時に保持したグローバル平均を用いてTEを適用する。
        """
        df_out = df.copy()

        for col in self.te_columns:
            # 保持している辞書でマッピング。未知のカテゴリは全体の平均値で補完。
            df_out[f"{col}_bte"] = df_out[col].map(self.global_mapping_dict.get(col, {})).fillna(self.global_mean_all)

        # 目的変数列は不要になるため削除（テストデータに存在する場合のみ）
        if self.target_col in df_out.columns:
            df_out = df_out.drop(columns=[self.target_col])

        return self._apply_label_encoding(df_out)

    def _apply_label_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        順序変数のラベルエンコーディングを実行する内部関数
        """
        # TEに使用した列の元データを上書きせず、別列（またはそのまま）にする場合はここを調整
        if 'Convert_Designation' in df.columns:
            df['Convert_Designation_label'] = df['Convert_Designation'].map(self.rank_map)
        
        if 'CityTier' in df.columns:
            df['CityTier'] = df['CityTier'].astype(int)

        if 'age_bucket' in df.columns:
            age_dtype = pd.CategoricalDtype(categories=self.age_order, ordered=True)
            df['age_bucket_label'] = df['age_bucket'].astype(age_dtype).cat.codes

        if 'income_qcut' in df.columns:
            income_dtype = pd.CategoricalDtype(categories=self.income_order, ordered=True)
            df['income_qcut_label'] = df['income_qcut'].astype(income_dtype).cat.codes

        if 'duration_bucket' in df.columns:
            dur_dtype = pd.CategoricalDtype(categories=self.dur_order, ordered=True)
            df['duration_bucket_label'] = df['duration_bucket'].astype(dur_dtype).cat.codes

        return df