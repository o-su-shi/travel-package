import pandas as pd
from sklearn.impute import SimpleImputer
from typing import List

class MissingValueImputer:
    def __init__(self):
        self.imp_median = SimpleImputer(strategy='median')
        self.imp_zero = SimpleImputer(strategy='constant', fill_value=0)
        self.imp_mode = SimpleImputer(strategy='most_frequent')

        self.num_median_cols: List[str] = ['Age_Clean', 'Duration_Clean', 'Convert_Trips', 'Convert_Income']
        self.num_zero_cols: List[str] = ['NumberOfFollowups']
        self.num_mode_cols: List[str] = ['TypeofContact', 'NumChildren']

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """学習データ用：欠損値の統計量を計算し、補完を適用する"""
        df_out = df.copy()
        df_out[self.num_median_cols] = self.imp_median.fit_transform(df_out[self.num_median_cols])
        df_out[self.num_zero_cols] = self.imp_zero.fit_transform(df_out[self.num_zero_cols])
        df_out[self.num_mode_cols] = self.imp_mode.fit_transform(df_out[self.num_mode_cols])
        return df_out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """テスト/推論データ用：保持した統計量で補完を適用する"""
        df_out = df.copy()
        df_out[self.num_median_cols] = self.imp_median.transform(df_out[self.num_median_cols])
        df_out[self.num_zero_cols] = self.imp_zero.transform(df_out[self.num_zero_cols])
        df_out[self.num_mode_cols] = self.imp_mode.transform(df_out[self.num_mode_cols])
        return df_out