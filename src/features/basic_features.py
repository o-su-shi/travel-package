import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class FeatureEngineer:
    """
    特徴量エンジニアリングを実行するクラス。
    学習データでfitされた統計量やモデル（状態）を保持し、推論時に一貫して適用する。
    """
    def __init__(self):
        self.lr_model = LinearRegression()
        self.income_qcut_bins = None
        self.mean_inc_by_agebin_map = {}
        self.train_age_mean = None
        
        # 削除対象の列
        self.drop_cols = [
            "id", "Age", "DurationOfPitch", "Gender", "ProductPitched", 
            "NumberOfTrips", "Designation", "MonthlyIncome", "customer_info"
        ]

    def fit(self, df: pd.DataFrame):
        """
        学習データから必要な境界値、平均値、回帰モデルを学習・保持する。
        """
        df_temp = df.copy()

        # 1. Age_Cleanの平均値（欠損補完用）
        self.train_age_mean = df_temp['Age_Clean'].mean()

        # 2. 年齢→収入の単回帰モデル学習
        mask = df_temp['Age_Clean'].notna() & df_temp['Convert_Income'].notna()
        X_lr = df_temp.loc[mask, ['Age_Clean']]
        y_lr = df_temp.loc[mask, 'Convert_Income']
        self.lr_model.fit(X_lr, y_lr)

        # 3. 収入の四分位ビニングの境界値（bins）を取得して保持
        _, bins = pd.qcut(df_temp['Convert_Income'], q=4, retbins=True, duplicates='drop')
        bins[0] = -np.inf  # 未知の下限に対応
        bins[-1] = np.inf  # 未知の上限に対応
        self.income_qcut_bins = bins

        # 4. 年齢バケットごとの平均収入を計算して辞書として保持
        df_temp['age_bucket'] = pd.cut(
            df_temp['Age_Clean'], 
            bins=[18, 25, 35, 45, 55, 60], 
            labels=['18-24','25-34','35-44','45-54','55-60'], 
            include_lowest=True
        )
        self.mean_inc_by_agebin_map = df_temp.groupby('age_bucket', observed=False)['Convert_Income'].mean().to_dict()

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        保持している状態を用いて特徴量を生成し、不要な列を削除する。
        """
        df_out = df.copy()

        # --- ① 年齢と収入による回帰特徴量 ---
        age_filled = df_out[['Age_Clean']].fillna(self.train_age_mean)
        df_out['pred_income_by_age'] = self.lr_model.predict(age_filled)
        df_out['income_age_resid'] = df_out['Convert_Income'] - df_out['pred_income_by_age']
        df_out['income_age_ratio'] = df_out['Convert_Income'] / (df_out['pred_income_by_age'] + 1e-6)

        # --- ② バケット化（ビニング） ---
        df_out['age_bucket'] = pd.cut(
            df_out['Age_Clean'], 
            bins=[18, 25, 35, 45, 55, 60], 
            labels=['18-24','25-34','35-44','45-54','55-60'], 
            include_lowest=True
        )
        df_out['duration_bucket'] = pd.cut(
            df_out['Duration_Clean'], 
            bins=[0, 5, 10, 15, 30, 60], 
            labels=['0-5','6-10','11-15','16-30','31-60']
        )
        # fitで得た境界値(bins)を適用
        df_out['income_qcut'] = pd.cut(
            df_out['Convert_Income'], 
            bins=self.income_qcut_bins, 
            labels=['Low','MidLow','MidHigh','High'], 
            include_lowest=True
        )

        # --- ③ 統計量マッピング ---
        df_out['mean_inc_agebin'] = df_out['age_bucket'].map(self.mean_inc_by_agebin_map).astype(float)
        df_out['income_agebin_inter'] = df_out['mean_inc_agebin'] * df_out['income_age_resid']

        # --- ④ 相互作用特徴量 ---
        if "CityTier" in df_out.columns:
            df_out["CityTier_income_inter"] = df_out["CityTier"] * df_out["Convert_Income"]
        df_out["age_income_interaction"] = df_out["Age_Clean"] * df_out["income_age_ratio"]

        # --- ⑤ 元の不要な列の削除 ---
        cols_to_drop = [c for c in self.drop_cols if c in df_out.columns]
        df_out.drop(columns=cols_to_drop, inplace=True)

        return df_out
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        学習データに対してfitとtransformを連続して実行する。
        """
        self.fit(df)
        return self.transform(df)