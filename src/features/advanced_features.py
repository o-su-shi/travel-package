import pandas as pd
from typing import List

class PostEncodingEngineer:
    """
    ターゲットエンコーディング後に実行すべき相互作用特徴量の生成と、
    モデル（LightGBM等）入力に向けた最終的なカテゴリ型変換を担うクラス。
    """
    def __init__(self):
        # 学習時のカテゴリ定義を保持する辞書
        self.category_mapping = {}

    def fit(self, df: pd.DataFrame):
        """学習データからカテゴリ変数の種類（水準）を記憶する"""
        df_temp = df.copy()
        
        # Object型 または Category型の列を抽出
        obj_cols = df_temp.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in obj_cols:
            df_temp[col] = df_temp[col].astype(str)
            # 学習データに存在するカテゴリ水準を記憶
            self.category_mapping[col] = pd.CategoricalDtype(
                categories=df_temp[col].unique(), ordered=False
            )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量生成とカテゴリ型の厳密な適用を行う"""
        df_out = df.copy()

        # --------------------------------------------------
        # 1. 相互作用・非線形特徴量の生成
        # --------------------------------------------------
        if "Convert_Designation_bte" in df_out.columns:
            df_out["desig_income_inter"] = df_out["Convert_Designation_bte"] * df_out["Convert_Income"]
            
        if "MaritalStatus_bte" in df_out.columns:
            df_out["mstatus_income_inter"] = df_out["MaritalStatus_bte"] * df_out["Convert_Income"]

        df_out["CityTier_income_inter"] = df_out["CityTier"] * df_out["Convert_Income"]
        df_out["age_income_interaction"] = df_out["Age_Clean"] * df_out["Convert_Income"]
        
        df_out["PitchSat_sq"] = df_out["PitchSatisfactionScore"] ** 2
        df_out["PitchSat_cube"] = df_out["PitchSatisfactionScore"] ** 3
        
        df_out["has_visitors"] = (df_out["NumberOfPersonVisiting"] >= 1).astype(int)
        df_out["pitch_trip_inter"] = df_out["PitchSatisfactionScore"] * df_out["Convert_Trips"]

        # --------------------------------------------------
        # 2. カテゴリ型の適用（未知のカテゴリはNaNになる安全設計）
        # --------------------------------------------------
        for col, dtype in self.category_mapping.items():
            if col in df_out.columns:
                # fit時に記憶した水準のみをカテゴリとして適用
                df_out[col] = df_out[col].astype(str).astype(dtype)

        return df_out
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """学習データに対してfitとtransformを連続して実行する"""
        self.fit(df)
        return self.transform(df)