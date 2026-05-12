import argparse
import pandas as pd
from src.preprocessing.cleaning import run_cleaning
from src.preprocessing.imputation import MissingValueImputer
from src.features.basic_features import FeatureEngineer
from src.features.categorical_features import CategoricalEncoder
from src.features.advanced_features import PostEncodingEngineer
from src.models.lgb_trainer import LightGBMTrainer
from src.models.cat_trainer import CatBoostTrainer
from src.models.xgb_trainer import XGBoostTrainer
from src.models.ensemble import ModelEnsemble
from src.utils.visualization import plot_feature_importance

def main(n_trials: int, mode: str, model_type: str):
    print("Loading data...")
    df_train_raw = pd.read_csv("data/raw/train.csv")
    df_test_raw = pd.read_csv("data/raw/test.csv")

    print("Running preprocessing and feature engineering...")
    df_train = run_cleaning(df_train_raw)
    df_test = run_cleaning(df_test_raw)

    imputer = MissingValueImputer()
    df_train = imputer.fit_transform(df_train)
    df_test = imputer.transform(df_test)

    fe = FeatureEngineer()
    df_train = fe.fit_transform(df_train)
    df_test = fe.transform(df_test)

    te_cols = ["Product_Clean", "Convert_Designation", "MaritalStatus"]
    encoder = CategoricalEncoder(te_columns=te_cols, target_col="ProdTaken")
    df_train = encoder.fit_transform(df_train)
    df_test = encoder.transform(df_test)

    post_fe = PostEncodingEngineer()
    df_train = post_fe.fit_transform(df_train)
    df_test = post_fe.transform(df_test)

    X_train = df_train.drop(columns=["ProdTaken"])
    y_train = df_train["ProdTaken"]
    X_test = df_test
    
    obj_cols = X_train.select_dtypes(include=['category','object']).columns.tolist()
    
    # ---------------------------------------------------------
    # 単一モデル実行モード
    # ---------------------------------------------------------
    if model_type in ["lgbm", "catboost", "xgboost"]:
        if model_type == "lgbm":
            trainer = LightGBMTrainer(cat_cols=obj_cols, n_splits=5)
        elif model_type == "catboost":
            trainer = CatBoostTrainer(cat_features=obj_cols, n_splits=5)
        elif model_type == "xgboost":
            trainer = XGBoostTrainer(n_splits=5)
        
        print(f"Starting {model_type.upper()} Pipeline...")
        print(f"Starting Optuna Optimization (trials: {n_trials})...")
        trainer.optimize(X_train, y_train, n_trials=n_trials)
        
        if mode == "cv":
            print("Starting K-Fold Training...")
            trainer.train_cv(X_train, y_train)
            
            print("Generating Feature Importance Plot...")
            plot_feature_importance(trainer.models, X_train)
            
            if hasattr(trainer, "save_models"):
                trainer.save_models(f"saved_models/{model_type}/")
            print("Mode 'cv' completed.")
            
        elif mode == "submit":
            print("Starting Full Training and Prediction...")
            test_preds = trainer.train_full_and_predict(X_train, y_train, X_test)
            
            submission = pd.DataFrame({'id': df_test_raw['id'], 'ProdTaken': test_preds})
            submission.to_csv(f'submission_{model_type}.csv', index=False, header=False, float_format='%.6f')
            print(f"Submission file generated. Mode 'submit' completed.")

    # ---------------------------------------------------------
    # アンサンブル実行モード
    # ---------------------------------------------------------
    elif model_type == "ensemble":
        print("Starting Ensemble Pipeline (LGBM, CatBoost, XGBoost)...")
        trainers = {
            "lgbm": LightGBMTrainer(cat_cols=obj_cols, n_splits=5),
            "catboost": CatBoostTrainer(cat_features=obj_cols, n_splits=5),
            "xgboost": XGBoostTrainer(n_splits=5)
        }

        oof_preds_list = []
        test_preds_list = []

        for name, t in trainers.items():
            print(f"\n--- Training {name.upper()} ---")
            t.optimize(X_train, y_train, n_trials=n_trials)
            
            # アンサンブルの重み計算にはOOFが必須なため、cvは必ず実行する
            t.train_cv(X_train, y_train)
            oof_preds_list.append(t.oof_preds)

            if mode == "submit":
                # 各モデルのテスト推論結果を取得
                test_preds_list.append(t.train_full_and_predict(X_train, y_train, X_test))

        print("\n--- Optimizing Ensemble Weights ---")
        ensemble = ModelEnsemble()
        ensemble.optimize_weights(oof_preds_list, y_train)

        if mode == "submit":
            final_preds = ensemble.predict(test_preds_list)
            submission = pd.DataFrame({'id': df_test_raw['id'], 'ProdTaken': final_preds})
            submission.to_csv('submission_ensemble.csv', index=False, header=False, float_format='%.6f')
            print("Ensemble submission generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Travel Package Purchase Prediction Pipeline")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--mode", type=str, choices=["cv", "submit"], required=True)
    parser.add_argument("--model", type=str, choices=["lgbm", "catboost", "xgboost", "ensemble"], default="lgbm", help="Choose model backend or run ensemble")
    
    args = parser.parse_args()
    main(n_trials=args.trials, mode=args.mode, model_type=args.model)