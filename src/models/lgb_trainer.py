import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib
import os
from typing import Dict, List, Tuple

class LightGBMTrainer:
    """
    LightGBMのハイパーパラメータ探索、K-Fold学習、モデル保存を担うクラス
    """
    def __init__(self, cat_cols: List[str], n_splits: int = 5, random_state: int = 42):
        self.cat_cols = cat_cols
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # 状態保持
        self.best_params = {}
        self.models = []
        self.oof_preds = None

    def optimize(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 10) -> Dict:
        """Optunaを用いたハイパーパラメータ探索"""
        
        def objective(trial):
            param = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'seed': self.random_state,
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 16, 128, step=16),
                'max_depth': trial.suggest_int('max_depth', 1, 3),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100, step=10),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.2, 1.0, step=0.1),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0, step=0.1),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
            }

            aucs = []
            for tr_idx, va_idx in self.cv.split(X, y):
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

                dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=self.cat_cols, free_raw_data=False)
                dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain, categorical_feature=self.cat_cols, free_raw_data=False)

                gbm = lgb.train(
                    param,
                    dtrain,
                    num_boost_round=1000,
                    valid_sets=[dvalid],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )

                preds = gbm.predict(X_va, num_iteration=gbm.best_iteration)
                aucs.append(roc_auc_score(y_va, preds))

            return np.mean(aucs)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        return self.best_params

    def train_cv(self, X: pd.DataFrame, y: pd.DataFrame, params: Dict = None) -> np.ndarray:
        """
        最適化されたパラメータ（または指定パラメータ）で本番用モデルを学習し、
        OOF予測値を返すとともにモデルをクラス内に保持する
        """
        train_params = params if params else self.best_params
        if not train_params:
            raise ValueError("Parameters not found. Run optimize() first or pass params.")
            
        # 必須固定パラメータの追加
        train_params.update({
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'seed': self.random_state
        })

        self.oof_preds = np.zeros(len(X))
        self.models = []

        for fold, (tr_idx, va_idx) in enumerate(self.cv.split(X, y)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=self.cat_cols, free_raw_data=False)
            dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain, categorical_feature=self.cat_cols, free_raw_data=False)

            gbm = lgb.train(
                train_params,
                dtrain,
                num_boost_round=3000,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
            )

            self.models.append(gbm)
            self.oof_preds[va_idx] = gbm.predict(X_va, num_iteration=gbm.best_iteration)
            
            print(f"Fold {fold+1} AUC: {roc_auc_score(y_va, self.oof_preds[va_idx]):.4f}")

        print(f"Overall OOF AUC: {roc_auc_score(y, self.oof_preds):.4f}")
        return self.oof_preds

    def save_models(self, save_dir: str):
        """学習済みモデル群を指定ディレクトリに保存"""
        os.makedirs(save_dir, exist_ok=True)
        for i, model in enumerate(self.models):
            joblib.dump(model, os.path.join(save_dir, f'lgb_fold_{i}.pkl'))

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """保持している全Foldモデルのアンサンブル（平均）による推論"""
        if not self.models:
            raise ValueError("No models trained. Run train_cv() first.")
        
        preds = np.zeros(len(X_test))
        for model in self.models:
            preds += model.predict(X_test, num_iteration=model.best_iteration) / len(self.models)
        return preds
    
    def train_full_and_predict(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
        """
        Optunaで得た最適パラメータを使用し、CVで最適イテレーション数を取得後、
        全訓練データで再学習し、テストデータの推論結果を返す。
        """
        if not self.best_params:
            raise ValueError("最適化が実行されていません。先に optimize() を実行してください。")

        # 固定パラメータの統合
        train_params = self.best_params.copy()
        train_params.update({
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': self.random_state
        })

        dtrain_full = lgb.Dataset(X_train, label=y_train, categorical_feature=self.cat_cols, free_raw_data=False)

        # CVで最適イテレーション数を取得
        cv_results = lgb.cv(
            params=train_params,
            train_set=dtrain_full,
            num_boost_round=3000,
            nfold=self.n_splits, # Optuna時とFold数を一致させる
            stratified=True,
            metrics='auc',
            seed=self.random_state,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False)
            ],
        )

        mean_key = [k for k in cv_results.keys() if k.endswith('-mean')][0]
        best_iter = len(cv_results[mean_key])
        print(f"Optimal num_boost_round by CV: {best_iter}")

        # 全データでの最終学習
        final_gbm = lgb.train(
            params=train_params,
            train_set=dtrain_full,
            num_boost_round=best_iter
        )
        
        # モデルの保持（必要に応じて）
        self.models.append(final_gbm)

        # 推論
        test_preds = final_gbm.predict(X_test, num_iteration=best_iter)
        return test_preds

