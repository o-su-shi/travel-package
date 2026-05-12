import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

class XGBoostTrainer:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.best_params = None
        self.models = []
        self.oof_preds = None

    def optimize(self, X, y, n_trials=50):
        """Optunaによるハイパーパラメータ探索"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'eval_metric': 'auc',
                'random_state': self.random_state,
                'enable_categorical': True,
                'early_stopping_rounds': 100
            }

            scores = []
            for tr_idx, va_idx in self.cv.split(X, y):
                X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

                model = XGBClassifier(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False
                )
                preds = model.predict_proba(X_va)[:, 1]
                scores.append(roc_auc_score(y_va, preds))

            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        self.best_params = study.best_params
        return self.best_params

    def train_cv(self, X, y):
        """最適パラメータを用いたKFold学習"""
        if not self.best_params:
            raise ValueError("先に optimize() を実行してパラメータを決定してください。")

        train_params = self.best_params.copy()
        train_params.update({
            'eval_metric': 'auc',
            'random_state': self.random_state,
            'enable_categorical': True,
            'early_stopping_rounds': 100
        })

        self.oof_preds = np.zeros(len(X))
        self.models = []

        for fold, (tr_idx, va_idx) in enumerate(self.cv.split(X, y)):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

            model = XGBClassifier(**train_params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False
            )

            preds = model.predict_proba(X_va)[:, 1]
            self.oof_preds[va_idx] = preds
            self.models.append(model)
            print(f"Fold {fold} AUC: {roc_auc_score(y_va, preds):.4f}")

        print(f"OOF AUC: {roc_auc_score(y, self.oof_preds):.4f}")

    def predict(self, X_test):
        """学習済みモデル群によるアンサンブル推論"""
        test_preds = np.zeros(len(X_test))
        for model in self.models:
            test_preds += model.predict_proba(X_test)[:, 1] / self.n_splits
        return test_preds
    
    def train_full_and_predict(self, X_train, y_train, X_test):
        """
        Optunaで得た最適パラメータを使用し、全訓練データで学習後、テストデータの推論結果を返す。
        """
        if not self.best_params:
            raise ValueError("最適化が実行されていません。先に optimize() を実行してください。")

        # 固定パラメータの統合
        train_params = self.best_params.copy()
        train_params.update({
            'eval_metric': 'auc',
            'random_state': self.random_state,
            'enable_categorical': True
        })
        
        # 全データ学習では検証データ(eval_set)がないため、early_stopping_roundsは除外
        if 'early_stopping_rounds' in train_params:
            del train_params['early_stopping_rounds']

        # 全データでの最終学習
        final_model = XGBClassifier(**train_params)
        final_model.fit(
            X_train, y_train,
            verbose=False
        )
        
        # モデルの保持
        self.models.append(final_model)

        # 推論
        test_preds = final_model.predict_proba(X_test)[:, 1]
        return test_preds