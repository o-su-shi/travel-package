import numpy as np
import pandas as pd
import optuna
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

class CatBoostTrainer:
    def __init__(self, cat_features=None, n_splits=5, random_state=42):
        self.cat_features = cat_features if cat_features else []
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
                'iterations': trial.suggest_int('iterations', 100, 2000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'early_stopping_rounds': 100,
                'eval_metric': 'AUC',
                'random_seed': self.random_state,
                'verbose': False
            }

            scores = []
            for tr_idx, va_idx in self.cv.split(X, y):
                X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

                model = CatBoostClassifier(**params)
                model.fit(
                    X_tr, y_tr,
                    cat_features=self.cat_features,
                    eval_set=(X_va, y_va),
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

        # 固定パラメータの追加
        train_params = self.best_params.copy()
        train_params.update({
            'early_stopping_rounds': 100,
            'eval_metric': 'AUC',
            'random_seed': self.random_state,
            'verbose': False
        })

        self.oof_preds = np.zeros(len(X))
        self.models = []
        fold_aucs = []

        for fold, (tr_idx, va_idx) in enumerate(self.cv.split(X, y)):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

            model = CatBoostClassifier(**train_params)
            model.fit(
                X_tr, y_tr,
                cat_features=self.cat_features,
                eval_set=(X_va, y_va),
                verbose=False
            )

            preds = model.predict_proba(X_va)[:, 1]
            self.oof_preds[va_idx] = preds
            
            auc = roc_auc_score(y_va, preds)
            fold_aucs.append(auc)
            self.models.append(model)
            print(f"Fold {fold} AUC: {auc:.4f}")

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
                'eval_metric': 'AUC',
                'random_seed': self.random_state,
                'verbose': False
            })
            
            # 全データ学習では検証データがないため、early_stoppingは除外
            if 'early_stopping_rounds' in train_params:
                del train_params['early_stopping_rounds']

            # 全データでの最終学習
            final_model = CatBoostClassifier(**train_params)
            final_model.fit(
                X_train, y_train,
                cat_features=self.cat_features,
                verbose=False
            )
            
            # モデルの保持
            self.models.append(final_model)

            # 推論
            test_preds = final_model.predict_proba(X_test)[:, 1]
            return test_preds