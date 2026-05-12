import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

class ModelEnsemble:
    def __init__(self):
        self.weights = None

    def optimize_weights(self, oof_preds_list, y):
        """Nelder-Mead法による最適な重みの探索"""
        def objective(weights):
            # 重みの合計を1に正規化
            weights = np.array(weights) / np.sum(weights)
            # 各モデルのOOF予測値に重みを掛けて合算
            ensemble_preds = np.zeros_like(y, dtype=float)
            for i, preds in enumerate(oof_preds_list):
                ensemble_preds += weights[i] * preds
            # 最小化問題のためAUCを負にして返す
            return -roc_auc_score(y, ensemble_preds)

        # 初期値は均等な重み
        initial_weights = [1.0 / len(oof_preds_list)] * len(oof_preds_list)
        # 各重みは0以上1以下
        bounds = [(0, 1)] * len(oof_preds_list)

        result = minimize(objective, initial_weights, method='Nelder-Mead', bounds=bounds)
        self.weights = result.x / np.sum(result.x)
        
        ensemble_oof_preds = np.zeros_like(y, dtype=float)
        for i, preds in enumerate(oof_preds_list):
            ensemble_oof_preds += self.weights[i] * preds
            
        print(f"Optimized Weights: {self.weights}")
        print(f"Ensemble OOF AUC: {roc_auc_score(y, ensemble_oof_preds):.4f}")
        return self.weights

    def predict(self, test_preds_list):
        """最適化された重みを用いたテストデータの推論"""
        if self.weights is None:
            raise ValueError("先に optimize_weights() を実行してください。")
            
        ensemble_test_preds = np.zeros_like(test_preds_list[0], dtype=float)
        for i, preds in enumerate(test_preds_list):
            ensemble_test_preds += self.weights[i] * preds
        return ensemble_test_preds