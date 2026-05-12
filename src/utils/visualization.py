import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(models, X, top_n=30):
    """
    複数モデルの平均特徴量重要度を計算し可視化する
    """
    import numpy as np
    
    # モデルリストから重要度を行列化して平均を取得
    imp_mat = np.vstack([model.feature_importance() for model in models])
    imp_mean = imp_mat.mean(axis=0)
    
    fi = pd.DataFrame({'feature': X.columns, 'importance': imp_mean})
    fi = fi.sort_values('importance', ascending=True).tail(top_n)
    
    plt.figure(figsize=(8, 10))
    plt.barh(fi['feature'], fi['importance'])
    plt.title('Average Feature Importances')
    plt.tight_layout()
    plt.show()