import pandas as pd

def reduce_memory_usage(df: pd.DataFrame, skip_cols: list = None) -> pd.DataFrame:
    """
    データフレームのメモリサイズを削減するために型をdowncastする関数。
    """
    df_out = df.copy()
    skip_cols = skip_cols or []

    # 整数列のダウンキャスト
    int_cols = [c for c in df_out.select_dtypes(include=['int64', 'int32']).columns if c not in skip_cols]
    df_out[int_cols] = df_out[int_cols].apply(pd.to_numeric, downcast='integer')

    # 浮動小数点列のダウンキャスト
    float_cols = [c for c in df_out.select_dtypes(include=['float64']).columns if c not in skip_cols]
    df_out[float_cols] = df_out[float_cols].apply(pd.to_numeric, downcast='float')

    return df_out