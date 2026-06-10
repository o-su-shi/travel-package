import re
import unicodedata
import numpy as np
import pandas as pd
from typing import Any
from typing import Optional, Union

# ==========================================
# 定数・マッピング定義
# ==========================================
KANJI_MAP = {
    '〇': 0, '一': 1, '二': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9
}

PRODUCT_REPLACE_MAP = str.maketrans({
    'В': 'B', 'Β': 'B', 'Ᏼ': 'B', 'ᗷ': 'B', '𐊡': 'B',
    'Ꭰ': 'D', 'ᗞ': 'D', 'ꓷ': 'D',
    'ｅ': 'e', 'е': 'e', 'ҽ': 'e', '℮': 'e',
    'ⅼ': 'l', 'ӏ': 'l', '|': 'l',
    'ᑌ': 'U', 'ᴜ': 'u',
    'х': 'x', '×': 'x', '⨯': 'x',
    'ϲ': 'c', 'с': 'c', '𝘤': 'c', '𝖈': 'c', 'ƈ': 'c', 'ς': 'c',
    'ꓢ': 'S', 'Տ': 'S', 'Ѕ': 'S', 'Ⴝ': 'S',
    'Ε': 'E', 'Ι': 'I', 'Α': 'A', 'С': 'C', 'ꓲ': 'I',
    'ı': 'i', 'ո': 'n'
})

CONFUSABLE_MAP = str.maketrans({
    'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Ζ': 'Z', 'Η': 'H', 'Ι': 'I', 'Κ': 'K', 'Μ': 'M', 'Ν': 'N', 'Ο': 'O', 'Ρ': 'P', 'Τ': 'T', 'Υ': 'Y', 'Χ': 'X',
    'α': 'a', 'β': 'b', 'γ': 'y', 'δ': 'd', 'ε': 'e', 'ι': 'i', 'κ': 'k', 'μ': 'm', 'ν': 'v', 'ο': 'o', 'ρ': 'p', 'τ': 't', 'υ': 'u', 'χ': 'x',
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H', 'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T', 'Υ': 'Y',
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x',
    'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E', 'Ｆ': 'F', 'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J', 'Ｋ': 'K', 'Ｌ': 'L', 'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O', 'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R', 'Ｓ': 'S', 'Ｔ': 'T', 'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X', 'Ｙ': 'Y', 'Ｚ': 'Z',
    'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e', 'ｆ': 'f', 'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j', 'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o', 'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r', 'ｓ': 's', 'ｔ': 't', 'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x', 'ｙ': 'y', 'ｚ': 'z',
    '×': 'x', '𝙧': 'r', 'ѵ': 'v', 'Ѕ': 'S', 'Տ': 'T',
})

CANON_TITLE = {
    "executive": "Executive",
    "manager": "Manager",
    "senior manager": "Senior Manager",
    "avp": "AVP",
    "vp": "VP",
}

MARITAL_MAP = {
    "結婚済み": "Married", "既婚": "Married",
    "離婚済み": "Divorced", "バツイチ": "Divorced",
    "未婚": "Unmarried", "独身": "Single",
}

CAR_MAP = {
    "車所持": 1, "自動車所有": 1, "乗用車所持": 1,
    "車未所持": 0, "自動車未所有": 0, "乗用車なし": 0,
    "車あり": 1, "車保有": 1, "自家用車あり": 1, "車有": 1,
    "車なし": 0, "車保有なし": 0, "自家用車なし": 0, "車無し": 0,
}

CHILD_NONE = {"子供なし", "子供無し", "こどもなし", "無子", "子供ゼロ", "子供0人", "子どもゼロ", "非育児家庭"}
CHILD_UNKNOWN = {"子の数不詳", "子育て状況不明", "子供の数不明", "わからない", "不明"}


# ==========================================
# 変換ロジック（単一データに対する処理）
# ==========================================

def _kanji_to_int(s: str) -> Optional[int]:
    if not isinstance(s, str):
        return None
    total, num = 0, 0
    for char in s:
        if char == '十':
            if num == 0: num = 1
            total += num * 10
            num = 0
        elif char in KANJI_MAP:
            num = KANJI_MAP[char]
        else:
            return None
    return total + num

def clean_age(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("歳", "").replace("才", "").replace("際", "")
    
    if s.isdigit():
        age = int(s)
    elif s.endswith("代") and s[:-1].isdigit():
        age = int(s[:-1]) + 5
    else:
        age = _kanji_to_int(s)

    if age is None:
        return np.nan
    return float(max(18, min(60, age)))

def convert_duration(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    x_str = str(x).strip()
    if "分" in x_str:
        return float(x_str.replace("分", ""))
    elif "秒" in x_str:
        try:
            return round(float(x_str.replace("秒", "")) / 60, 1)
        except ValueError:
            return np.nan
    return np.nan

def convert_gender(x: Any) -> Optional[str]:
    if isinstance(x, str):
        x = unicodedata.normalize("NFKC", x).casefold().replace(" ", "").replace("　", "").capitalize()
        return x if x in ["Male", "Female"] else None
    return None

def convert_product(x: Any) -> Optional[str]:
    if isinstance(x, str):
        x = unicodedata.normalize("NFKC", x).translate(PRODUCT_REPLACE_MAP).casefold().replace("　", " ").title()
        valid_products = ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
        return x if x in valid_products else None
    return None

def convert_trips(x: Union[str, int, float]) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.isdigit():
        return float(s)
    
    patterns = [
        (r"年に(\d+)回", 1),
        (r"半年に(\d+)回", 2),
        (r"四半期に(\d+)回", 4),
        (r"月に(\d+)回", 12),
        (r"週に(\d+)回", 52)
    ]
    for pattern, multiplier in patterns:
        m = re.fullmatch(pattern, s)
        if m:
            return float(int(m.group(1)) * multiplier)
    return np.nan

def normalize_designation(raw: str) -> str:
    if pd.isna(raw):
        return raw
    s = unicodedata.normalize("NFKC", raw).translate(CONFUSABLE_MAP)
    key = re.sub(r"\s+", " ", s).strip().casefold()
    key = CANON_TITLE.get(key, key.title())
    return 'Senior Manager' if key == 'Tenior Manager' else key

def convert_income(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(round(x))
    
    s = str(x).strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(round(float(s)))
    
    m = re.fullmatch(r"月収(\d+(?:\.\d+)?)万円", s)
    if m:
        return float(round(float(m.group(1)) * 10_000))
    return np.nan

def parse_customer_info(raw: Any) -> pd.Series:
    if pd.isna(raw):
        return pd.Series([np.nan, np.nan, np.nan])
    
    s = unicodedata.normalize("NFKC", str(raw))
    s = re.sub(r"[、,，／/・\t]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    ms = next((v for k, v in MARITAL_MAP.items() if k in s), np.nan)
    car = next((v for k, v in CAR_MAP.items() if k in s), np.nan)
    
    if any(key in s for key in CHILD_NONE):
        child = 0.0
    elif any(key in s for key in CHILD_UNKNOWN):
        child = np.nan
    else:
        m = re.search(r"(?:子[供ども]|こども)?\s*(\d+)人|(\d+)児", s)
        child = float(m.group(1) or m.group(2)) if m else np.nan

    return pd.Series([ms, car, child])


# ==========================================
# データフレーム適用パイプライン
# ==========================================

def run_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    データフレーム全体に対してクリーニング処理を適用し、新しい列を追加・更新したDataFrameを返す。
    """
    df = df.copy()
    
    df["Age_Clean"] = df["Age"].map(clean_age).astype("Int64")
    df["Duration_Clean"] = df["DurationOfPitch"].map(convert_duration)
    df["Gender_Clean"] = df["Gender"].map(convert_gender)
    df["Product_Clean"] = df["ProductPitched"].map(convert_product)
    df["Convert_Trips"] = df["NumberOfTrips"].map(convert_trips)
    df["Convert_Designation"] = df["Designation"].map(normalize_designation)
    df["Convert_Income"] = df["MonthlyIncome"].map(convert_income)
    
    # customer_infoのパース展開
    if "customer_info" in df.columns:
        parsed_cols = ["MaritalStatus", "CarOwner", "NumChildren"]
        df[parsed_cols] = df["customer_info"].apply(parse_customer_info)

    # 外れ値・異常値の補正
    if "NumberOfFollowups" in df.columns:
        mask = df['NumberOfFollowups'] >= 100
        df.loc[mask, 'NumberOfFollowups'] = df.loc[mask, 'NumberOfFollowups'] / 100
        
    return df