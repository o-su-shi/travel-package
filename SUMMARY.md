# 旅行パッケージ成約率予測コンペ まとめ

SIGNATE「旅行パッケージの成約率予測」（テーブルデータ・二値分類）の取り組みと、コード・特徴量エンジニアリングの根拠をまとめたドキュメント。

---

## 1. コンペ概要

| 項目 | 内容 |
|---|---|
| 課題 | 旅行会社の顧客DBから旅行パッケージの**成約(`ProdTaken`)を予測** |
| タイプ | テーブルデータ・二値分類 |
| 評価指標 | **AUC**（ROC曲線下面積、0.5=ランダム、1.0=完全） |
| データ規模 | train 3,549件 / test 3,555件 |
| クラス分布 | 成約=14.2% / 不成約=85.8%（**不均衡**） |
| 最大の特徴 | データに**表記揺れ・欠損・ノイズが意図的に混入**（実務想定）。「綺麗に整形してから予測」が課題の本質 |

背景ストーリー：旅行会社がマーケティングコスト削減のため、購入しそうな顧客を機械学習で予測したい。ただし度重なるDB仕様変更でデータは構造化されておらず、まず整形が必要。

ポイントは、**データクレンジングそのものがコンペの主眼**に据えられていること（キリル文字・ギリシャ文字混入、全角半角混在、和文表記など）。

---

## 2. データ仕様（元カラム）

| カラム | 型 | 説明 | 欠損数 |
|---|---|---|---|
| id | 数値 | 顧客ID | 0 |
| Age | カテゴリ | 年齢 | 100 |
| TypeofContact | カテゴリ | 連絡・接触方法 | 6 |
| CityTier | 数値 | 都市層 | 0 |
| DurationOfPitch | カテゴリ | セールス時間 | 121 |
| Occupation | カテゴリ | 職業 | 0 |
| Gender | カテゴリ | 性別 | 0 |
| NumberOfPersonVisiting | 数値 | 同行者数 | 0 |
| NumberOfFollowups | 数値 | フォローアップ回数 | 33 |
| ProductPitched | カテゴリ | セールスした商品種 | 0 |
| PreferredPropertyStar | 数値 | 希望ホテルランク | 0 |
| NumberOfTrips | カテゴリ | 年間旅行数 | 22 |
| Passport | 数値 | パスポート所持(0/1) | 0 |
| PitchSatisfactionScore | 数値 | セールス満足度 | 0 |
| Designation | カテゴリ | 役職 | 0 |
| MonthlyIncome | カテゴリ | 月収 | 56 |
| customer_info | カテゴリ | メモ（婚姻状況/車有無/子供同伴） | 0 |
| **ProdTaken** | 数値 | **契約状態(0:不成約/1:成約)** ＝目的変数 | 0 |

---

## 3. パイプライン全体像（`main.py`）

```
raw csv
 └─ run_cleaning            … 表記揺れ・ノイズの正規化（cleaning.py）
 └─ MissingValueImputer     … 欠損補完（imputation.py）
 └─ FeatureEngineer         … 仮説/回帰/ビニング特徴量（basic_features.py）
 └─ CategoricalEncoder      … Target/Label Encoding（categorical_features.py）
 └─ PostEncodingEngineer    … エンコード後の交互作用特徴量（advanced_features.py）
 └─ LGBM / CatBoost / XGBoost（Optuna最適化 + 5-fold CV）
 └─ ModelEnsemble           … OOFで重み最適化（Nelder-Mead）
```

実行コマンド：

```bash
python main.py --model {lgbm|catboost|xgboost|ensemble} --mode {cv|submit} --trials N
```

全ステップが `fit`（train統計の学習）/`transform`（適用）を分離した設計で、**train統計をtestへ漏らさない**（データリーク対策）。

---

## 4. データクレンジングの根拠（`cleaning.py`）

意図的ノイズに対し、列ごとに**ドメイン知識ベースの正規化関数**を定義。基本方針は `unicodedata.normalize("NFKC")`（全角半角・互換文字の統一）＋ `str.maketrans` による混同文字の置換の二段構え。

| 列 | 処理 | 根拠 |
|---|---|---|
| `Age` → `Age_Clean` | 「50歳/才/際」「○代→+5」「漢数字」を数値化、**18〜60にクリップ** | 単位ゆれ・全角ゆれ・異常年齢の補正 |
| `DurationOfPitch` → `Duration_Clean` | 「分」/「秒(÷60)」を分単位に統一 | 単位混在の解消 |
| `Gender` → `Gender_Clean` | NFKC正規化＋casefold→Male/Female以外はNaN | 表記ゆれ・不正値除去 |
| `ProductPitched` → `Product_Clean` | **混同文字マップ(キリル/ギリシャ/全角)を置換**しBasic〜Kingに正規化 | コンペ最大のノイズ列への対応 |
| `NumberOfTrips` → `Convert_Trips` | 「年に/半年に/四半期に/月に/週に○回」を**年間回数に換算** | 頻度表現を年換算で揃える |
| `Designation` → `Convert_Designation` | 混同文字マップ＋役職正規表現、`Tenior Manager→Senior Manager`修正 | OCR的誤字の補正 |
| `MonthlyIncome` → `Convert_Income` | 「月収○万円」を円に換算 | 単位表記の統一 |
| `customer_info` → `MaritalStatus / CarOwner / NumChildren` | フリーテキストから**婚姻状況/車所有/子供数**を抽出して3列に展開 | メモ欄から構造化特徴を獲得 |
| `NumberOfFollowups` | **100以上は÷100**で補正 | 桁ずれ外れ値の補正 |

主要な定数マップ：
- `PRODUCT_REPLACE_MAP` / `CONFUSABLE_MAP`：キリル・ギリシャ・全角の混同文字をラテン文字へ
- `KANJI_MAP`：漢数字→アラビア数字
- `MARITAL_MAP` / `CAR_MAP` / `CHILD_NONE` / `CHILD_UNKNOWN`：customer_info パース用の和文辞書

---

## 5. 欠損補完の根拠（`imputation.py`）

`SimpleImputer` を性質別に使い分け：

- **中央値**：`Age_Clean / Duration_Clean / Convert_Trips / Convert_Income`（連続値、外れ値に頑健）
- **0埋め**：`NumberOfFollowups`（フォローなし＝0が自然）
- **最頻値**：`TypeofContact / NumChildren`（カテゴリ・少数水準）

train で `fit` した統計を test に `transform` で適用＝リーク防止。

---

## 6. 特徴量エンジニアリングの根拠

進め方の方針（①元カラム追加 ②仮説ベース ③機械学習的 ④EDAベース ⑤欠損補完 ⑥外れ値対処 ⑦不要列削除）に沿って実装。

### ① 仮説/回帰ベース（`basic_features.py`）
- **年齢→収入の単回帰**を学習し、`income_age_resid`（残差）・`income_age_ratio`（比）を生成 → 「年齢の割に高/低収入か」という購買力シグナル
- **ビニング**：`age_bucket / duration_bucket / income_qcut`（qcut境界はtrainで固定し、未知値は±infで吸収）
- **年齢帯ごとの平均収入**マッピング + 残差との交互作用 `income_agebin_inter`
- **交互作用**：`CityTier_income_inter`（富裕都市×収入）、`age_income_interaction`
- 元の生カラム（id, Age, DurationOfPitch 等9列）を削除

### ② カテゴリエンコード（`categorical_features.py`）
- **Bayesian Target Encoding（平滑化、prior_weight=100）** を `Product_Clean / Convert_Designation / MaritalStatus` に適用
  - **StratifiedKFoldでOOF生成**しリーク防止、test用にはグローバル平滑平均(`global_mapping_dict`)を保持、未知カテゴリは全体平均で補完
- **順序ラベルエンコード**：役職(Executive=4…Manager=0)、年齢帯、収入帯、商談時間帯 → 順序情報を保持

### ③ エンコード後の交互作用（`advanced_features.py`）
- `desig_income_inter`（役職TE×収入）、`mstatus_income_inter`（婚姻TE×収入）
- `PitchSat_sq / PitchSat_cube`（満足度の非線形効果）
- `has_visitors`（同行者有無）、`pitch_trip_inter`（満足度×旅行頻度）
- 学習時のカテゴリ水準を記憶し、**未知カテゴリはNaN化する安全設計**

### 初期仮説メモとの対応
ドキュメントの初期仮説（収入・年齢・役職・フォローアップ数・満足度・都市層が成約に効く／交差特徴量 Age×Income, ProductPitched×Income など）が、ほぼそのまま特徴量として実装に落とし込まれている。

---

## 7. モデリングとアンサンブル

- **3モデル**：LightGBM / CatBoost / XGBoost、いずれも **Optuna**でハイパラ探索 + **5-fold StratifiedKFold**
  - LGBM：`max_depth 1〜3` と浅め＝過学習抑制重視（少データ・不均衡向け）
  - CatBoost/XGB：ネイティブにカテゴリ対応（`cat_features` / `enable_categorical=True`）
- **本番推論**：`lgb.cv`で最適イテレーション数を決めてから全データ再学習（`train_full_and_predict`）
- **アンサンブル**（`ensemble.py`）：各モデルのOOF予測に対し**Nelder-Mead法でAUC最大化する重み**を探索し、testに適用
- **成果物**：`saved_models/lgbm/` に5fold分のpkl、`submission_ensemble.csv`

---

## 8. この実装の強み・改善余地

### 強み
- **クレンジングがコンペの本質**という見立てが正確。列ごとに専用関数を用意した泥臭い対応が秀逸
- `fit/transform`分離・OOFターゲットエンコードで**リーク対策が徹底**されている
- 仮説 → 特徴量 → 交互作用の流れが一貫している

### 改善余地
- **不均衡対応**：`scale_pos_weight` / `class_weight` が未使用（14%の少数クラス向けに検討余地）
- **再現性**：Optuna の `create_study` に `sampler` のseed未固定で試行が再現しない
- **モデル管理**：`train_full_and_predict` がCV版とは別の全データ学習モデルを `self.models` に追記するため、用途が混在しないよう注意
- `PostEncodingEngineer` で `CityTier_income_inter` / `age_income_interaction` が `FeatureEngineer` と重複定義（後段で上書き）
