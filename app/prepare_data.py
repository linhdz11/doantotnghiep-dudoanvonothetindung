"""Chạy một lần: tái tạo tập test như notebook, kiểm tra khớp số, lưu dữ liệu cho app vào app/data/."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (ROOT, RAW_COLUMNS, INPUT_COLUMNS, THRESHOLD, BILL_COLS, PAY_AMT_COLS, PAY_STATUS_COLS,
                      build_features, risk_band, describe_customer)

RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parent / 'data'
DATA_DIR.mkdir(exist_ok=True)

# ---- Tái tạo notebook, từng bước như các cell 5, 7, 13, 19, 24, 26, 57, 94 ----
uci = fetch_ucirepo(id=350)
col_names = dict(zip(uci.variables['name'], uci.variables['description']))
col_names['Y'] = 'Default'
data = pd.concat([uci.data.features, uci.data.targets], axis=1).rename(columns=col_names)
assert list(data.columns) == RAW_COLUMNS + ['Default'], data.columns
data = data.drop_duplicates().reset_index(drop=True)
assert len(data) == 29965, len(data)
data[['SEX', 'EDUCATION', 'MARRIAGE']] = data[['SEX', 'EDUCATION', 'MARRIAGE']].astype('int64')
data['EDUCATION'] = np.where(data['EDUCATION'] == 5, 4, data['EDUCATION'])
data['EDUCATION'] = np.where(data['EDUCATION'] == 6, 4, data['EDUCATION'])
data['EDUCATION'] = np.where(data['EDUCATION'] == 0, 4, data['EDUCATION'])
data['MARRIAGE'] = np.where(data['MARRIAGE'] == 0, 3, data['MARRIAGE'])
raw = data.copy()  # 23 biến gốc (đã gộp) + Default

data['AVG_BILL_AMT'] = data[BILL_COLS].mean(axis=1)
data['AVG_PAY_AMT'] = data[PAY_AMT_COLS].mean(axis=1)
data['PAY_RATIO'] = data['AVG_PAY_AMT'] / data['AVG_BILL_AMT'].clip(lower=1)
data['NUM_MONTHS_DELAYED'] = (data[PAY_STATUS_COLS] > 0).sum(axis=1)
data['MAX_DELAY'] = data[PAY_STATUS_COLS].max(axis=1)
data['CREDIT_UTIL'] = data['AVG_BILL_AMT'] / (data['LIMIT_BAL'] + 1)
for c in PAY_STATUS_COLS:
    data[f'{c}_nouse'] = (data[c] == -2).astype(int)
    data[f'{c}_revolving'] = (data[c] == 0).astype(int)
X, y = data.drop('Default', axis=1), data['Default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
X_test4 = X_test.assign(BILL_TREND=(X_test['BILL_AMT1'] - X_test['BILL_AMT6']) / (X_test['BILL_AMT6'].abs() + 1))
assert list(X_test4.columns) == INPUT_COLUMNS

# ---- Dữ liệu cho app ----
test_raw = raw.loc[X_test.index].rename_axis('ID').reset_index()
assert len(test_raw) == 5993
test_raw.to_csv(DATA_DIR / 'test_raw.csv', index=False)

model = joblib.load(ROOT / 'models' / 'final_model.joblib')
proba_nb = model.predict_proba(X_test4)[:, 1]
X_app = build_features(test_raw)
proba = model.predict_proba(X_app)[:, 1]
pred = model.predict(X_app)

# ---- Kiểm tra khớp notebook ----
max_diff = np.abs(proba - proba_nb).max()
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
acc = accuracy_score(y_test, pred)
print(f"max |proba_app - proba_notebook| = {max_diff:.2e}")
print(f"Accuracy = {acc:.4f}, TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}")
checks = {'proba khớp 1e-12': max_diff <= 1e-12, 'Accuracy 0.6790': round(acc, 4) == 0.6790,
          'TP 940': tp == 940, 'FP 1538': fp == 1538, 'pred = proba >= ngưỡng': (pred == (proba >= THRESHOLD)).all()}
failed = [k for k, ok in checks.items() if not ok]
if failed:
    sys.exit(f"LỖI: không khớp notebook: {failed}")
print("Tất cả kiểm tra khớp notebook: OK")

preds = test_raw.assign(proba=proba, pred=pred, band=[risk_band(p) for p in proba])
preds.to_csv(DATA_DIR / 'test_predictions.csv', index=False)

# ---- 20 khách hàng mẫu ----
OUTCOME = {(1, 1): 'TP – phát hiện đúng vỡ nợ', (0, 0): 'TN – đúng, không vỡ nợ',
           (0, 1): 'FP – cảnh báo nhầm', (1, 0): 'FN – bỏ sót vỡ nợ'}
preds['Loại ca'] = [OUTCOME[(d, p)] for d, p in zip(preds['Default'], preds['pred'])]
preds['code'] = preds['Loại ca'].str[:2]

fixed = {19965: 'Ca trong báo cáo (mục 4.6b): vỡ nợ, rủi ro cao',
         20558: 'Ca trong báo cáo (mục 4.6b): không vỡ nợ, rủi ro thấp'}
picked = [preds[preds['ID'] == i].assign(**{'Lý do chọn': r}) for i, r in fixed.items()]
PLAN = {'Thấp': {'TN': 5, 'FN': 1}, 'Trung bình': {'TP': 2, 'TN': 2, 'FP': 2, 'FN': 1}, 'Cao': {'TP': 3, 'FP': 2}}
pool = preds[~preds['ID'].isin(fixed)]
for band, cells in PLAN.items():
    for code, n in cells.items():
        cell = pool[(pool['band'] == band) & (pool['code'] == code)]
        # ponytail: các ô đều có hàng trăm ca nên không cần lấy bù từ ô khác; nếu dữ liệu đổi thì báo lỗi
        assert len(cell) >= n, f"Ô {band}/{code} chỉ có {len(cell)} ca"
        picked.append(cell.sample(n, random_state=RANDOM_STATE).assign(**{'Lý do chọn': f"Nhóm {band} – {code}, chọn ngẫu nhiên (random_state=42)"}))
samples = pd.concat(picked).drop(columns='code')
samples.insert(1, 'Mô tả', samples.apply(describe_customer, axis=1))
samples.to_csv(DATA_DIR / 'sample_customers.csv', index=False)
print(samples[['ID', 'Mô tả', 'proba', 'band', 'pred', 'Default', 'Loại ca']].to_string(index=False))
