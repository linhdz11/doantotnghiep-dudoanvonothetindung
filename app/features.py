"""Tiền xử lý giống hệt notebook `prediction-of-credit-card-default.ipynb` (Chương 3 + add_bill_trend mục 4.x)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
METADATA = json.loads((ROOT / 'models' / 'metadata.json').read_text(encoding='utf-8'))
INPUT_COLUMNS = METADATA['input_columns']
THRESHOLD = METADATA['threshold']
BAND_EDGES = METADATA['risk_bands']['edges']
BAND_LABELS = METADATA['risk_bands']['labels']

BILL_COLS = [f'BILL_AMT{i}' for i in range(1, 7)]
PAY_AMT_COLS = [f'PAY_AMT{i}' for i in range(1, 7)]
PAY_STATUS_COLS = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
RAW_COLUMNS = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE'] + PAY_STATUS_COLS + BILL_COLS + PAY_AMT_COLS


def merge_categories(df):
    """Gộp EDUCATION {0,5,6} -> 4 và MARRIAGE 0 -> 3 (notebook cell 19, 24)."""
    df = df.copy()
    df['EDUCATION'] = np.where(df['EDUCATION'].isin([0, 5, 6]), 4, df['EDUCATION'])
    df['MARRIAGE'] = np.where(df['MARRIAGE'] == 0, 3, df['MARRIAGE'])
    return df


def add_bill_trend(df):
    """Thêm BILL_TREND = (BILL_AMT1 - BILL_AMT6) / (|BILL_AMT6| + 1), tính theo từng dòng."""
    return df.assign(BILL_TREND=(df['BILL_AMT1'] - df['BILL_AMT6']) / (df['BILL_AMT6'].abs() + 1))


def build_features(df_raw):
    """23 biến gốc -> 42 cột đầu vào của final_model, đúng thứ tự metadata['input_columns']."""
    data = merge_categories(df_raw[RAW_COLUMNS].astype('int64'))
    data['AVG_BILL_AMT'] = data[BILL_COLS].mean(axis=1)
    data['AVG_PAY_AMT'] = data[PAY_AMT_COLS].mean(axis=1)
    data['PAY_RATIO'] = data['AVG_PAY_AMT'] / data['AVG_BILL_AMT'].clip(lower=1)
    data['NUM_MONTHS_DELAYED'] = (data[PAY_STATUS_COLS] > 0).sum(axis=1)
    data['MAX_DELAY'] = data[PAY_STATUS_COLS].max(axis=1)
    data['CREDIT_UTIL'] = data['AVG_BILL_AMT'] / (data['LIMIT_BAL'] + 1)
    for c in PAY_STATUS_COLS:
        data[f'{c}_nouse'] = (data[c] == -2).astype(int)
        data[f'{c}_revolving'] = (data[c] == 0).astype(int)
    return add_bill_trend(data)[INPUT_COLUMNS]


def risk_band(p):
    """Thấp: p < 0,10; Trung bình: 0,10 <= p < 0,30; Cao: p >= 0,30."""
    return BAND_LABELS[int(np.searchsorted(BAND_EDGES, p, side='right'))]


# (giá trị nhỏ nhất, lớn nhất) cho phép; None = không giới hạn
RANGES = {'LIMIT_BAL': (10_000, 1_000_000), 'SEX': (1, 2), 'EDUCATION': (0, 6), 'MARRIAGE': (0, 3), 'AGE': (21, 79),
          **{c: (-2, 8) for c in PAY_STATUS_COLS}, **{c: (None, None) for c in BILL_COLS},
          **{c: (0, None) for c in PAY_AMT_COLS}}


def validate_raw(df):
    """Trả danh sách lỗi (tiếng Việt); rỗng nghĩa là hợp lệ."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        return [f"Thiếu cột: {', '.join(missing)}"]
    errors = []
    for c, (lo, hi) in RANGES.items():
        s = pd.to_numeric(df[c], errors='coerce')
        if s.isna().any():
            errors.append(f"{c}: có giá trị trống hoặc không phải số (dòng {', '.join(map(str, s[s.isna()].index[:5]))})")
            continue
        if (s != s.round()).any():
            errors.append(f"{c}: phải là số nguyên")
        bad = s[(s < (-np.inf if lo is None else lo)) | (s > (np.inf if hi is None else hi))]
        if len(bad):
            rng = f"{'' if lo is None else lo}…{'' if hi is None else hi}"
            errors.append(f"{c}: giá trị ngoài khoảng cho phép [{rng}] (ví dụ {bad.iloc[0]:g})")
    return errors


def describe_customer(row):
    """Mô tả ngắn tiếng Việt từ 23 biến gốc, ví dụ 'Trễ hạn 2 tháng (tháng 9), trễ 3/6 tháng, dùng 85% hạn mức'."""
    pay = [int(row[c]) for c in PAY_STATUS_COLS]
    bills = [row[c] for c in BILL_COLS]
    n_late = sum(p > 0 for p in pay)
    if pay[0] > 0:
        parts = [f"Trễ hạn {pay[0]} tháng (tháng 9)"]
    elif all(p == -2 for p in pay):
        parts = ["Không phát sinh chi tiêu 6 tháng"]
    else:
        parts = [{-2: "Không chi tiêu tháng 9", -1: "Trả đủ tháng 9", 0: "Trả tối thiểu tháng 9"}[pay[0]]]
    if n_late:
        parts.append(f"trễ {n_late}/6 tháng")
    avg_bill = sum(bills) / 6
    parts.append(f"dùng {avg_bill / (row['LIMIT_BAL'] + 1):.0%} hạn mức".replace('.', ','))
    if avg_bill > 0:
        parts.append(f"trả {sum(row[c] for c in PAY_AMT_COLS) / 6 / avg_bill:.0%} sao kê".replace('.', ','))
    return ', '.join(parts)
