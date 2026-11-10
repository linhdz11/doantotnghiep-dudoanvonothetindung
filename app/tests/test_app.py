import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
from features import ROOT, RAW_COLUMNS, INPUT_COLUMNS, build_features, risk_band, validate_raw  # noqa: E402

MODEL = joblib.load(ROOT / 'models' / 'final_model.joblib')
TEST_RAW = pd.read_csv(APP / 'data' / 'test_raw.csv')


def proba(df):
    return MODEL.predict_proba(build_features(df))[:, 1]


def test_build_features_matches_saved_predictions():
    X = build_features(TEST_RAW)
    assert list(X.columns) == INPUT_COLUMNS
    saved = pd.read_csv(APP / 'data' / 'test_predictions.csv')
    assert (saved['ID'] == TEST_RAW['ID']).all()
    np.testing.assert_allclose(MODEL.predict_proba(X)[:, 1], saved['proba'], rtol=0, atol=1e-12)


@pytest.mark.parametrize('p, band', [(0.0, 'Thấp'), (0.0999, 'Thấp'), (0.10, 'Trung bình'),
                                     (0.2999, 'Trung bình'), (0.30, 'Cao'), (1.0, 'Cao')])
def test_risk_band_edges(p, band):
    assert risk_band(p) == band


def test_report_customers():
    rows = TEST_RAW.set_index('ID').loc[[19965, 20558]]
    p = proba(rows)
    assert p[0] == pytest.approx(0.847, abs=5e-4)
    assert p[1] == pytest.approx(0.062, abs=5e-4)


def test_valid_input_passes():
    assert validate_raw(TEST_RAW[RAW_COLUMNS]) == []


@pytest.mark.parametrize('col, value', [('AGE', 15), ('LIMIT_BAL', 5_000), ('PAY_0', 9), ('PAY_AMT1', -1),
                                        ('SEX', 3), ('MARRIAGE', 4), ('AGE', np.nan), ('BILL_AMT1', 'abc'),
                                        ('AGE', 30.5)])
def test_invalid_input_rejected(col, value):
    df = TEST_RAW[RAW_COLUMNS].head(1).astype(object)
    df.loc[df.index[0], col] = value
    assert validate_raw(df)


def test_missing_column_rejected():
    assert validate_raw(TEST_RAW[RAW_COLUMNS].drop(columns='PAY_0'))
