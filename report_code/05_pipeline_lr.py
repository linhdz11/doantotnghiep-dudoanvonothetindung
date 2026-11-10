# Pipeline Logistic Regression: one-hot + log1p(PAY_RATIO) + chuẩn hoá, class_weight='balanced'
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression

lr_cat_cols = ['EDUCATION', 'MARRIAGE']
# Tập đặc trưng (b): bỏ AVG_BILL_AMT, AVG_PAY_AMT, BILL_AMT2–6; thêm BILL_TREND
lr_num_cols = [c for c in X_train4.columns if c not in
               ['AVG_BILL_AMT', 'AVG_PAY_AMT', 'PAY_RATIO', 'EDUCATION', 'MARRIAGE']
               + [f'BILL_AMT{i}' for i in range(2, 7)]]

log_scale = Pipeline([('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
                      ('scale', StandardScaler())])
lr_pipe = Pipeline([
    ('prep', ColumnTransformer([
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), lr_cat_cols),
        ('log_scale', log_scale, ['PAY_RATIO']),
        ('scale', StandardScaler(), lr_num_cols),
    ], remainder='drop', verbose_feature_names_out=False)),
    # C tốt nhất từ RandomizedSearchCV (mục 4.2); cấu hình B: class_weight='balanced'
    ('clf', LogisticRegression(C=58.4833, class_weight='balanced', max_iter=5000, random_state=42)),
])
lr_pipe.fit(X_train4, y_train)
