# Pipeline Decision Tree với siêu tham số tốt nhất (mục 4.2), cấu hình B
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier

tree_cols = list(X_train.columns)   # 41 đặc trưng của Chương 3 (không dùng BILL_TREND)

dt_pipe = Pipeline([
    ('prep', ColumnTransformer([('keep', 'passthrough', tree_cols)],
                               remainder='drop', verbose_feature_names_out=False)),
    ('clf', DecisionTreeClassifier(
        criterion='entropy',
        max_depth=14,
        min_samples_leaf=152,       # lá lớn => chống overfit
        ccp_alpha=7.73e-06,         # tỉa cây theo độ phức tạp
        class_weight='balanced',    # cấu hình B
        random_state=42)),
])
dt_pipe.fit(X_train4, y_train)
