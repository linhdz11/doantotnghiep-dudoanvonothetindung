# Pipeline Random Forest với siêu tham số tốt nhất (mục 4.2)
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

def make_rf(class_weight):
    return Pipeline([
        ('prep', ColumnTransformer([('keep', 'passthrough', tree_cols)],
                                   remainder='drop', verbose_feature_names_out=False)),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=21,
            max_features=0.5,
            class_weight=class_weight,
            random_state=42)),
    ])

rf_B = make_rf('balanced')   # cấu hình B: dùng để so sánh 3 thuật toán (ngưỡng 0,5)
rf_A = make_rf(None)         # cấu hình A: mô hình cuối (ngưỡng theo chi phí 0,17)
rf_B.fit(X_train4, y_train)
rf_A.fit(X_train4, y_train)
