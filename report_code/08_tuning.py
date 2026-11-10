# Tinh chỉnh siêu tham số Random Forest bằng RandomizedSearchCV (chỉ trên train)
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_space = {
    'clf__n_estimators': [200, 300, 500],
    'clf__max_depth': list(range(5, 26)) + [None],
    'clf__min_samples_leaf': randint(1, 50),
    'clf__max_features': ['sqrt', 'log2', 0.3, 0.5],
    'clf__class_weight': [None, 'balanced'],
}
scoring = {'ap': 'average_precision', 'roc_auc': 'roc_auc', 'f1': 'f1',
           'recall': 'recall', 'precision': 'precision'}

search = RandomizedSearchCV(
    make_rf(None), param_space, n_iter=30, scoring=scoring,
    refit='ap',                 # chọn cấu hình theo PR-AUC
    cv=skf, n_jobs=-1, random_state=42)
search.fit(X_train4, y_train)

print(search.best_params_)      # max_depth=6, max_features=0.5, min_samples_leaf=21, n_estimators=200, balanced
print(f"PR-AUC CV tốt nhất: {search.best_score_:.4f}")   # 0,5645
