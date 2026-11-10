# Đánh giá mô hình cuối trên tập test (chỉ dùng một lần)
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, brier_score_loss, accuracy_score)

# Mô hình cuối: RF cấu hình A + ngưỡng chi phí 0,17 (ngưỡng nằm trong mô hình)
final_model = FixedThresholdClassifier(rf_A, threshold=0.17).fit(X_train4, y_train)

proba = final_model.predict_proba(X_test4)[:, 1]
y_pred = final_model.predict(X_test4)

print(classification_report(y_test, y_pred, digits=4))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
auc = roc_auc_score(y_test, proba)
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Accuracy = {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC = {auc:.4f}, Gini = {2 * auc - 1:.4f}")
print(f"PR-AUC = {average_precision_score(y_test, proba):.4f}")
print(f"Brier = {brier_score_loss(y_test, proba):.4f}")
print(f"P(default) trung bình = {proba.mean():.4f} (thực tế {y_test.mean():.4f})")
