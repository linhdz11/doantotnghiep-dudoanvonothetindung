# Kết quả huấn luyện: dự đoán out-of-fold (OOF) trên train, báo cáo và ma trận nhầm lẫn
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

models = {'Logistic Regression (B)': (lr_pipe, 0.5), 'Decision Tree (B)': (dt_pipe, 0.5),
          'Random Forest (B)': (rf_B, 0.5), 'Random Forest (A + ngưỡng chi phí)': (rf_A, 0.17)}

for name, (pipe, threshold) in models.items():
    # Mỗi mẫu được dự đoán bởi mô hình không học từ chính mẫu đó
    proba = cross_val_predict(pipe, X_train4, y_train, cv=skf,
                              method='predict_proba', n_jobs=-1)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    print(f"===== {name}, ngưỡng {threshold} =====")
    print(classification_report(y_train, y_pred, digits=4))
    print(f"Accuracy = {accuracy_score(y_train, y_pred):.4f}; TN={tn}, FP={fp}, FN={fn}, TP={tp}")
