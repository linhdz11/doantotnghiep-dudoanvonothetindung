# Chọn ngưỡng quyết định tối thiểu hoá chi phí kỳ vọng (CFN:CFP = 5:1) trên xác suất OOF của train
import numpy as np
from sklearn.model_selection import cross_val_predict

oof = cross_val_predict(rf_A, X_train4, y_train, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
thresholds = np.round(np.linspace(0.01, 0.99, 99), 2)

def expected_cost(y_true, proba, c_fn, c_fp=1):
    """Chi phí kỳ vọng trên mỗi khách hàng tại từng ngưỡng."""
    y_true = np.asarray(y_true)[:, None]
    pred = proba[:, None] >= thresholds
    fn = (~pred & (y_true == 1)).sum(axis=0)     # bỏ sót khách hàng vỡ nợ
    fp = (pred & (y_true == 0)).sum(axis=0)      # cảnh báo nhầm
    return (c_fn * fn + c_fp * fp) / len(y_true)

for c_fn in [3, 5, 10]:                          # phân tích độ nhạy
    cost = expected_cost(y_train, oof, c_fn)
    print(f"CFN:CFP = {c_fn}:1 -> ngưỡng tối ưu {thresholds[cost.argmin()]:.2f}, "
          f"chi phí/KH {cost.min():.4f} (ngưỡng 0,5: {cost[49]:.4f})")
# Kết quả với RF: 3:1 -> 0,26; 5:1 -> 0,17; 10:1 -> 0,09
