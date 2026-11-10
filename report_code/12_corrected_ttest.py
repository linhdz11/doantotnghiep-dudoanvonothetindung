# Corrected resampled t-test (Nadeau & Bengio, 2003) so sánh 2 mô hình trên cùng các fold
import numpy as np
from scipy.stats import t as t_dist, ttest_rel

def corrected_ttest(a, b, test_train_ratio=1 / 4):
    """a, b: điểm của 2 mô hình trên k fold (RepeatedStratifiedKFold 5x10 => k = 50)."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    # Hiệu chỉnh phương sai vì các tập train giữa các fold chồng lấn nhau
    t_stat = d.mean() / np.sqrt((1 / k + test_train_ratio) * d.var(ddof=1))
    p_value = 2 * t_dist.sf(abs(t_stat), df=k - 1)
    return t_stat, p_value

# Ví dụ: so sánh PR-AUC của Random Forest và Logistic Regression
t_c, p_c = corrected_ttest(fold_scores['rf']['PR-AUC'], fold_scores['lr']['PR-AUC'])
t_n, p_n = ttest_rel(fold_scores['rf']['PR-AUC'], fold_scores['lr']['PR-AUC'])
alpha = 0.05 / 3                                 # hiệu chỉnh Bonferroni cho 3 cặp
print(f"Corrected: t = {t_c:.2f}, p = {p_c:.2e}, có ý nghĩa: {p_c < alpha}")
print(f"t-test thường: t = {t_n:.2f}, p = {p_n:.2e}")
