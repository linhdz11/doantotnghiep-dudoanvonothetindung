# Làm sạch dữ liệu: loại dòng trùng lặp, gộp nhóm EDUCATION và MARRIAGE
import numpy as np

# 1) Loại dòng trùng lặp hoàn toàn (sau khi đã bỏ ID)
print(f"Số dòng trùng lặp: {data.duplicated().sum()}")        # 35
data = data.drop_duplicates().reset_index(drop=True)          # còn 29.965 dòng

# 2) Kiểm tra giá trị thiếu
assert data.isnull().sum().sum() == 0

# 3) EDUCATION: gộp 0, 5, 6 (không xác định) vào nhóm 4 = "khác"
data['EDUCATION'] = np.where(data['EDUCATION'].isin([0, 5, 6]), 4, data['EDUCATION'])

# 4) MARRIAGE: gộp 0 vào nhóm 3 = "khác"
data['MARRIAGE'] = np.where(data['MARRIAGE'] == 0, 3, data['MARRIAGE'])

print(data['EDUCATION'].value_counts().sort_index())
print(data['MARRIAGE'].value_counts().sort_index())
