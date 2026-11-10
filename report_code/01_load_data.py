# Tải bộ dữ liệu Default of Credit Card Clients (UCI, id=350) và đổi tên cột
import pandas as pd
from ucimlrepo import fetch_ucirepo

uci = fetch_ucirepo(id=350)

# ucimlrepo trả về tên cột X1..X23 và Y; tên gốc nằm ở cột 'description' của bảng biến
col_names = dict(zip(uci.variables['name'], uci.variables['description']))
col_names['Y'] = 'Default'
data = pd.concat([uci.data.features, uci.data.targets], axis=1).rename(columns=col_names)

print(f"Kích thước dữ liệu: {data.shape}")   # (30000, 24)
print(data.columns.tolist())
