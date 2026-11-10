# Tạo 6 đặc trưng tổng hợp và 12 cờ nhị phân cho trạng thái trả nợ PAY_x
bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

data['AVG_BILL_AMT'] = data[bill_cols].mean(axis=1)
data['AVG_PAY_AMT'] = data[pay_amt_cols].mean(axis=1)
# Mẫu số chặn dưới ở 1 để tránh chia cho 0 / tỷ lệ âm khi sao kê <= 0
data['PAY_RATIO'] = data['AVG_PAY_AMT'] / data['AVG_BILL_AMT'].clip(lower=1)
data['NUM_MONTHS_DELAYED'] = (data[pay_status_cols] > 0).sum(axis=1)
data['MAX_DELAY'] = data[pay_status_cols].max(axis=1)
data['CREDIT_UTIL'] = data['AVG_BILL_AMT'] / (data['LIMIT_BAL'] + 1)

# Cờ trạng thái: -2 = không phát sinh chi tiêu, 0 = tín dụng quay vòng (giữ nguyên cột gốc)
flag_cols = []
for c in pay_status_cols:
    data[f'{c}_nouse'] = (data[c] == -2).astype(int)
    data[f'{c}_revolving'] = (data[c] == 0).astype(int)
    flag_cols += [f'{c}_nouse', f'{c}_revolving']

print(f"Số đặc trưng sau Feature Engineering: {data.shape[1] - 1}")   # 41
