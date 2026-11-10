# Chia tập train/test 80/20, phân tầng theo biến mục tiêu
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
X = data.drop(columns='Default')
y = data['Default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# Đặc trưng xu hướng sao kê (tính theo từng dòng nên không gây rò rỉ dữ liệu)
def add_bill_trend(df):
    return df.assign(BILL_TREND=(df['BILL_AMT1'] - df['BILL_AMT6']) / (df['BILL_AMT6'].abs() + 1))

X_train4, X_test4 = add_bill_trend(X_train), add_bill_trend(X_test)

print(f"Train: {len(y_train)} mẫu, {y_train.sum()} vỡ nợ ({y_train.mean():.2%})")   # 23.972 / 5.304
print(f"Test : {len(y_test)} mẫu, {y_test.sum()} vỡ nợ ({y_test.mean():.2%})")      # 5.993 / 1.326
