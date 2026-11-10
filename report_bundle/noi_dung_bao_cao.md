# Tư liệu cho báo cáo — Dự đoán vỡ nợ thẻ tín dụng

File này gom toàn bộ phần chữ của notebook (markdown và output dạng văn bản), các bảng kết quả Chương 4 (ch4_*, ch4b_*) và metadata mô hình cuối. Hình nằm trong thư mục figures/.


# PHẦN A. NỘI DUNG NOTEBOOK (markdown và output văn bản)

# Dự đoán khách hàng vỡ nợ thẻ tín dụng

### Đồ án tốt nghiệp: ứng dụng học máy (Logistic Regression, Decision Tree, Random Forest) cho bài toán phân loại vỡ nợ thẻ tín dụng.


Bộ dữ liệu chứa thông tin về tình trạng thanh toán, yếu tố nhân khẩu học, dữ liệu tín dụng, lịch sử thanh toán và sao kê của khách hàng sử dụng thẻ tín dụng tại Đài Loan từ tháng 4/2005 đến tháng 9/2005 (UCI Machine Learning Repository, id=350).

# Mục lục

* [Mục tiêu](#obj)
* [Nhập thư viện và tải dữ liệu](#imp)
* [Tiền xử lý dữ liệu](#prep)
* [Feature Engineering](#fe)
    * [Tạo thêm đặc trưng nghiệp vụ mới](#fe2)
* [Phân tích khám phá dữ liệu (EDA)](#eda)
    * [Ánh xạ biến mục tiêu](#map)
    * [Thống kê mô tả](#stat)
    * [Tỷ lệ vỡ nợ theo nhóm](#rate)
    * [Chuẩn hoá và trực quan hoá đặc trưng](#std)
    * [Tương quan](#corr)
* [Học máy: Các mô hình phân loại](#ml)
    * [Xử lý mất cân bằng dữ liệu](#imb)
    * [Chia tập train/test](#sp)
    * [Đa cộng tuyến (VIF) cho Logistic Regression](#vif)
    * [Huấn luyện mô hình](#train)
    * [Đánh giá mô hình trên tập test](#eval)
    * [So sánh ROC và Precision-Recall](#roc)
    * [Đánh giá chéo (Cross-Validation)](#cv)
    * [Kiểm định thống kê](#ttest)
* [Feature Importance](#fi)
* [Kết luận](#conclusion)


<a id='obj'></a>
## Mục tiêu:<br>
-       Xác định các yếu tố chính ảnh hưởng đến khả năng vỡ nợ thẻ tín dụng của khách hàng.
-       Dự đoán khả năng vỡ nợ thẻ tín dụng cho khách hàng của ngân hàng, chỉ sử dụng 3 thuật toán: Logistic Regression, Decision Tree và Random Forest.


<a id='imp'></a>
## Nhập thư viện và tải dữ liệu


```
Kích thước dữ liệu: (30000, 24)
```

```
LIMIT_BAL  SEX  EDUCATION  MARRIAGE  AGE  PAY_0  PAY_2  PAY_3  PAY_4  \
2308       30000    1          2         2   25      0      0      0      0   
22404     150000    2          1         2   26      0      0      0      0   
23397      70000    2          3         1   32      0      0      0      0   
25058     130000    1          3         2   49      0      0      0      0   
2664       50000    2          2         2   36      0      0      0      0   

       PAY_5  ...  BILL_AMT4  BILL_AMT5  BILL_AMT6  PAY_AMT1  PAY_AMT2  \
2308       0  ...      12580      13716      14828      1500      2000   
22404      0  ...     101581      77741      77264      4486      4235   
23397      0  ...      69753      70111      70212      2431      3112   
25058      0  ...      16898      11236       6944      1610      1808   
2664       0  ...      19574      20295      19439      2000      1500   

       PAY_AMT3  PAY_AMT4  PAY_AMT5  PAY_AMT6  Default  
2308       1500      1500      1500      2000        0  
22404      3161      2647      2669      2669        0  
23397      3000      2438      2500      2554        0  
25058      7014        27      7011      4408        0  
2664       1000      1800         0      1000        1  

[5 rows x 24 columns]
```

<a id='prep'></a>
## Tiền xử lý dữ liệu

Trước khi đi vào Feature Engineering, ta cần **kiểm chứng dữ liệu bằng code**: kiểm tra dòng trùng lặp (sau khi đã bỏ cột ID — nếu còn ID thì mọi dòng đều khác nhau), giá trị thiếu ở từng cột, và kiểu dữ liệu của các biến phân loại.


```
Số dòng trùng lặp hoàn toàn (23 đặc trưng + Default): 35
Số dòng trùng lặp chỉ xét 23 đặc trưng: 56
Đã loại bỏ dòng trùng lặp. Số dòng còn lại: 29965
```

Sau khi bỏ cột ID, dữ liệu có **35 dòng trùng lặp hoàn toàn** (trùng cả 23 đặc trưng lẫn nhãn Default); nếu chỉ xét 23 đặc trưng thì có 56 dòng trùng (21 dòng trùng đặc trưng nhưng khác nhãn — đây là nhiễu nhãn không thể xử lý bằng mô hình). Ta loại 35 dòng trùng lặp hoàn toàn để tránh cùng một bản ghi xuất hiện ở cả train lẫn test; dữ liệu còn **29.965 dòng**.

```
Không có giá trị thiếu ở bất kỳ cột nào.
```

Kết quả xác nhận dữ liệu không có giá trị thiếu. Bước kiểm chứng này là cần thiết trước khi mô hình hoá.


```
Kiểu dữ liệu trước khi chuyển đổi:
SEX          int64
EDUCATION    int64
MARRIAGE     int64
dtype: object

Kiểu dữ liệu sau khi chuyển đổi sang category:
SEX          category
EDUCATION    category
MARRIAGE     category
dtype: object
```

Ba biến trên đã được xác nhận đúng bản chất là biến phân loại. Để tiện cho EDA, tính tương quan và các mô hình cây, ta chuyển các cột này về lại kiểu số nguyên (giữ nguyên mã hoá 1, 2, 3,...). Decision Tree và Random Forest dùng trực tiếp mã hoá này; riêng Logistic Regression sẽ **one-hot encode** EDUCATION và MARRIAGE bên trong Pipeline (xem mục [Chia tập train/test](#sp)), vì mô hình tuyến tính không nên coi mã 1 < 2 < 3 < 4 là có thứ tự. SEX chỉ có 2 giá trị nên giữ nguyên.

```
SEX          int64
EDUCATION    int64
MARRIAGE     int64
dtype: object
```

Có những biến cần được chuyển sang dạng phân loại (đã xử lý ở bước tiền xử lý):
* __SEX:__ Giới tính
                    1 = nam
                    2 = nữ
* __EDUCATION:__ Học vấn
                     1 = sau đại học
                     2 = đại học
                     3 = trung học phổ thông
                     4 = khác
                     5 = không xác định
                     6 = không xác định
* __MARRIAGE:__ Tình trạng hôn nhân
                    1 = đã kết hôn
                    2 = độc thân
                    3 = khác
* __PAY_0,2,3,4,5,6:__ Tình trạng trả nợ vào tháng 9, 8, 7, 6, 5, 4 năm 2005 (tương ứng)
                    -2 = không phát sinh chi tiêu (không sử dụng thẻ)
                    -1 = trả đúng hạn, trả hết
                     0 = sử dụng tín dụng quay vòng (revolving credit, trả tối thiểu)
                    1 = trễ hạn 1 tháng
                    2 = trễ hạn 2 tháng
                    ...
                    8 = trễ hạn 8 tháng
                    9 = trễ hạn từ 9 tháng trở lên


<a id='fe'></a>
## Feature engineering

Dữ liệu đã được mã hoá sẵn, tuy nhiên một số biến phân loại có các nhóm trùng lặp về ý nghĩa. Ví dụ, biến 'education' có ba nhóm mang thông tin tương tự nhau:<br>
4: khác, 5: không xác định, và 6: không xác định<br>


```
<class 'pandas.DataFrame'>
RangeIndex: 29965 entries, 0 to 29964
Data columns (total 24 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   LIMIT_BAL  29965 non-null  int64
 1   SEX        29965 non-null  int64
 2   EDUCATION  29965 non-null  int64
 3   MARRIAGE   29965 non-null  int64
 4   AGE        29965 non-null  int64
 5   PAY_0      29965 non-null  int64
 6   PAY_2      29965 non-null  int64
 7   PAY_3      29965 non-null  int64
 8   PAY_4      29965 non-null  int64
 9   PAY_5      29965 non-null  int64
 10  PAY_6      29965 non-null  int64
 11  BILL_AMT1  29965 non-null  int64
 12  BILL_AMT2  29965 non-null  int64
 13  BILL_AMT3  29965 non-null  int64
 14  BILL_AMT4  29965 non-null  int64
 15  BILL_AMT5  29965 non-null  int64
 16  BILL_AMT6  29965 non-null  int64
 17  PAY_AMT1   29965 non-null  int64
 18  PAY_AMT2   29965 non-null  int64
 19  PAY_AMT3   29965 non-null  int64
 20  PAY_AMT4   29965 non-null  int64
 21  PAY_AMT5   29965 non-null  int64
 22  PAY_AMT6   29965 non-null  int64
 23  Default    29965 non-null  int64
dtypes: int64(24)
memory usage: 5.5 MB
```

```
array([2, 1, 3, 5, 4, 6, 0])
```

Các nhóm 4: khác, 5: không xác định, và 6: không xác định có thể gộp chung vào một nhóm '4'.


Sau khi gộp nhóm, biến education có các nhóm sau:


```
array([2, 1, 3, 4])
```

Tương tự, cột 'marriage' nên chỉ có ba nhóm: 1 = đã kết hôn, 2 = độc thân, 3 = khác nhưng lại chứa nhóm '0' — sẽ được gộp vào nhóm '3'.


```
array([1, 2, 3, 0])
```

```
array([1, 2, 3])
```

<a id='fe2'></a>
### Tạo thêm đặc trưng nghiệp vụ mới

Ngoài việc gộp nhóm EDUCATION/MARRIAGE, ta tạo thêm các đặc trưng tổng hợp từ lịch sử sao kê (BILL_AMT1-6), lịch sử thanh toán (PAY_AMT1-6) và tình trạng trễ hạn (PAY_0, PAY_2-6) — các đặc trưng này được kỳ vọng phản ánh hành vi tín dụng của khách hàng tốt hơn so với từng cột riêng lẻ theo tháng:

* **AVG_BILL_AMT**: trung bình số tiền sao kê trong 6 tháng gần nhất.
* **AVG_PAY_AMT**: trung bình số tiền đã thanh toán trong 6 tháng gần nhất.
* **PAY_RATIO**: tỷ lệ giữa số tiền đã trả và số tiền sao kê (AVG_PAY_AMT / max(AVG_BILL_AMT, 1); mẫu số chặn dưới ở 1 để tránh chia cho 0 và tỷ lệ âm khi sao kê ≤ 0) — tỷ lệ càng thấp, khách hàng càng có nguy cơ vỡ nợ.
* **NUM_MONTHS_DELAYED**: số tháng (trong 6 tháng) mà khách hàng trễ hạn thanh toán (PAY_x > 0).
* **MAX_DELAY**: số tháng trễ hạn lớn nhất trong 6 tháng.
* **CREDIT_UTIL**: tỷ lệ sử dụng hạn mức tín dụng (AVG_BILL_AMT / (LIMIT_BAL + 1)).

Ngoài ra, với mỗi cột tình trạng trả nợ PAY_x (PAY_0, PAY_2–PAY_6), ta thêm **2 cờ nhị phân** và **giữ nguyên cột gốc**:

* **PAY_x_nouse** = 1 nếu PAY_x = −2 (tháng đó không phát sinh chi tiêu).
* **PAY_x_revolving** = 1 nếu PAY_x = 0 (tháng đó sử dụng tín dụng quay vòng).

Lý do: các giá trị −2, −1, 0 là **trạng thái** chứ không phải "số tháng trễ hạn", nên việc coi PAY_x là biến số liên tục (−2 < −1 < 0 < 1 < ...) là không hợp lý với mô hình tuyến tính. Cờ nhị phân cho phép Logistic Regression học riêng hiệu ứng của từng trạng thái.


```
Tỷ lệ các cờ trạng thái (trung bình = % số dòng có cờ = 1):
PAY_0_nouse        0.092
PAY_0_revolving    0.492
PAY_2_nouse        0.125
PAY_2_revolving    0.525
PAY_3_nouse        0.135
PAY_3_revolving    0.526
PAY_4_nouse        0.144
PAY_4_revolving    0.549
PAY_5_nouse        0.151
PAY_5_revolving    0.566
PAY_6_nouse        0.162
PAY_6_revolving    0.544
```

So sánh phân bố các đặc trưng mới giữa nhóm default và non-default để kiểm chứng ý nghĩa của chúng:


Có thể quan sát PAY_RATIO, NUM_MONTHS_DELAYED và MAX_DELAY có xu hướng khác biệt rõ giữa hai nhóm — khách hàng vỡ nợ thường có tỷ lệ trả nợ thấp hơn và số tháng trễ hạn nhiều hơn, phù hợp với kỳ vọng nghiệp vụ và biện minh cho việc bổ sung các đặc trưng này.


Tách biến mục tiêu (target) và các đặc trưng (features), thực hiện **sau khi** đã hoàn tất Feature Engineering để `features` phản ánh đầy đủ các cột đã gộp nhóm và tạo mới:


<a id='eda'></a>
## Phân tích khám phá dữ liệu (EDA)

<a id='map'></a>
### Ánh xạ biến mục tiêu

Sau khi loại dòng trùng lặp, trong 29.965 chủ thẻ tín dụng có 6.630 thẻ vỡ nợ; tức là tỷ lệ vỡ nợ trong dữ liệu là 22,1% (22,13%).

<a id='stat'></a>
### Thống kê mô tả
Bảng dưới đây trình bày thống kê mô tả các biến trong bộ dữ liệu.


```
THỐNG KÊ TÓM TẮT CÁC CỘT SỐ

                      count       mean        std        min       25%        50%        75%  \
LIMIT_BAL           29965.0  167442.01  129760.14   10000.00  50000.00  140000.00  240000.00   
SEX                 29965.0       1.60       0.49       1.00      1.00       2.00       2.00   
EDUCATION           29965.0       1.84       0.74       1.00      1.00       2.00       2.00   
MARRIAGE            29965.0       1.56       0.52       1.00      1.00       2.00       2.00   
AGE                 29965.0      35.49       9.22      21.00     28.00      34.00      41.00   
PAY_0               29965.0      -0.02       1.12      -2.00     -1.00       0.00       0.00   
PAY_2               29965.0      -0.13       1.20      -2.00     -1.00       0.00       0.00   
PAY_3               29965.0      -0.16       1.20      -2.00     -1.00       0.00       0.00   
PAY_4               29965.0      -0.22       1.17      -2.00     -1.00       0.00       0.00   
PAY_5               29965.0      -0.26       1.13      -2.00     -1.00       0.00       0.00   
PAY_6               29965.0      -0.29       1.15      -2.00     -1.00       0.00       0.00   
BILL_AMT1           29965.0   51283.01   73658.13 -165580.00   3595.00   22438.00   67260.00   
BILL_AMT2           29965.0   49236.37   71195.57  -69777.00   3010.00   21295.00   64109.00   
BILL_AMT3           29965.0   47067.92   69371.35 -157264.00   2711.00   20135.00   60201.00   
BILL_AMT4           29965.0   43313.33   64353.51 -170000.00   2360.00   19081.00   54601.00   
BILL_AMT5           29965.0   40358.33   60817.13  -81334.00   1787.00   18130.00   50247.00   
BILL_AMT6           29965.0   38917.01   59574.15 -339603.00   1262.00   17124.00   49252.00   
PAY_AMT1            29965.0    5670.10   16571.85       0.00   1000.00    2102.00    5008.00   
PAY_AMT2            29965.0    5927.98   23053.46       0.00    850.00    2010.00    5000.00   
PAY_AMT3            29965.0    5231.69   17616.36       0.00    390.00    1804.00    4512.00   
PAY_AMT4            29965.0    4831.62   15674.46       0.00    300.00    1500.00    4016.00   
PAY_AMT5            29965.0    4804.90   15286.37       0.00    261.00    1500.00    4042.00   
PAY_AMT6            29965.0    5221.50   17786.98       0.00    131.00    1500.00    4000.00   
Default             29965.0       0.22       0.42       0.00      0.00       0.00       0.00   
AVG_BILL_AMT        29965.0   45029.33   63279.07  -56043.17   4841.00   21110.83   57166.17   
AVG_PAY_AMT         29965.0    5281.30   10142.31       0.00   1116.67    2400.00    5600.00   
PAY_RATIO           29965.0      20.68     688.16       0.00      0.04       0.09       0.61   
NUM_MONTHS_DELAYED  29965.0       0.83       1.55       0.00      0.00       0.00       1.00   
MAX_DELAY           29965.0       0.44       1.34      -2.00      0.00       0.00       2.00   
CREDIT_UTIL         29965.0       0.37       0.35      -0.23      0.03       0.29       0.69   
PAY_0_nouse         29965.0       0.09       0.29       0.00      0.00       0.00       0.00   
PAY_0_revolving     29965.0       0.49       0.50       0.00      0.00       0.00       1.00   
PAY_2_nouse         29965.0       0.13       0.33       0.00      0.00       0.00       0.00   
PAY_2_revolving     29965.0       0.52       0.50       0.00      0.00       1.00       1.00   
PAY_3_nouse         29965.0       0.14       0.34       0.00      0.00       0.00       0.00   
PAY_3_revolving     29965.0       0.53       0.50       0.00      0.00       1.00       1.00   
PAY_4_nouse         29965.0       0.14       0.35       0.00      0.00       0.00       0.00   
PAY_4_revolving     29965.0       0.55       0.50       0.00      0.00       1.00       1.00   
PAY_5_nouse         29965.0       0.15       0.36       0.00      0.00       0.00       0.00   
PAY_5_revolving     29965.0       0.57       0.50       0.00      0.00       1.00       1.00   
PAY_6_nouse         29965.0       0.16       0.37       0.00      0.00       0.00       0.00   
PAY_6_revolving     29965.0       0.54       0.50       0.00      0.00       1.00       1.00   

                         max  
LIMIT_BAL           1.00e+06  
SEX                 2.00e+00  
EDUCATION           4.00e+00  
MARRIAGE            3.00e+00  
AGE                 7.90e+01  
PAY_0               8.00e+00  
PAY_2               8.00e+00  
PAY_3               8.00e+00  
PAY_4               8.00e+00  
PAY_5               8.00e+00  
PAY_6               8.00e+00  
BILL_AMT1           9.65e+05  
BILL_AMT2           9.84e+05  
BILL_AMT3           1.66e+06  
BILL_AMT4           8.92e+05  
BILL_AMT5           9.27e+05  
BILL_AMT6           9.62e+05  
PAY_AMT1            8.74e+05  
PAY_AMT2            1.68e+06  
PAY_AMT3            8.96e+05  
PAY_AMT4            6.21e+05  
PAY_AMT5            4.27e+05  
PAY_AMT6            5.29e+05  
Default             1.00e+00  
AVG_BILL_AMT        8.77e+05  
AVG_PAY_AMT         6.27e+05  
PAY_RATIO           5.80e+04  
NUM_MONTHS_DELAYED  6.00e+00  
MAX_DELAY           8.00e+00  
CREDIT_UTIL         5.36e+00  
PAY_0_nouse         1.00e+00  
PAY_0_revolving     1.00e+00  
PAY_2_nouse         1.00e+00  
PAY_2_revolving     1.00e+00  
PAY_3_nouse         1.00e+00  
PAY_3_revolving     1.00e+00  
PAY_4_nouse         1.00e+00  
PAY_4_revolving     1.00e+00  
PAY_5_nouse         1.00e+00  
PAY_5_revolving     1.00e+00  
PAY_6_nouse         1.00e+00  
PAY_6_revolving     1.00e+00
```

Hạn mức tín dụng (LIMIT_BAL) có giá trị trung bình 167.442 Đài tệ, độ lệch chuẩn 129.760 Đài tệ, trung vị 140.000 Đài tệ, dao động từ 10.000 đến 1.000.000 Đài tệ.<br>
Trình độ học vấn chủ yếu là đại học (2: 46,8%) và sau đại học (1: 35,3%), tiếp theo là trung học phổ thông (3: 16,4%); nhóm khác (4, sau khi gộp 0/5/6) chỉ chiếm 1,6%. Về hôn nhân, 53,2% độc thân, 45,5% đã kết hôn, 1,3% khác. Nữ chiếm 60,4%. Độ tuổi trung bình 35,5 tuổi, độ lệch chuẩn 9,2 tuổi (từ 21 đến 79 tuổi).<br>
BILL_AMT có giá trị âm (sao kê dư có, ví dụ BILL_AMT6 thấp nhất −339.603), còn PAY_RATIO lệch phải rất mạnh (trung bình 20,68, trung vị 0,09, độ lệch chuẩn 688) do một số khách hàng có sao kê rất nhỏ nhưng thanh toán lớn.

### Tần suất các biến giải thích theo nhóm vỡ nợ và không vỡ nợ


Bộ dữ liệu (sau khi loại trùng lặp) có 29.965 khách hàng sử dụng thẻ tín dụng.

Biểu đồ tần suất cho thấy nhóm vỡ nợ tập trung ở các mức hạn mức thấp; tỷ lệ vỡ nợ theo nhóm hạn mức được định lượng cụ thể ở mục [Tỷ lệ vỡ nợ theo nhóm](#rate).

Vì giá trị 0 của Default nghĩa là 'không vỡ nợ' và giá trị 1 nghĩa là 'vỡ nợ', trung bình 0,221 cho thấy có 22,1% hợp đồng thẻ tín dụng vỡ nợ vào tháng tiếp theo.

Có thể thấy PAY_0 (tình trạng trả nợ tháng 9) và PAY_2 (tình trạng trả nợ tháng 8) có khả năng phân biệt tốt hơn so với tình trạng trả nợ các tháng khác.


<a id='rate'></a>
### Tỷ lệ vỡ nợ theo nhóm

Bảng tỷ lệ vỡ nợ theo từng mức PAY_0, theo nhóm hạn mức LIMIT_BAL (5 phân vị), theo nhóm CREDIT_UTIL (5 phân vị) và theo số tháng trễ hạn NUM_MONTHS_DELAYED. Đây là thống kê mô tả trên toàn bộ dữ liệu (không có bước nào học tham số cho mô hình).

```
===== Tỷ lệ vỡ nợ Theo PAY_0 =====
       so_KH  so_vo_no  ty_le_vo_no_%
PAY_0                                
-2      2750       364          13.24
-1      5682       954          16.79
 0     14737      1888          12.81
 1      3667      1248          34.03
 2      2666      1843          69.13
 3       322       244          75.78
 4        76        52          68.42
 5        26        13          50.00
 6        11         6          54.55
 7         9         7          77.78
 8        19        11          57.89

===== Tỷ lệ vỡ nợ Theo LIMIT_BAL (5 phân vị) =====
                       so_KH  so_vo_no  ty_le_vo_no_%
LIMIT_BAL (phân vị)                                  
(9999.999, 50000.0]     7673      2439          31.79
(50000.0, 100000.0]     4817      1244          25.83
(100000.0, 180000.0]    6114      1215          19.87
(180000.0, 270000.0]    5412       913          16.87
(270000.0, 1000000.0]   5949       819          13.77

===== Tỷ lệ vỡ nợ Theo CREDIT_UTIL (5 phân vị) =====
                       so_KH  so_vo_no  ty_le_vo_no_%
CREDIT_UTIL (phân vị)                                
(-0.234, 0.0178]        5993      1359          22.68
(0.0178, 0.135]         5993       853          14.23
(0.135, 0.453]          5993      1059          17.67
(0.453, 0.758]          5993      1489          24.85
(0.758, 5.364]          5993      1870          31.20

===== Tỷ lệ vỡ nợ Theo NUM_MONTHS_DELAYED =====
                    so_KH  so_vo_no  ty_le_vo_no_%
NUM_MONTHS_DELAYED                                
0                   19918      2333          11.71
1                    4405      1316          29.88
2                    1899       736          38.76
3                    1154       587          50.87
4                     951       545          57.31
5                     298       171          57.38
6                    1340       942          70.30
```

**Nhận xét:**

* **PAY_0:** các trạng thái không trễ hạn có tỷ lệ vỡ nợ thấp: −2 (không chi tiêu) 13,2%, −1 (trả hết) 16,8%, 0 (tín dụng quay vòng) 12,8%. Tỷ lệ tăng vọt khi bắt đầu trễ hạn: trễ 1 tháng 34,0%, trễ 2 tháng 69,1%, trễ 3 tháng 75,8%. Các mức ≥ 4 có tỷ lệ 50–78% nhưng rất ít mẫu (tổng 141 khách hàng) nên không ổn định. Quan hệ **không đơn điệu** ở vùng −2/−1/0 (−1 cao hơn 0) củng cố việc tạo cờ nhị phân thay vì coi PAY_x là biến liên tục.
* **LIMIT_BAL:** tỷ lệ vỡ nợ giảm đơn điệu theo hạn mức: 31,8% ở nhóm thấp nhất (≤ 50.000) → 25,8% → 19,9% → 16,9% → 13,8% ở nhóm cao nhất (> 270.000).
* **CREDIT_UTIL:** quan hệ hình chữ U: nhóm sử dụng gần như 0 (≤ 1,8% hạn mức, gồm cả khách hàng dư có) có tỷ lệ 22,7%, thấp nhất là nhóm 1,8–13,5% (14,2%), sau đó tăng dần lên 31,2% ở nhóm sử dụng > 75,8% hạn mức.
* **NUM_MONTHS_DELAYED:** tỷ lệ vỡ nợ tăng gần như đơn điệu từ 11,7% (không trễ tháng nào, 19.918 khách hàng) lên 29,9% (1 tháng), 38,8% (2), 50,9% (3), 57,3% (4–5) và 70,3% (trễ cả 6 tháng).

<a id='std'></a>
### Chuẩn hoá và trực quan hoá dữ liệu
Biểu đồ boxplot dưới đây cho thấy các đặc trưng có thang đo và đơn vị khác nhau. Nhiều mô hình sử dụng một dạng khoảng cách nào đó khi dự đoán, vì vậy nên chuẩn hoá dữ liệu.


Các đặc trưng (trừ các cờ nhị phân PAY_x_nouse/PAY_x_revolving) được chuẩn hoá (standardization) để trực quan hoá; nghĩa là mỗi đặc trưng có giá trị trung bình bằng 0 và phương sai bằng 1. **Phép chuẩn hoá này chỉ phục vụ vẽ biểu đồ**, không dùng cho mô hình — mô hình dùng `StandardScaler` nằm trong Pipeline, chỉ fit trên tập train.

<a id='corr'></a>
### Tương quan
Ma trận tương quan giữa các biến được thể hiện qua biểu đồ heatmap bên dưới. Biến duy nhất có tương quan dương đáng kể với biến phụ thuộc 'Default' là tình trạng trả nợ tháng gần nhất (tháng 9). Tương quan âm cao nhất với default xảy ra ở Limit_Balance, cho thấy khách hàng có hạn mức tín dụng thấp hơn có xu hướng vỡ nợ nhiều hơn. Cũng có thể quan sát một số biến có tương quan cao với nhau, chẳng hạn giữa số tiền sao kê và tình trạng trả nợ ở các tháng khác nhau.

Xem xét ma trận tương quan, được xác định qua hệ số Pearson.


Heatmap cho thấy các đặc trưng có tương quan với nhau (đa cộng tuyến), ví dụ như PAY_0,2,3,4,5,6 và BILL_AMT1,2,3,4,5,6. Trong các trường hợp đó, tương quan là dương.

Dữ liệu không tương quan tiềm năng hữu ích hơn: có khả năng phân biệt tốt hơn!

**Tương quan có ý nghĩa gì?**<br>

Hãy thử fit riêng dữ liệu tương quan và không tương quan bằng hồi quy tuyến tính:


```
Dữ liệu không tương quan tiềm năng hữu ích hơn: có khả năng phân biệt tốt hơn!
```

<a id='ml'></a>
## Học máy: Các mô hình phân loại

Ba mô hình phân loại được sử dụng trong đồ án này: Logistic Regression, Decision Tree và Random Forest Classifier.<br>

Dữ liệu được chia thành đặc trưng (X) và mục tiêu (y), sau đó chia thành tập train (80%) và test (20%) — dùng **chung một cách chia** cho cả 3 mô hình (stratify theo y, random_state=42). **Mọi bước có học tham số từ dữ liệu** (StandardScaler, OneHotEncoder, SMOTE) đều nằm trong `Pipeline` và chỉ được fit trên tập train (hoặc trên phần train của từng fold khi cross-validation). Tập test chỉ được dùng một lần ở bước đánh giá cuối.

Hai tập đặc trưng được sử dụng:

* **Mô hình cây (Decision Tree, Random Forest):** toàn bộ đặc trưng, EDUCATION/MARRIAGE giữ mã hoá số, không chuẩn hoá (mô hình cây không nhạy với thang đo).
* **Logistic Regression:** bỏ AVG_BILL_AMT và AVG_PAY_AMT (là tổ hợp tuyến tính của BILL_AMT1–6 và PAY_AMT1–6, gây đa cộng tuyến hoàn hảo — xem mục [VIF](#vif)); one-hot EDUCATION và MARRIAGE (bỏ nhóm đầu làm nhóm tham chiếu); chuẩn hoá các cột còn lại bằng StandardScaler.

<a id='imb'></a>
### Xử lý mất cân bằng dữ liệu

Như đã chỉ ra ở phần EDA, khách hàng vỡ nợ chỉ chiếm **22,1%** dữ liệu (6.630/29.965) — đây là bài toán phân loại mất cân bằng (imbalanced classification). Nếu không xử lý, mô hình có xu hướng thiên về dự đoán lớp đa số (non-default), dẫn đến recall thấp cho lớp thiểu số (default) — trong khi đây lại là lớp quan trọng nhất đối với ngân hàng.

Mỗi mô hình được thử nghiệm với 3 phương án:

1. **`none`**: không xử lý mất cân bằng.
2. **`balanced`**: `class_weight='balanced'` — tự động tăng trọng số cho lớp thiểu số khi tính hàm mất mát.
3. **`smote`**: SMOTE (Synthetic Minority Over-sampling Technique, thư viện `imbalanced-learn`) đặt **bên trong Pipeline**, nên chỉ sinh mẫu tổng hợp trên phần dữ liệu đang được fit (train/fold train), không bao giờ chạm vào tập test hay fold validation. Vì SMOTE dựa trên khoảng cách k-láng giềng, dữ liệu được chuẩn hoá trước bước SMOTE (với mô hình cây, bước chuẩn hoá này không ảnh hưởng đến cách chia nút).

<a id='sp'></a>
### Chia tập dữ liệu: train và test

Cả 3 mô hình nhận cùng `X_train`/`X_test`; Pipeline của Logistic Regression tự chọn tập cột của mình (bỏ AVG_BILL_AMT, AVG_PAY_AMT) thông qua `ColumnTransformer`.

```
Kích thước tập train: (23972, 41), tập test: (5993, 41)
Tỷ lệ default trong tập train: 0.221, tập test: 0.221
Số đặc trưng cho mô hình cây: 41; cho Logistic Regression (trước one-hot): 39
```

<a id='vif'></a>
### Đa cộng tuyến (VIF) cho Logistic Regression

Hệ số phóng đại phương sai VIF<sub>j</sub> = 1 / (1 − R<sub>j</sub><sup>2</sup>), với R<sub>j</sub><sup>2</sup> là hệ số xác định khi hồi quy đặc trưng j theo tất cả đặc trưng còn lại. VIF<sub>j</sub> cũng chính là phần tử đường chéo thứ j của ma trận nghịch đảo của ma trận tương quan. Quy ước thường dùng: VIF > 10 là đa cộng tuyến nghiêm trọng.

VIF được tính trên **tập train**, với ma trận thiết kế giống hệt đầu vào của Logistic Regression (one-hot EDUCATION/MARRIAGE, bỏ nhóm đầu).

```
Giữ AVG_*: 44 cột, hạng ma trận tương quan = 42
```

```
Bỏ AVG_* : 42 cột, hạng ma trận tương quan = 42

Số đặc trưng có VIF > 10: 7 / 42
```

```
VIF
BILL_AMT2           28.27
BILL_AMT5           25.56
BILL_AMT3           22.47
NUM_MONTHS_DELAYED  21.54
BILL_AMT4           21.11
BILL_AMT1           15.69
BILL_AMT6           15.45
PAY_5_nouse          7.40
PAY_4_nouse          7.12
PAY_2_nouse          6.91
PAY_3_nouse          6.52
PAY_5                6.52
PAY_2_revolving      6.42
PAY_2                6.12
PAY_4                6.04
PAY_3                5.97
PAY_6                5.50
PAY_6_nouse          5.10
PAY_0_revolving      5.06
PAY_0                4.68
MAX_DELAY            4.61
PAY_3_revolving      4.43
PAY_0_nouse          4.41
PAY_4_revolving      3.94
PAY_5_revolving      3.78
CREDIT_UTIL          3.44
PAY_6_revolving      3.01
PAY_AMT2             2.41
LIMIT_BAL            2.36
PAY_AMT3             1.88
PAY_AMT1             1.82
PAY_AMT4             1.75
PAY_AMT5             1.71
EDUCATION_3          1.41
AGE                  1.39
EDUCATION_2          1.38
MARRIAGE_2           1.34
PAY_AMT6             1.20
EDUCATION_4          1.04
PAY_RATIO            1.04
SEX                  1.03
MARRIAGE_3           1.03
```

**Nhận xét VIF:**

* Nếu giữ AVG_BILL_AMT và AVG_PAY_AMT, ma trận tương quan 44 cột chỉ có hạng 42 → **suy biến** (đa cộng tuyến hoàn hảo, VIF = ∞). Sau khi bỏ hai cột này, ma trận 42 cột có hạng đầy đủ.
* Còn **7/42 đặc trưng có VIF > 10**: BILL_AMT1–6 (VIF từ 15,5 đến 28,3, do sao kê các tháng liên tiếp tương quan rất cao) và NUM_MONTHS_DELAYED (21,5, do được tính trực tiếp từ PAY_x). Các cờ PAY_x_nouse/PAY_x_revolving và PAY_x có VIF 3–7,4; các biến nhân khẩu học, PAY_AMT, PAY_RATIO đều < 2,5.
* Ở bước này ta **chưa loại thêm** đặc trưng: Logistic Regression dùng điều chuẩn L2 mặc định (C = 1) giúp ổn định hệ số khi có đa cộng tuyến, nhưng cần lưu ý hệ số của từng BILL_AMT riêng lẻ không nên diễn giải độc lập. Việc gộp/bỏ BILL_AMT là ứng viên cho bước tinh chỉnh tiếp theo.

<a id='train'></a>
### Huấn luyện mô hình

Ở bước này **chưa tinh chỉnh siêu tham số**: cả 3 mô hình dùng siêu tham số mặc định của scikit-learn (Logistic Regression: `C=1.0`, `max_iter=5000`; Decision Tree: cây không giới hạn độ sâu; Random Forest: 100 cây). Mỗi mô hình được thử với 3 phương án xử lý mất cân bằng (`none`, `balanced`, `smote`), đánh giá bằng `StratifiedKFold(n_splits=5)` **chỉ trên tập train**. Tập test hoàn toàn không tham gia vào việc lựa chọn.

#### Logistic Regression

Logistic Regression là một trong những thuật toán đơn giản nhất, ước lượng mối quan hệ giữa một biến nhị phân phụ thuộc và các biến độc lập, tính xác suất xảy ra của một sự kiện. Tham số điều chuẩn C kiểm soát sự đánh đổi giữa việc tăng độ phức tạp (overfitting) và giữ mô hình đơn giản (underfitting).


#### Decision Tree

Decision Tree là một thuật toán phổ biến cho bài toán phân loại vì dễ diễn giải. Một node nội bộ đại diện cho một đặc trưng, nhánh đại diện cho một quy tắc quyết định, và mỗi node lá đại diện cho kết quả. Decision Tree không cần chuẩn hoá dữ liệu, nhưng dễ bị overfit với dữ liệu nhiễu và cho kết quả thiên lệch khi dữ liệu mất cân bằng.


#### Random Forest

Random Forest gồm nhiều Decision Tree kết hợp lại. Mô hình tạo ra các tập con ngẫu nhiên khác nhau từ tập train làm các bộ dự đoán và chọn lời giải tốt nhất bằng cách bỏ phiếu (voting), nhờ đó tránh được overfitting.


```
Kết quả 5-fold CV trên tập train (trung bình qua 5 fold):
```

Với bài toán ngân hàng, chi phí bỏ sót một khách hàng sắp vỡ nợ (false negative) thường cao hơn chi phí cảnh báo nhầm (false positive), do đó với mỗi mô hình ta chọn phương án có **Recall trung bình 5-fold trên tập train** cao nhất, kể cả khi phải đánh đổi một phần Precision hoặc Accuracy. Phương án được chọn sau đó được fit lại trên toàn bộ tập train.

```
Logistic Regression: chọn phương án 'balanced' (Recall CV = 0.6135)
Decision Tree: chọn phương án 'smote' (Recall CV = 0.4600)
Random Forest: chọn phương án 'balanced' (Recall CV = 0.4981)
```

**Nhận xét (CV 5-fold trên train, siêu tham số mặc định):**

* **Logistic Regression:** `balanced` (Recall 0,6135) và `smote` (0,61) gần như tương đương, cả hai tăng Recall từ 0,37 lên ~0,61 so với `none`, đổi lại Precision giảm từ 0,65 xuống ~0,46; ROC-AUC không đổi (0,77) — xử lý mất cân bằng chủ yếu dịch chuyển ngưỡng quyết định chứ không làm mô hình phân biệt tốt hơn.
* **Decision Tree:** cây không giới hạn độ sâu bị overfit rõ rệt (ROC-AUC chỉ 0,61, PR-AUC 0,29 ở cả 3 phương án); `smote` cho Recall cao nhất (0,46) nên được chọn. Đây là mô hình được kỳ vọng cải thiện nhiều nhất khi tinh chỉnh `max_depth`, `min_samples_leaf`.
* **Random Forest:** `balanced` được chọn (Recall 0,50, F1 0,52), tốt hơn `smote` (0,49) một chút.

<a id='eval'></a>
### Đánh giá mô hình trên tập test

Với mỗi mô hình: in `classification_report`, vẽ confusion matrix, tính ROC-AUC, vẽ đường cong Precision-Recall và tính PR-AUC. Do dữ liệu mất cân bằng, PR-AUC phản ánh hiệu năng thực tế trên lớp thiểu số (default) tốt hơn ROC-AUC.


```
===== Logistic Regression =====
              precision    recall  f1-score   support

           0       0.87      0.80      0.84      4667
           1       0.46      0.59      0.52      1326

    accuracy                           0.76      5993
   macro avg       0.67      0.70      0.68      5993
weighted avg       0.78      0.76      0.77      5993

ROC-AUC: 0.7616   PR-AUC: 0.5275
```

```
===== Decision Tree =====
              precision    recall  f1-score   support

           0       0.84      0.76      0.80      4667
           1       0.37      0.48      0.42      1326

    accuracy                           0.70      5993
   macro avg       0.60      0.62      0.61      5993
weighted avg       0.73      0.70      0.71      5993
```

```
ROC-AUC: 0.6221   PR-AUC: 0.2910
```

```
===== Random Forest =====
              precision    recall  f1-score   support

           0       0.86      0.89      0.87      4667
           1       0.54      0.48      0.51      1326

    accuracy                           0.80      5993
   macro avg       0.70      0.68      0.69      5993
weighted avg       0.79      0.80      0.79      5993

ROC-AUC: 0.7607   PR-AUC: 0.5279
```

**Kết quả trên tập test (5.993 khách hàng, 1.326 vỡ nợ):**

| Mô hình (phương án) | Precision | Recall | F1 | ROC-AUC | PR-AUC | Accuracy |
|---|---|---|---|---|---|---|
| Logistic Regression (balanced) | 0,46 | 0,59 | 0,52 | 0,7616 | 0,5275 | 0,76 |
| Decision Tree (smote) | 0,37 | 0,48 | 0,42 | 0,6221 | 0,2910 | 0,70 |
| Random Forest (balanced) | 0,54 | 0,48 | 0,51 | 0,7607 | 0,5279 | 0,80 |

Kết quả trên test khớp với CV trên train (chênh lệch < 0,02), cho thấy không có dấu hiệu rò rỉ dữ liệu hay lựa chọn quá khớp với tập test. Logistic Regression và Random Forest có ROC-AUC/PR-AUC gần như bằng nhau; LR có Recall cao hơn (0,59 so với 0,48), RF có Precision và Accuracy cao hơn.

<a id='roc'></a>
### So sánh ROC và Precision-Recall giữa 3 mô hình

Receiver Operating Characteristic (ROC) là đồ thị của True Positive Rate so với False Positive Rate. AUC (Area Under Curve) càng cao thể hiện mô hình phân biệt càng tốt giữa hai lớp; AUC = 0.5 nghĩa là mô hình không có khả năng phân biệt.


AUC càng cao (gần 1) thể hiện mô hình phân biệt tốt hơn giữa hai lớp. Với dữ liệu mất cân bằng, PR-AUC phản ánh hiệu năng trên lớp thiểu số (default) thực tế hơn ROC-AUC, vì ROC-AUC có thể "lạc quan giả" khi số lượng mẫu âm (non-default) áp đảo.


<a id='cv'></a>
### Đánh giá chéo (Cross-Validation) với StratifiedKFold

Kết quả `StratifiedKFold(n_splits=5)` + `cross_validate` của phương án được chọn cho mỗi mô hình (đã tính ở mục Huấn luyện, **chỉ trên tập train**): accuracy, f1, roc_auc và average_precision (PR-AUC) qua từng fold. Toàn bộ tiền xử lý (scaler, one-hot, SMOTE) nằm trong Pipeline nên chỉ được fit trên phần train của từng fold.

```
Logistic Regression: accuracy=0.7578±0.0059, f1=0.5285±0.0097, roc_auc=0.7693±0.0099, pr_auc=0.5216±0.0157
Decision Tree: accuracy=0.6982±0.0075, f1=0.4029±0.0067, roc_auc=0.6131±0.0053, pr_auc=0.2845±0.0046
Random Forest: accuracy=0.7984±0.0044, f1=0.5223±0.0112, roc_auc=0.7681±0.0077, pr_auc=0.5255±0.0056
```

Lập bảng tổng hợp cuối cùng so sánh 3 mô hình. Precision/Recall lấy từ tập test giữ riêng (mục Đánh giá mô hình ở trên); F1, ROC-AUC, PR-AUC, Accuracy lấy trung bình ± độ lệch chuẩn qua 5 fold trên tập train (`cross_validate`), để phản ánh độ ổn định của mô hình thay vì chỉ một lần chia dữ liệu.

```
Model Phương án  Precision (test)  Recall (test) F1 (5-fold mean±std)  \
0  Logistic Regression  balanced              0.46           0.59      0.5285 ± 0.0097   
1        Decision Tree     smote              0.37           0.48      0.4029 ± 0.0067   
2        Random Forest  balanced              0.54           0.48      0.5223 ± 0.0112   

  ROC-AUC (5-fold mean±std) PR-AUC (5-fold mean±std) Accuracy (5-fold mean±std)  
0           0.7693 ± 0.0099          0.5216 ± 0.0157            0.7578 ± 0.0059  
1           0.6131 ± 0.0053          0.2845 ± 0.0046            0.6982 ± 0.0075  
2           0.7681 ± 0.0077          0.5255 ± 0.0056            0.7984 ± 0.0044
```

<a id='ttest'></a>
### Kiểm định thống kê: Paired t-test

Chọn 2 mô hình có F1-score (5-fold) trung bình cao nhất, sau đó dùng `scipy.stats.ttest_rel` (paired t-test) trên kết quả F1 của từng fold (5-fold CV trên tập train) để kiểm định xem chênh lệch có ý nghĩa thống kê hay không, với mức ý nghĩa α = 0.05.


```
So sánh Logistic Regression (F1=0.5285) vs. Random Forest (F1=0.5223)
t-statistic = 1.1915, p-value = 0.2993
=> p-value >= 0.05: chưa đủ cơ sở kết luận chênh lệch F1-score giữa Logistic Regression và Random Forest có ý nghĩa thống kê.
```

Hai mô hình có F1 trung bình cao nhất là Logistic Regression (0,5285) và Random Forest (0,5223); paired t-test trên 5 fold cho t = 1,19, **p = 0,30 > 0,05** → chưa đủ cơ sở kết luận chênh lệch F1 giữa hai mô hình có ý nghĩa thống kê. Lưu ý: với 5 fold, kiểm định có độ mạnh thấp, và các fold dùng chung dữ liệu train nên giả định độc lập của t-test chỉ gần đúng.

<a id='fi'></a>
## Feature Importance

So sánh mức độ quan trọng của các đặc trưng theo Random Forest (`feature_importances_`) và theo hệ số hồi quy (coefficients) của Logistic Regression, đặc biệt chú ý các đặc trưng mới tạo ở bước Feature Engineering (AVG_BILL_AMT, AVG_PAY_AMT, PAY_RATIO, NUM_MONTHS_DELAYED, MAX_DELAY, CREDIT_UTIL và các cờ PAY_x_nouse/PAY_x_revolving). Lưu ý: hệ số của Logistic Regression tính trên dữ liệu đã chuẩn hoá nên so sánh được độ lớn giữa các đặc trưng số; AVG_BILL_AMT, AVG_PAY_AMT không có trong mô hình LR.


```
Đặc trưng chung trong top 10 của cả 2 mô hình: ['LIMIT_BAL', 'MAX_DELAY', 'NUM_MONTHS_DELAYED', 'PAY_0']
Đặc trưng mới (Feature Engineering) xuất hiện trong top 10 của Random Forest: ['AVG_BILL_AMT', 'AVG_PAY_AMT', 'CREDIT_UTIL', 'MAX_DELAY', 'NUM_MONTHS_DELAYED', 'PAY_RATIO']
Đặc trưng mới (Feature Engineering) xuất hiện trong top 10 của Logistic Regression: ['MAX_DELAY', 'NUM_MONTHS_DELAYED', 'PAY_0_revolving', 'PAY_2_nouse', 'PAY_2_revolving']
```

Mức độ đồng thuận giữa hai mô hình: **LIMIT_BAL, PAY_0, MAX_DELAY và NUM_MONTHS_DELAYED** cùng nằm trong top 10 của cả Random Forest và Logistic Regression — hai trong số đó là đặc trưng mới tạo ở bước Feature Engineering. Top 10 của Random Forest chứa cả 6 đặc trưng tổng hợp (AVG_BILL_AMT, AVG_PAY_AMT, CREDIT_UTIL, MAX_DELAY, NUM_MONTHS_DELAYED, PAY_RATIO); top 10 của Logistic Regression chứa các cờ trạng thái PAY_0_revolving, PAY_2_nouse, PAY_2_revolving — minh chứng việc bổ sung cờ nhị phân có ích cho mô hình tuyến tính.

<a id='conclusion'></a>
## Kết luận

**Tổng kết (baseline, chưa tinh chỉnh siêu tham số):** Bảng so sánh ở mục [Đánh giá chéo](#cv) là căn cứ để lựa chọn mô hình. Trong bài toán dự đoán vỡ nợ thẻ tín dụng, **Recall và PR-AUC** nên được ưu tiên hơn Accuracy hoặc ROC-AUC đơn thuần, vì:

- Chi phí bỏ sót một khách hàng thực sự vỡ nợ (false negative) — không trích lập dự phòng, không có biện pháp xử lý kịp thời — thường lớn hơn nhiều so với chi phí cảnh báo nhầm một khách hàng tốt (false positive).
- Dữ liệu mất cân bằng (22,1% default) khiến Accuracy dễ gây hiểu lầm — một mô hình luôn dự đoán "không vỡ nợ" vẫn đạt Accuracy khoảng 77,9% nhưng vô dụng trong thực tế.
- PR-AUC phản ánh sát hơn ROC-AUC hiệu năng trên lớp thiểu số (default), vốn là lớp mà ngân hàng quan tâm nhất.

Ở bước baseline, **Logistic Regression (`class_weight='balanced'`)** có Recall cao nhất (0,61 CV; 0,59 test) với PR-AUC tương đương Random Forest (0,52–0,53), nên là mô hình đề xuất tạm thời. Random Forest (`balanced`) cho Precision và Accuracy cao hơn; chênh lệch F1 giữa hai mô hình không có ý nghĩa thống kê (p = 0,30). Decision Tree mặc định bị overfit và kém nhất trên mọi chỉ số.

**Hạn chế:**
- Chưa tinh chỉnh siêu tham số: Decision Tree không giới hạn độ sâu, Random Forest và Logistic Regression dùng tham số mặc định.
- Recall ở lớp default vẫn còn thấp so với kỳ vọng thực tế (≤ 0,61), một phần do các đặc trưng hiện có (chủ yếu là lịch sử 6 tháng gần nhất) chưa phản ánh đầy đủ hành vi tín dụng dài hạn của khách hàng.
- Còn đa cộng tuyến đáng kể trong tập đặc trưng của Logistic Regression (BILL_AMT1–6, NUM_MONTHS_DELAYED có VIF > 10).
- SMOTE thông thường sinh giá trị không nguyên cho các cờ nhị phân và mã one-hot; SMOTENC là lựa chọn phù hợp hơn cho dữ liệu hỗn hợp. Ở bước này SMOTE không cải thiện so với `class_weight='balanced'` (trừ Decision Tree).
- Ba mô hình sử dụng đều là các mô hình cơ bản; các kỹ thuật ensemble nâng cao hơn (Gradient Boosting, stacking) có thể cải thiện hiệu năng nhưng nằm ngoài phạm vi của đồ án này.

**Hướng phát triển tiếp theo:**
- Tinh chỉnh siêu tham số bằng `RandomizedSearchCV`/`GridSearchCV` với Pipeline, chấm điểm theo Recall/PR-AUC, chỉ trên tập train.
- Xử lý đa cộng tuyến của BILL_AMT (gộp, lấy xu hướng, hoặc bỏ bớt) cho Logistic Regression.
- Bổ sung thêm đặc trưng theo thời gian (xu hướng tăng/giảm số tiền sao kê qua các tháng, thay vì chỉ lấy trung bình).
- Điều chỉnh ngưỡng phân loại (classification threshold) thay vì mặc định 0,5, dựa trên đường cong Precision-Recall (chọn ngưỡng bằng CV trên train), để tối ưu Recall theo yêu cầu nghiệp vụ cụ thể của ngân hàng.


<a id='ch4'></a>
# Chương 4. Thực nghiệm: cải tiến, tinh chỉnh và đánh giá mô hình

Chương này mở rộng phần baseline ở trên nhằm trả lời ba câu hỏi nghiên cứu:

* **CH1** – Những yếu tố nào ảnh hưởng mạnh nhất đến khả năng vỡ nợ? (mục [4.6](#ch4_6))
* **CH2** – Mô hình nào dự đoán tốt nhất, và chênh lệch giữa các mô hình có ý nghĩa thống kê không? (mục [4.4](#ch4_4))
* **CH3** – Xử lý mất cân bằng lớp và ngưỡng quyết định ảnh hưởng thế nào tới khả năng phát hiện khách hàng vỡ nợ? (mục [4.3](#ch4_3))

Nội dung:

* [4.1. Cải tiến đặc trưng cho Logistic Regression](#ch4_1)
* [4.2. Tinh chỉnh siêu tham số](#ch4_2)
* [4.3. Mất cân bằng lớp và ngưỡng quyết định](#ch4_3)
* [4.4. So sánh mô hình và kiểm định thống kê](#ch4_4)
* [4.5. Đánh giá cuối trên tập test](#ch4_5)
* [4.5b. Mô hình với xác suất được hiệu chỉnh: RF cấu hình A + ngưỡng theo chi phí](#ch4_5b)
* [4.6. Diễn giải mô hình](#ch4_6) (và [4.6b](#ch4_6b) cho mô hình cuối)
* [4.7. Chọn mô hình và lưu](#ch4_7)
* [4.8. Kết quả huấn luyện (OOF) và tổng hợp Accuracy](#ch4_8)

**Nguyên tắc:** mọi lựa chọn (tập đặc trưng, siêu tham số, phương án mất cân bằng, ngưỡng, mô hình) chỉ dựa trên `X_train`/`y_train` bằng cross-validation; `X_test` chỉ được dùng ở mục 4.5 và 4.6. Mọi biến đổi có học tham số nằm trong `imblearn.Pipeline`; `random_state = 42` ở mọi nơi. Hình được lưu vào `figures/ch4_*.png`, bảng kết quả vào `results/ch4_*.csv`.

```
scikit-learn 1.9.1, imbalanced-learn 0.14.2
```

<a id='ch4_1'></a>
## 4.1. Cải tiến đặc trưng cho Logistic Regression

Hai vấn đề của tập đặc trưng LR ở Chương 3: (1) PAY_RATIO lệch phải rất mạnh (trung bình 20,68, trung vị 0,09) nên sau `StandardScaler` vài giá trị cực lớn chi phối hệ số; (2) BILL_AMT1–6 có VIF 15–28. Cải tiến:

* Thêm `FunctionTransformer(np.log1p)` cho PAY_RATIO (PAY_RATIO ≥ 0) **trước** `StandardScaler`, bên trong `ColumnTransformer` của Pipeline.
* So sánh 2 tập đặc trưng cho LR: **(a)** như Chương 3; **(b)** thay BILL_AMT1–6 bằng BILL_AMT1 và đặc trưng xu hướng **BILL_TREND = (BILL_AMT1 − BILL_AMT6) / (|BILL_AMT6| + 1)**.

BILL_TREND chỉ dùng giá trị của chính dòng đó (không học tham số từ dữ liệu), nên được tính trực tiếp trên `X_train`/`X_test` mà không gây rò rỉ. Mô hình cây giữ nguyên tập đặc trưng của Chương 3 (Pipeline của cây chọn đúng các cột cũ), nên cột BILL_TREND chỉ được LR sử dụng nếu phương án (b) được chọn. Hàm `make_model_v2(kind, variant)` mở rộng `make_model` với các thay đổi trên.

```
Số đặc trưng số của LR: (a) 37, (b) 33 (+ one-hot ['EDUCATION', 'MARRIAGE'])

Phân bố BILL_TREND trên tập train:
count    23972.0000
mean      2313.8140
std      15035.7710
min      -2628.0000
25%         -0.1400
50%          0.0990
75%          1.5330
max     421504.0000
```

VIF của phương án (b) được tính trên `X_train` với ma trận thiết kế giống đầu vào của LR (one-hot EDUCATION/MARRIAGE bỏ nhóm đầu, PAY_RATIO đã lấy log1p); để đối chiếu, tính cả VIF của phương án (a) sau khi lấy log1p PAY_RATIO.

```
Phương án (a): 42 cột, số cột VIF > 10: 7, VIF lớn nhất: 28.29 (BILL_AMT2)
Phương án (b): 38 cột, số cột VIF > 10: 1, VIF lớn nhất: 21.50 (NUM_MONTHS_DELAYED)
```

```
VIF (a)  VIF (b)
NUM_MONTHS_DELAYED  21.5600  21.5000
PAY_5_nouse          7.4000   7.3500
PAY_4_nouse          7.1200   7.0700
PAY_2_nouse          6.9100   6.8900
PAY_5                6.5200   6.4900
PAY_3_nouse          6.5200   6.4700
PAY_2_revolving      6.4400   6.4000
PAY_2                6.1300   6.0900
PAY_4                6.0400   6.0100
PAY_3                5.9700   5.9100
PAY_6                5.5000   5.4500
PAY_6_nouse          5.1000   5.1700
PAY_0_revolving      5.0600   5.0300
PAY_0                4.6800   4.6700
MAX_DELAY            4.6200   4.6200
```

Chọn tập đặc trưng bằng **PR-AUC 5-fold CV trên train** (cùng `skf` như Chương 3), dùng phương án mất cân bằng đã chọn cho LR ở Chương 3 để so sánh công bằng với baseline.

```
Tập đặc trưng được chọn cho LR: (b) — PR-AUC CV = 0.5219
```

```
PR-AUC          ROC-AUC               F1  \
LR (balanced)                                                                         
Chương 3 (không log1p)            0.5216 ± 0.0157  0.7693 ± 0.0099  0.5285 ± 0.0097   
(a) log1p PAY_RATIO               0.5219 ± 0.0154  0.7689 ± 0.0099  0.5276 ± 0.0098   
(b) log1p PAY_RATIO + BILL_TREND  0.5219 ± 0.0154  0.7685 ± 0.0102  0.5277 ± 0.0099   

                                           Recall  
LR (balanced)                                      
Chương 3 (không log1p)            0.6135 ± 0.0123  
(a) log1p PAY_RATIO               0.6124 ± 0.0120  
(b) log1p PAY_RATIO + BILL_TREND  0.6118 ± 0.0123
```

**Nhận xét 4.1:**

* **log1p(PAY_RATIO)** gần như không thay đổi hiệu năng: PR-AUC CV 0,5219 so với 0,5216 ở Chương 3 (cùng `class_weight='balanced'`), ROC-AUC 0,7689 so với 0,7693. Chênh lệch nhỏ hơn nhiều so với độ lệch chuẩn giữa các fold (≈ 0,015). Lý do là LR có điều chuẩn L2 và PAY_RATIO vốn không phải đặc trưng quan trọng của LR. Tuy vậy, phép biến đổi làm hệ số của PAY_RATIO ổn định hơn vì không còn bị vài giá trị cực lớn chi phối.
* **Phương án (b)** thay BILL_AMT2–6 bằng BILL_TREND. Số đặc trưng có VIF > 10 giảm từ **7 xuống 1**: chỉ còn NUM_MONTHS_DELAYED (21,5), vốn được tính trực tiếp từ PAY_x. VIF lớn nhất của nhóm sao kê giảm từ 28,29 (BILL_AMT2) xuống dưới 10.
* PR-AUC CV của (a) và (b) bằng nhau tới 4 chữ số thập phân (0,5219 ± 0,0154). Quy tắc chọn theo PR-AUC chọn **(b)** vì (b) cao hơn ở các chữ số sau. Kết quả này cho thấy BILL_AMT2–6 hầu như không mang thêm thông tin so với BILL_AMT1 và xu hướng sao kê. Phương án (b) đạt cùng hiệu năng với ít đặc trưng hơn (33 so với 37 cột số) và ít đa cộng tuyến hơn, nên hệ số LR dễ diễn giải hơn.
* Lưu ý: BILL_TREND rất lệch (trung vị 0,10 nhưng lớn nhất 421.504) khi BILL_AMT6 ≈ 0, nên sau chuẩn hoá phần lớn giá trị tập trung quanh 0. Đặc trưng này đóng góp ít cho LR. Lợi ích chính của (b) là giảm đa cộng tuyến chứ không phải tăng khả năng dự đoán.

<a id='ch4_2'></a>
## 4.2. Tinh chỉnh siêu tham số

`RandomizedSearchCV` cho từng mô hình với `cv = StratifiedKFold(5, shuffle=True, random_state=42)` (biến `skf`), chấm điểm đồng thời PR-AUC (`ap`), ROC-AUC, F1, Recall, Precision và chọn cấu hình theo **PR-AUC** (`refit='ap'`) — độ đo không phụ thuộc ngưỡng và phản ánh hiệu năng trên lớp thiểu số. `class_weight ∈ {None, 'balanced'}` nằm trong không gian tìm kiếm. Số tổ hợp thử: LR 50, DT 50, RF 30.

| Mô hình | Không gian tìm kiếm |
|---|---|
| LR | C ~ loguniform(10⁻⁴, 10²) |
| DT | max_depth ∈ {3, …, 20, None}; min_samples_leaf ~ randint(1, 200); ccp_alpha ~ loguniform(10⁻⁶, 10⁻²); criterion ∈ {gini, entropy} |
| RF | n_estimators ∈ {200, 300, 500}; max_depth ∈ {5, …, 25, None}; min_samples_leaf ~ randint(1, 50); max_features ∈ {sqrt, log2, 0,3, 0,5} |

```
Logistic Regression: 50 tổ hợp × 5 fold trong 2 giây, PR-AUC CV tốt nhất = 0.5258
```

```
Decision Tree: 50 tổ hợp × 5 fold trong 6 giây, PR-AUC CV tốt nhất = 0.5391
```

```
Random Forest: 30 tổ hợp × 5 fold trong 209 giây, PR-AUC CV tốt nhất = 0.5645
```

```
Siêu tham số tốt nhất           PR-AUC  \
Mô hình                                                                                   
Logistic Regression                     C=58.483337, class_weight=None  0.5258 ± 0.0155   
Decision Tree        ccp_alpha=8e-06, class_weight=None, criterion=...  0.5391 ± 0.0134   
Random Forest        class_weight=balanced, max_depth=6, max_featur...  0.5645 ± 0.0101   

                             ROC-AUC               F1           Recall        Precision  \
Mô hình                                                                                   
Logistic Regression  0.7679 ± 0.0106  0.4726 ± 0.0109  0.3705 ± 0.0103  0.6526 ± 0.0162   
Decision Tree        0.7719 ± 0.0071  0.4639 ± 0.0152  0.3582 ± 0.0180  0.6598 ± 0.0238   
Random Forest        0.7858 ± 0.0077  0.5392 ± 0.0087  0.6312 ± 0.0157  0.4708 ± 0.0097   

                    PR-AUC baseline tốt nhất (Chương 3) Chênh lệch PR-AUC  
Mô hình                                                                    
Logistic Regression                       0.5254 (none)           +0.0003  
Decision Tree                         0.2924 (balanced)           +0.2467  
Random Forest                             0.5367 (none)           +0.0278
```

```
count   mean    max
max_depth                     
3              2 0.5013 0.5017
4              2 0.5164 0.5175
5              5 0.5240 0.5288
6              1 0.5199 0.5199
7              3 0.5215 0.5356
8              3 0.5143 0.5374
9              3 0.5080 0.5270
10             3 0.5193 0.5391
11             5 0.5062 0.5382
12             2 0.4876 0.5328
13             3 0.5140 0.5234
14             5 0.5187 0.5391
15             2 0.4514 0.5118
16             1 0.4986 0.4986
17             2 0.5166 0.5366
18             2 0.4656 0.4656
19             3 0.5278 0.5375
20             1 0.5059 0.5059
None           2 0.5316 0.5331
```

**Nhận xét 4.2:** 

* **Decision Tree** cải thiện nhiều nhất: PR-AUC CV tăng từ 0,2924 (baseline tốt nhất, cây không giới hạn) lên **0,5391 ± 0,0134** (+0,2467), ROC-AUC từ 0,61 lên 0,7719. Cấu hình tốt nhất là `criterion='entropy'`, `max_depth=14`, **`min_samples_leaf=152`**, `ccp_alpha≈8·10⁻⁶`. Điều chuẩn qua kích thước lá tối thiểu đã loại bỏ hiện tượng overfit của cây mặc định.
* **Random Forest** tăng PR-AUC từ 0,5367 lên **0,5645 ± 0,0101** (+0,0278) với cây nông (`max_depth=6`, `min_samples_leaf=21`, `max_features=0,5`, 200 cây, `class_weight='balanced'`). Kết quả này cũng cho thấy rừng các cây nông, được điều chuẩn tốt hơn rừng các cây sâu mặc định.
* **Logistic Regression** gần như không đổi (0,5258 so với 0,5254). C tối ưu = 58,5, tức gần như không điều chuẩn, và bề mặt PR-AUC theo C rất phẳng. Hiệu năng của LR bị giới hạn bởi dạng tuyến tính chứ không bởi siêu tham số.
* Với refit theo PR-AUC, LR và DT chọn `class_weight=None`: PR-AUC không phụ thuộc ngưỡng nên trọng số lớp gần như không ảnh hưởng. Hệ quả là Recall ở ngưỡng 0,5 thấp (0,37 và 0,36). Việc xử lý mất cân bằng/ngưỡng được tách riêng ở mục 4.3.
* **Biểu đồ max_depth:** PR-AUC tốt nhất tăng nhanh từ 0,5017 (depth 3) lên 0,5288 (depth 5), đạt đỉnh ≈ 0,539 ở depth 8–14 (0,5391 tại depth 10 và 14), sau đó không tăng thêm. Ở cùng một độ sâu, PR-AUC dao động mạnh, ví dụ depth 12 từ 0,44 đến 0,53. Các điểm thấp nhất là những tổ hợp có lá rất nhỏ (màu tối): depth 15 với `min_samples_leaf=3` đạt 0,39, depth 12 với `min_samples_leaf=4` đạt 0,44, tức cây bị overfit. Một số tổ hợp có lá lớn nhưng PR-AUC thấp (≈ 0,48–0,49) là do `ccp_alpha` lớn (≈ 0,004–0,006), khiến cây bị tỉa quá mức. Với DT, `min_samples_leaf` và `ccp_alpha` quan trọng không kém `max_depth`: khi lá đủ lớn và tỉa vừa phải, cây sâu không còn bị overfit.

<a id='ch4_3'></a>
## 4.3. Mất cân bằng lớp và ngưỡng quyết định (trả lời CH3)

Giữ siêu tham số tốt nhất của mỗi mô hình ở 4.2 (bỏ `class_weight`), so sánh 4 cấu hình bằng 5-fold CV trên train (cùng `skf`):

* **A – none:** không xử lý mất cân bằng, ngưỡng 0,5.
* **B – balanced:** `class_weight='balanced'`.
* **C – SMOTE:** SMOTE bên trong Pipeline (chỉ sinh mẫu trên phần train của từng fold).
* **D – none + ngưỡng tối ưu F1:** `TunedThresholdClassifierCV(scoring='f1', cv=5)` bọc ngoài Pipeline của A; ngưỡng được chọn bằng CV **bên trong** phần train của mỗi fold ngoài, nên điểm của D là CV lồng (nested CV) không lạc quan.

Ngoài các độ đo thông thường, **F2** (β = 2, coi Recall quan trọng gấp đôi Precision) được dùng làm tiêu chí chọn cấu hình, vì bỏ sót khách hàng vỡ nợ tốn kém hơn cảnh báo nhầm.

```
12 tổ hợp (3 mô hình × 4 cấu hình) × 5 fold trong 84 giây
Logistic Regression: chọn cấu hình B: balanced (F2 CV = 0.5752)
Decision Tree: chọn cấu hình B: balanced (F2 CV = 0.5934)
Random Forest: chọn cấu hình B: balanced (F2 CV = 0.5908)
```

### Ngưỡng theo chi phí

Với cấu hình A, xác suất dự đoán out-of-fold trên train (`cross_val_predict(method='predict_proba')`, cùng `skf`) được dùng để tìm ngưỡng *t* tối thiểu hoá **chi phí kỳ vọng trên mỗi khách hàng**:

Chi phí(*t*) = (C<sub>FN</sub> · FN(*t*) + C<sub>FP</sub> · FP(*t*)) / n, với C<sub>FP</sub> = 1.

Kịch bản chính C<sub>FN</sub> : C<sub>FP</sub> = 5 : 1; phân tích độ nhạy với 3 : 1 và 10 : 1. Lý thuyết quyết định Bayes cho ngưỡng tối ưu t* = C<sub>FP</sub> / (C<sub>FP</sub> + C<sub>FN</sub>) nếu xác suất được hiệu chỉnh tốt (lần lượt 0,25; 0,167; 0,091) — dùng để đối chiếu.

```
Ngưỡng tối ưu  Ngưỡng Bayes 1/(1+CFN)  Chi phí/KH tại ngưỡng tối ưu  \
Mô hình             CFN:CFP                                                                        
Logistic Regression 3:1             0.2500                  0.2500                        0.4116   
                    5:1             0.1500                  0.1670                        0.5688   
                    10:1            0.1100                  0.0910                        0.7203   
Decision Tree       3:1             0.2900                  0.2500                        0.4019   
                    5:1             0.1600                  0.1670                        0.5660   
                    10:1            0.0800                  0.0910                        0.7211   
Random Forest       3:1             0.2600                  0.2500                        0.3955   
                    5:1             0.1700                  0.1670                        0.5474   
                    10:1            0.0900                  0.0910                        0.7012   

                             Chi phí/KH tại 0,5  Recall  Precision     F2  
Mô hình             CFN:CFP                                                
Logistic Regression 3:1                  0.4615  0.5777     0.4933 0.5586  
                    5:1                  0.7401  0.7376     0.3695 0.6150  
                    10:1                 1.4366  0.9112     0.2779 0.6259  
Decision Tree       3:1                  0.4631  0.5722     0.5177 0.5604  
                    5:1                  0.7432  0.7670     0.3551 0.6226  
                    10:1                 1.4434  0.9146     0.2755 0.6247  
Random Forest       3:1                  0.4652  0.5850     0.5188 0.5705  
                    5:1                  0.7510  0.7287     0.3947 0.6232  
                    10:1                 1.4656  0.9282     0.2747 0.6289
```

**Nhận xét 4.3 (CH3):** 

* **Tác động của xử lý mất cân bằng:** ở ngưỡng 0,5, không xử lý (A) cho Recall chỉ 0,35–0,37 ở cả 3 mô hình. `class_weight='balanced'` (B) nâng Recall lên 0,61 (LR), **0,66 (DT)** và 0,63 (RF). Đổi lại, Precision giảm từ 0,65–0,68 xuống 0,43–0,47. F1 tăng khoảng 0,05–0,07 và F2 tăng khoảng 0,17–0,20.
* **PR-AUC và ROC-AUC gần như không đổi giữa A, B, D** (ví dụ RF: PR-AUC 0,5649 / 0,5645 / 0,5649). Xử lý mất cân bằng không làm mô hình **phân biệt** tốt hơn mà chỉ **dịch điểm làm việc** trên cùng một đường Precision–Recall. Biểu đồ đánh đổi minh hoạ rõ điều này: các điểm A, B, C, D của mỗi mô hình nằm trên hoặc sát đường PR out-of-fold của cấu hình A.
* **SMOTE (C)** không tốt hơn `balanced` ở mô hình nào. Với DT, SMOTE còn làm giảm PR-AUC (0,5044 so với 0,5369) vì các mẫu tổng hợp nội suy trên các cờ nhị phân/biến rời rạc làm nhiễu các nút chia.
* **Ngưỡng tối ưu F1 (D)** cho F1 cao nhất ở cả 3 mô hình (LR 0,5304; DT 0,5437; RF 0,5515) với Precision và Recall cân bằng (≈ 0,51–0,57). Tuy nhiên Recall (0,55–0,57) thấp hơn B.
* **Chọn cấu hình cuối theo F2:** F2 = 5·P·R/(4P + R) coi Recall quan trọng gấp đôi Precision, phù hợp với việc bỏ sót khách hàng vỡ nợ tốn kém hơn cảnh báo nhầm. **B (balanced) có F2 cao nhất ở cả 3 mô hình**: LR 0,5752, DT 0,5934, RF 0,5908 (so với D 0,5424 / 0,5508 / 0,5627 và C 0,5736 / 0,5664 / 0,5826). Vì vậy cả 3 mô hình cuối đều dùng `class_weight='balanced'` với ngưỡng 0,5.
* **Ngưỡng theo chi phí:** với C<sub>FN</sub>:C<sub>FP</sub> = 5:1, ngưỡng tối ưu trên xác suất OOF của cấu hình A là 0,15 (LR), 0,16 (DT) và 0,17 (RF). Các giá trị này rất gần ngưỡng Bayes 1/6 ≈ 0,167. Chi phí kỳ vọng/khách hàng giảm từ ≈ 0,74–0,75 (ngưỡng 0,5) xuống 0,55–0,57, tức giảm khoảng 23–27%. Tại ngưỡng này Recall đạt 0,73–0,77 nhưng Precision chỉ 0,36–0,39.
* **Độ nhạy với tỷ lệ chi phí:** ở tỷ lệ 3:1, ngưỡng tối ưu tăng lên 0,25–0,29 (Recall ≈ 0,58). Ở 10:1, ngưỡng giảm xuống 0,08–0,11 (Recall 0,91–0,93, Precision ≈ 0,28). Ngưỡng tối ưu thay đổi mạnh theo tỷ lệ chi phí, trong khi thứ hạng các mô hình gần như giữ nguyên: RF luôn có chi phí thấp nhất. Nếu ngân hàng xác định được tỷ lệ chi phí thực tế, nên đặt ngưỡng theo chi phí thay vì dùng ngưỡng 0,5 mặc định.

**Trả lời CH3:** xử lý mất cân bằng và điều chỉnh ngưỡng không cải thiện khả năng phân biệt (PR-AUC, ROC-AUC) nhưng quyết định điểm làm việc của mô hình. `class_weight='balanced'` tăng Recall từ ≈ 0,36 lên ≈ 0,61–0,66 và là cấu hình tốt nhất theo F2. SMOTE không mang lại lợi ích so với trọng số lớp. Khi có thông tin chi phí, ngưỡng theo chi phí (≈ 0,17 với tỷ lệ 5:1) giảm chi phí kỳ vọng khoảng một phần tư so với ngưỡng 0,5.

<a id='ch4_4'></a>
## 4.4. So sánh mô hình và kiểm định thống kê (trả lời CH2)

Ba mô hình cuối (siêu tham số ở 4.2, cấu hình ở 4.3) được đánh giá bằng `RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)` trên train — **cùng 50 fold cho cả 3 mô hình** nên có thể so sánh theo cặp. Cấu hình D (nếu được chọn) vẫn giữ `TunedThresholdClassifierCV` bên trong mỗi fold. Độ đo: PR-AUC, ROC-AUC, F1, Recall, Brier score (càng nhỏ càng tốt), **Gini = 2·AUC − 1** và **KS = max|TPR − FPR|** (thống kê Kolmogorov–Smirnov, scorer tự viết).

Do 50 fold dùng chung dữ liệu (các tập train chồng lấn), t-test ghép cặp thông thường đánh giá thấp phương sai và cho p-value quá nhỏ. Ta dùng **corrected resampled t-test** (Nadeau & Bengio, 2003):

t = mean(d) / √[(1/k + n<sub>test</sub>/n<sub>train</sub>) · var(d)],  với d là chênh lệch điểm trên k = 50 fold, n<sub>test</sub>/n<sub>train</sub> = 1/4, bậc tự do k − 1 = 49.

Ba cặp (LR–DT, LR–RF, DT–RF) được kiểm định trên PR-AUC và F1, hiệu chỉnh Bonferroni α = 0,05/3 ≈ 0,0167 cho mỗi độ đo.

```
3 mô hình × 50 fold trong 57 giây
```

```
PR-AUC          ROC-AUC             Gini  \
Mô hình (mean ± std trên 50 fold)                                                      
Logistic Regression (B: balanced)  0.5236 ± 0.0150  0.7693 ± 0.0085  0.5386 ± 0.0169   
Decision Tree (B: balanced)        0.5350 ± 0.0139  0.7691 ± 0.0079  0.5381 ± 0.0159   
Random Forest (B: balanced)        0.5636 ± 0.0146  0.7859 ± 0.0073  0.5718 ± 0.0147   

                                                KS               F1           Recall  \
Mô hình (mean ± std trên 50 fold)                                                      
Logistic Regression (B: balanced)  0.4186 ± 0.0154  0.5284 ± 0.0110  0.6132 ± 0.0153   
Decision Tree (B: balanced)        0.4204 ± 0.0125  0.5164 ± 0.0087  0.6623 ± 0.0178   
Random Forest (B: balanced)        0.4390 ± 0.0129  0.5404 ± 0.0088  0.6309 ± 0.0137   

                                             Brier  
Mô hình (mean ± std trên 50 fold)                   
Logistic Regression (B: balanced)  0.1870 ± 0.0024  
Decision Tree (B: balanced)        0.1871 ± 0.0026  
Random Forest (B: balanced)        0.1783 ± 0.0023
```

**Nhận xét 4.4 (CH2):** 

* **Hiệu năng trên 50 fold:** Random Forest dẫn đầu ở mọi độ đo xếp hạng: PR-AUC **0,5636 ± 0,0146**, ROC-AUC 0,7859 (Gini 0,5718), KS 0,4390, Brier thấp nhất 0,1783. RF cũng có F1 cao nhất (0,5404). Decision Tree có PR-AUC 0,5350 và Recall cao nhất (0,6623) nhưng F1 thấp nhất (0,5164). Logistic Regression có PR-AUC 0,5236, ROC-AUC 0,7693 (xấp xỉ DT 0,7691).
* **Kiểm định PR-AUC** (corrected t-test, α Bonferroni = 0,0167): RF tốt hơn LR (chênh lệch 0,0400; t = −8,38; p ≈ 5·10⁻¹¹) và tốt hơn DT (0,0286; t = −6,43; p ≈ 5·10⁻⁸), cả hai đều **có ý nghĩa**. Chênh lệch LR–DT (−0,0114; p = 0,12) **không có ý nghĩa**.
* **Kiểm định F1:** RF tốt hơn LR (0,0120; p = 0,0012) và DT (0,0240; p ≈ 10⁻⁶), đều có ý nghĩa. LR tốt hơn DT 0,0120 với p = 0,01664, chỉ vừa dưới ngưỡng 0,01667 nên đây là kết luận **sát ranh giới**, cần thận trọng.
* **So với t-test thường:** `ttest_rel` cho |t| lớn gấp √13,5 ≈ 3,67 lần (do hệ số hiệu chỉnh (1/50 + 1/4)/(1/50) = 13,5 trên phương sai) và p-value nhỏ hơn nhiều bậc (ví dụ LR–DT trên PR-AUC: p ≈ 5·10⁻⁷ so với 0,12). Nếu dùng t-test thường sẽ kết luận sai rằng LR và DT khác nhau rõ rệt. Nguyên nhân là 50 fold dùng chung dữ liệu nên các điểm không độc lập; hiệu chỉnh Nadeau–Bengio cần thiết để tránh sai lầm loại I.
* Boxplot cho thấy phân bố PR-AUC của RF nằm cao hơn hẳn hai mô hình còn lại. Phân bố của LR và DT chồng lấn nhiều.

**Trả lời CH2:** Random Forest (đã tinh chỉnh, `class_weight='balanced'`) là mô hình tốt nhất. Ưu thế về PR-AUC (+0,03 đến +0,04) và F1 (+0,01 đến +0,02) so với LR và DT có ý nghĩa thống kê sau khi hiệu chỉnh cho các fold phụ thuộc và cho so sánh bội. Logistic Regression và Decision Tree đã tinh chỉnh không khác nhau có ý nghĩa về PR-AUC.

<a id='ch4_5'></a>
## 4.5. Đánh giá cuối trên tập test

Đây là lần **duy nhất** tập test được dùng để đánh giá. Mỗi mô hình cuối được fit trên toàn bộ `X_train` và bọc trong `FixedThresholdClassifier` với ngưỡng đã chọn ở 4.3 (0,5 cho cấu hình A/B/C; với cấu hình D, ngưỡng do `TunedThresholdClassifierCV` chọn bằng 5-fold CV trên toàn bộ train), nên đối tượng mô hình chứa luôn ngưỡng quyết định.

```
Logistic Regression: cấu hình B: balanced, ngưỡng = 0.5000
```

```
Decision Tree: cấu hình B: balanced, ngưỡng = 0.5000
```

```
Random Forest: cấu hình B: balanced, ngưỡng = 0.5000
===== Logistic Regression =====
              precision    recall  f1-score   support

           0     0.8736    0.8039    0.8373      4667
           1     0.4611    0.5905    0.5179      1326

    accuracy                         0.7567      5993
   macro avg     0.6674    0.6972    0.6776      5993
weighted avg     0.7823    0.7567    0.7666      5993

===== Decision Tree =====
              precision    recall  f1-score   support

           0     0.8749    0.7570    0.8117      4667
           1     0.4199    0.6192    0.5005      1326

    accuracy                         0.7265      5993
   macro avg     0.6474    0.6881    0.6561      5993
weighted avg     0.7743    0.7265    0.7428      5993

===== Random Forest =====
              precision    recall  f1-score   support

           0     0.8802    0.8016    0.8391      4667
           1     0.4687    0.6161    0.5324      1326

    accuracy                         0.7606      5993
   macro avg     0.6745    0.7089    0.6857      5993
weighted avg     0.7892    0.7606    0.7712      5993
```

```
Cấu hình  Ngưỡng  Accuracy  Precision  Recall     F1     F2  ROC-AUC  \
Mô hình                                                                                        
Logistic Regression  B: balanced  0.5000    0.7567     0.4611  0.5905 0.5179 0.5591   0.7624   
Decision Tree        B: balanced  0.5000    0.7265     0.4199  0.6192 0.5005 0.5655   0.7535   
Random Forest        B: balanced  0.5000    0.7606     0.4687  0.6161 0.5324 0.5797   0.7736   

                     PR-AUC   Gini     KS  Brier  
Mô hình                                           
Logistic Regression  0.5275 0.5247 0.4030 0.1876  
Decision Tree        0.5272 0.5069 0.3865 0.1916  
Random Forest        0.5559 0.5471 0.4251 0.1815
```

**Khoảng tin cậy bootstrap:** lấy mẫu có hoàn lại 1.000 lần tập test (cùng các chỉ số mẫu cho cả 3 mô hình), tính ROC-AUC và PR-AUC trên mỗi mẫu; khoảng tin cậy 95% là phân vị 2,5% – 97,5%.

```
Giá trị (test)  CI 95% dưới  CI 95% trên
Mô hình             Độ đo                                            
Logistic Regression ROC-AUC          0.7624       0.7472       0.7779
                    PR-AUC           0.5275       0.4999       0.5580
Decision Tree       ROC-AUC          0.7535       0.7387       0.7688
                    PR-AUC           0.5272       0.4997       0.5551
Random Forest       ROC-AUC          0.7736       0.7593       0.7884
                    PR-AUC           0.5559       0.5279       0.5822
```

```
Baseline (Chương 3)                               \
                    Phương án baseline           Precision Recall     F1 ROC-AUC PR-AUC   
Logistic Regression           balanced              0.4609 0.5905 0.5177  0.7616 0.5275   
Decision Tree                    smote              0.3665 0.4804 0.4158  0.6221 0.2910   
Random Forest                 balanced              0.5441 0.4789 0.5094  0.7607 0.5279   

                    Sau tinh chỉnh (Chương 4)                              Chênh lệch         \
                                    Precision Recall     F1 ROC-AUC PR-AUC  Precision Recall   
Logistic Regression                    0.4611 0.5905 0.5179  0.7624 0.5275     0.0003 0.0000   
Decision Tree                          0.4199 0.6192 0.5005  0.7535 0.5272     0.0534 0.1388   
Random Forest                          0.4687 0.6161 0.5324  0.7736 0.5559    -0.0754 0.1373   

                                           
                        F1 ROC-AUC PR-AUC  
Logistic Regression 0.0002  0.0008 0.0000  
Decision Tree       0.0847  0.1313 0.2362  
Random Forest       0.0230  0.0129 0.0280
```

**Nhận xét 4.5:** 

* **Kết quả test khớp với CV** (chênh lệch ≤ 0,016 ở PR-AUC và ROC-AUC, ví dụ RF: PR-AUC 0,5559 test so với 0,5636 CV). Không có dấu hiệu lựa chọn quá khớp. Thứ hạng giữ nguyên: **RF** có ROC-AUC 0,7736 [CI 95%: 0,7593–0,7884], PR-AUC 0,5559 [0,5279–0,5822], Gini 0,5471, KS 0,4251, Brier 0,1815. LR có PR-AUC 0,5275 [0,4999–0,5580], DT 0,5272 [0,4997–0,5551].
* Khoảng tin cậy bootstrap của PR-AUC giữa RF và LR/DT **chồng lấn một phần**. Với một tập test 5.993 dòng, chênh lệch ≈ 0,03 chưa đủ để tách bạch hai khoảng. Kết luận về ý nghĩa thống kê dựa trên kiểm định ghép cặp 50 fold ở mục 4.4, không dựa trên việc so khoảng tin cậy độc lập.
* **Ở ngưỡng đã chọn:** RF phát hiện 61,6% khách hàng vỡ nợ (Recall 0,6161) với Precision 0,4687, F1 0,5324, F2 0,5797 (cao nhất). DT có Recall cao nhất (0,6192) nhưng Precision thấp nhất (0,4199). LR có Recall 0,5905 và Precision 0,4611.
* **So với baseline Chương 3 (cùng tập test):** DT cải thiện mạnh (PR-AUC +0,2362, ROC-AUC +0,1313, Recall +0,1388). RF tăng Recall +0,1373 (0,48 → 0,62), PR-AUC +0,0280 và F1 +0,0230, đổi lại Precision giảm 0,0754 vì chuyển sang ưu tiên Recall. LR gần như không đổi (chênh lệch ≤ 0,001), phù hợp với kết quả CV ở 4.1–4.2.
* **Calibration:** cả 3 đường cong nằm **dưới đường chéo** ở vùng xác suất trung bình và cao (ví dụ dự đoán ≈ 0,75 nhưng tỷ lệ vỡ nợ thực tế chỉ ≈ 0,4). Đây là hệ quả của `class_weight='balanced'`, vốn phóng đại xác suất của lớp thiểu số. Mô hình phù hợp để **xếp hạng/phân loại**. Nếu cần xác suất vỡ nợ (PD) đúng nghĩa, ví dụ cho trích lập dự phòng, cần hiệu chỉnh thêm (`CalibratedClassifierCV`, isotonic/sigmoid) trên dữ liệu train.

<a id='ch4_5b'></a>
## 4.5b. Mô hình với xác suất được hiệu chỉnh: Random Forest cấu hình A + ngưỡng theo chi phí

Calibration curve ở mục 4.5 cho thấy `class_weight='balanced'` **phóng đại xác suất vỡ nợ**: nhóm được dự đoán ≈ 0,75 thực tế chỉ có ≈ 0,4 vỡ nợ, và mức nền SHAP bằng 0,5 thay vì tỷ lệ vỡ nợ thực tế ≈ 0,221. Ứng dụng ở Chương 5 cần hiển thị **xác suất vỡ nợ có ý nghĩa**, nên ta xét một mô hình ứng viên khác đạt mục tiêu ưu tiên Recall bằng cách hạ ngưỡng thay vì đặt trọng số lớp:

* **RF A:** Random Forest cấu hình A (siêu tham số tốt nhất ở 4.2, `class_weight=None`). Mô hình không đặt trọng số lớp nên xác suất dự đoán phản ánh đúng tỷ lệ gốc.
* **Ngưỡng chi phí:** ngưỡng tối thiểu hoá chi phí kỳ vọng với C<sub>FN</sub>:C<sub>FP</sub> = 5:1, chọn trên xác suất OOF của train (mục 4.3), đọc lại từ `cost_table`.
* **Mô hình ứng viên "RF A + ngưỡng chi phí"** = `FixedThresholdClassifier(RF A, threshold = ngưỡng chi phí)`.

Kết quả của cấu hình B (mục 4.3–4.5) được giữ nguyên để so sánh.

**Quy tắc chọn mô hình cuối (cố định TRƯỚC khi xem tập test).** So sánh trên train bằng cùng `RepeatedStratifiedKFold` 5×10 (`rskf`) như mục 4.4. Với RF A, ngưỡng được chọn **lại bên trong mỗi fold** bằng `TunedThresholdClassifierCV` (chi phí 5:1, cv = 5) để kết quả không lạc quan. Chọn **RF A + ngưỡng chi phí** nếu đồng thời thoả ba điều kiện:
1. F2 **không kém** RF B một cách có ý nghĩa: không xảy ra trường hợp F2 trung bình thấp hơn với p < 0,05 (corrected t-test);
2. chi phí kỳ vọng 5:1 **không kém** RF B một cách có ý nghĩa: không xảy ra trường hợp chi phí trung bình cao hơn với p < 0,05;
3. Brier score trung bình **thấp hơn** RF B.

Nếu không, giữ RF B.

```
Ngưỡng chi phí của RF A (CFN:CFP = 5:1, chọn trên OOF train ở mục 4.3): 0.17
```

```
2 mô hình × 50 fold trong 367 giây
```

```
Precision           Recall               F1  \
Mô hình (mean ± std trên 50 fold)                                                      
RF B (balanced, ngưỡng 0,5)        0.4728 ± 0.0096  0.6309 ± 0.0137  0.5404 ± 0.0088   
RF A + ngưỡng chi phí              0.3862 ± 0.0207  0.7412 ± 0.0317  0.5068 ± 0.0122   

                                                F2 Chi phí/KH (5:1)           PR-AUC  \
Mô hình (mean ± std trên 50 fold)                                                      
RF B (balanced, ngưỡng 0,5)        0.5913 ± 0.0106  0.5641 ± 0.0142  0.5636 ± 0.0146   
RF A + ngưỡng chi phí              0.6249 ± 0.0110  0.5490 ± 0.0137  0.5637 ± 0.0142   

                                           ROC-AUC            Brier         Log loss  
Mô hình (mean ± std trên 50 fold)                                                     
RF B (balanced, ngưỡng 0,5)        0.7859 ± 0.0073  0.1783 ± 0.0023  0.5430 ± 0.0060  
RF A + ngưỡng chi phí              0.7853 ± 0.0073  0.1334 ± 0.0022  0.4261 ± 0.0056
```

```
F2 của A kém B có ý nghĩa? False | Chi phí 5:1 của A kém B có ý nghĩa? False | Brier của A tốt hơn? True
=> QUYẾT ĐỊNH: mô hình cuối là RF A + ngưỡng chi phí
```

```
RF A + ngưỡng chi phí   RF B  Chênh lệch (A − B)  t (corrected)  p (corrected)  \
Độ đo                                                                                              
F2                               0.6249 0.5913              0.0337         5.4773         0.0000   
Chi phí/KH (5:1)                 0.5490 0.5641             -0.0151        -2.2146         0.0315   
Brier                            0.1334 0.1783             -0.0449       -51.4084         0.0000   

                  t (ttest_rel)  
Độ đo                            
F2                      20.1248  
Chi phí/KH (5:1)        -8.1369  
Brier                 -188.8866
```

```
RF A: xác suất OOF trung bình = 0.2214 (tỷ lệ vỡ nợ thực tế trên train = 0.2213)
RF B: xác suất OOF trung bình = 0.4262 (tỷ lệ vỡ nợ thực tế trên train = 0.2213)
```

**Nhận xét so sánh trên train:** 

* **Ngưỡng chi phí** đọc từ `cost_table` là **0,17**, chọn trên xác suất OOF train với C<sub>FN</sub>:C<sub>FP</sub> = 5:1, gần với ngưỡng Bayes 1/6 ≈ 0,167.
* **Khả năng phân biệt như nhau:** PR-AUC 0,5637 (A) so với 0,5636 (B), ROC-AUC 0,7853 so với 0,7859. Hai mô hình có cùng siêu tham số, chỉ khác trọng số lớp nên xếp hạng khách hàng gần như giống hệt.
* **Điểm làm việc:** RF A + ngưỡng chi phí có Recall **0,7412** (B: 0,6309) và Precision 0,3862 (B: 0,4728). F2 tăng từ 0,5913 lên **0,6249** (+0,0337; corrected t = 5,48; p < 10⁻⁵). Chi phí kỳ vọng 5:1 giảm từ 0,5641 xuống **0,5490**/khách hàng (−0,0151; t = −2,21; p = 0,032). F1 giảm (0,5068 so với 0,5404) vì F1 coi Precision và Recall ngang nhau. Recall của A dao động nhiều hơn giữa các fold (std 0,032 so với 0,014) vì ngưỡng được chọn lại trong từng fold.
* **Chất lượng xác suất tốt hơn rõ rệt:** Brier 0,1334 so với 0,1783 (−0,0449; t = −51,4; p < 10⁻⁴), log loss 0,4261 so với 0,5430. Xác suất OOF trung bình của RF A là **0,2214**, khớp tỷ lệ vỡ nợ thực tế 0,2213. RF B cho 0,4262, gần gấp đôi. Trên calibration curve OOF, RF A bám sát đường chéo, còn RF B nằm dưới đường chéo ở mọi bin.
* **Áp dụng quy tắc chọn:** F2 và chi phí 5:1 của RF A không những không kém mà còn **tốt hơn RF B có ý nghĩa** (α = 0,05), và Brier thấp hơn. Vì vậy **RF A + ngưỡng chi phí được chọn làm mô hình cuối**. Quyết định này được đưa ra hoàn toàn trên train, trước khi đánh giá test.
* Để đối chiếu: nếu dùng t-test thường, |t| lớn gấp ≈ 3,67 lần. Với F2 và Brier, kết luận không đổi. Với chi phí, p của t-test thường nhỏ hơn nhiều so với p = 0,032 của corrected t-test.

### Đánh giá RF A + ngưỡng chi phí trên tập test

Mô hình ứng viên được fit trên toàn bộ `X_train4` với ngưỡng chi phí đã chọn trên train, rồi đánh giá trên `X_test4`. RF B lấy lại từ mục 4.5 (`final_models['rf']`), nên cùng một đối tượng mô hình đã đánh giá trước đó. Khoảng tin cậy bootstrap dùng lại đúng `boot_idx` của mục 4.5.

```
===== RF B (balanced, ngưỡng 0,5) =====
              precision    recall  f1-score   support

           0     0.8802    0.8016    0.8391      4667
           1     0.4687    0.6161    0.5324      1326

    accuracy                         0.7606      5993
   macro avg     0.6745    0.7089    0.6857      5993
weighted avg     0.7892    0.7606    0.7712      5993

===== RF A + ngưỡng chi phí =====
              precision    recall  f1-score   support

           0     0.8902    0.6705    0.7648      4667
           1     0.3793    0.7089    0.4942      1326

    accuracy                         0.6790      5993
   macro avg     0.6348    0.6897    0.6295      5993
weighted avg     0.7772    0.6790    0.7050      5993

Tỷ lệ vỡ nợ thực tế trên test: 0.2213
Xác suất vỡ nợ dự đoán trung bình — RF B (balanced, ngưỡng 0,5): 0.4260
Xác suất vỡ nợ dự đoán trung bình — RF A + ngưỡng chi phí: 0.2206
```

```
Mô hình                RF B (balanced, ngưỡng 0,5)  RF A + ngưỡng chi phí
Ngưỡng                                      0.5000                 0.1700
Precision                                   0.4687                 0.3793
Recall                                      0.6161                 0.7089
F1                                          0.5324                 0.4942
F2                                          0.5797                 0.6040
Chi phí/KH (5:1)                            0.5792                 0.5787
PR-AUC                                      0.5559                 0.5586
ROC-AUC                                     0.7736                 0.7733
Gini                                        0.5471                 0.5466
KS                                          0.4251                 0.4239
Brier                                       0.1815                 0.1359
Log loss                                    0.5503                 0.4332
P(default) trung bình                       0.4260                 0.2206
```

```
Giá trị (test)  CI 95% dưới  CI 95% trên
Mô hình                     Độ đo                                            
RF B (balanced, ngưỡng 0,5) ROC-AUC          0.7736       0.7593       0.7884
                            PR-AUC           0.5559       0.5279       0.5822
                            Brier            0.1815       0.1769       0.1859
RF A + ngưỡng chi phí       ROC-AUC          0.7733       0.7586       0.7879
                            PR-AUC           0.5586       0.5299       0.5848
                            Brier            0.1359       0.1299       0.1417
```

**Nhận xét 4.5b (tập test):** 

* **Xác suất có ý nghĩa:** xác suất vỡ nợ dự đoán trung bình trên test của RF A là **0,2206**, sát tỷ lệ thực tế 0,2213. RF B cho 0,4260. Calibration curve trên test của RF A nằm gần như trên đường chéo, ví dụ bin dự đoán ≈ 0,42 có tỷ lệ thực tế ≈ 0,41. RF B nằm dưới đường chéo ở mọi bin, ví dụ bin dự đoán ≈ 0,70 chỉ có ≈ 0,40 vỡ nợ.
* **Brier:** 0,1359 [CI 95%: 0,1299–0,1417] so với 0,1815 [0,1769–0,1859]; hai khoảng tin cậy **không chồng lấn**. Log loss 0,4332 so với 0,5503.
* **Khả năng phân biệt không đổi:** PR-AUC 0,5586 [0,5299–0,5848] so với 0,5559, ROC-AUC 0,7733 so với 0,7736, Gini 0,5466 và KS 0,4239 xấp xỉ RF B.
* **Đánh đổi ở điểm làm việc:** với ngưỡng 0,17, RF A phát hiện **70,9%** khách hàng vỡ nợ (940/1.326) so với 61,6% (817) của RF B, tức thêm 123 khách hàng vỡ nợ được cảnh báo. Đổi lại có thêm ≈ 612 cảnh báo nhầm: FP tăng từ 926 lên 1.538, Precision giảm từ 0,4687 xuống 0,3793, Accuracy từ 0,7606 xuống 0,6790. F2 tăng từ 0,5797 lên 0,6040. Chi phí kỳ vọng 5:1 trên test gần như bằng nhau (0,5787 so với 0,5792): 123 × 5 ≈ 615 đơn vị tiết kiệm từ FN bù cho ≈ 612 đơn vị chi phí FP tăng thêm. Ưu thế về chi phí thấy trên CV (−0,015) không lặp lại rõ trên một tập test.
* Kết quả test nhất quán với lựa chọn đã làm trên train. RF A + ngưỡng chi phí giữ nguyên khả năng phân biệt của RF B, tăng Recall và cho xác suất vỡ nợ **dùng được trực tiếp** (hiển thị trong ứng dụng, phân nhóm rủi ro, tính tổn thất kỳ vọng).

<a id='ch4_6'></a>
## 4.6. Diễn giải mô hình (trả lời CH1)

* **Permutation importance** trên tập test (`scoring='average_precision'`, 30 lần hoán vị): mức giảm PR-AUC khi xáo trộn ngẫu nhiên một cột — đo trực tiếp mức độ mô hình dựa vào đặc trưng đó trên dữ liệu chưa thấy, áp dụng được cho cả RF và LR. Hoán vị được thực hiện trên cột đầu vào gốc (trước one-hot/log1p), nên EDUCATION hay PAY_RATIO được đánh giá như một đặc trưng.
* **MDI** (`feature_importances_` của RF): tính trên dữ liệu train, thiên vị các đặc trưng liên tục/nhiều giá trị — dùng để đối chiếu.
* **Hệ số và odds ratio** của LR trên dữ liệu đã chuẩn hoá: odds ratio = exp(hệ số) là hệ số nhân của odds vỡ nợ khi đặc trưng tăng 1 độ lệch chuẩn (với biến one-hot: so với nhóm tham chiếu).

```
Permutation importance trong 13 giây
```

```
RF: tương quan hạng Spearman giữa permutation importance và MDI = 0.686; 12/15 đặc trưng chung trong top 15
Top 5 RF (perm): ['PAY_0', 'NUM_MONTHS_DELAYED', 'MAX_DELAY', 'PAY_RATIO', 'CREDIT_UTIL']
Top 5 LR (perm): ['NUM_MONTHS_DELAYED', 'PAY_0_revolving', 'PAY_0', 'MAX_DELAY', 'PAY_2_revolving']
```

```
RF perm mean  RF perm std  RF perm hạng  RF MDI  RF MDI hạng  LR perm mean  \
PAY_0                     0.0847       0.0073        1.0000  0.1753       3.0000        0.0295   
NUM_MONTHS_DELAYED        0.0550       0.0064        2.0000  0.2836       1.0000        0.0930   
MAX_DELAY                 0.0388       0.0058        3.0000  0.2221       2.0000        0.0221   
PAY_RATIO                 0.0137       0.0042        4.0000  0.0213       9.0000        0.0009   
CREDIT_UTIL               0.0102       0.0018        5.0000  0.0247       7.0000        0.0015   
PAY_2                     0.0099       0.0028        6.0000  0.0529       4.0000        0.0060   
PAY_0_revolving           0.0068       0.0030        7.0000  0.0112      12.0000        0.0588   
PAY_6                     0.0043       0.0020        8.0000  0.0011      28.0000       -0.0004   
BILL_AMT1                 0.0040       0.0016        9.0000  0.0177      10.0000        0.0046   
AVG_BILL_AMT              0.0034       0.0009       10.0000  0.0289       6.0000        0.0000   
PAY_3                     0.0032       0.0022       11.0000  0.0079      15.0000        0.0005   
AVG_PAY_AMT               0.0029       0.0018       12.0000  0.0393       5.0000        0.0000   
LIMIT_BAL                 0.0028       0.0006       13.0000  0.0215       8.0000        0.0073   
BILL_AMT2                 0.0023       0.0014       14.0000  0.0060      18.0000        0.0000   
PAY_5                     0.0022       0.0014       15.0000  0.0025      26.0000       -0.0004   

                    LR perm std  LR perm hạng  
PAY_0                    0.0060        3.0000  
NUM_MONTHS_DELAYED       0.0059        1.0000  
MAX_DELAY                0.0046        4.0000  
PAY_RATIO                0.0004       19.0000  
CREDIT_UTIL              0.0017       16.0000  
PAY_2                    0.0016        9.0000  
PAY_0_revolving          0.0066        2.0000  
PAY_6                    0.0011       41.0000  
BILL_AMT1                0.0017       10.0000  
AVG_BILL_AMT             0.0000       34.0000  
PAY_3                    0.0005       21.0000  
AVG_PAY_AMT              0.0000       34.0000  
LIMIT_BAL                0.0029        7.0000  
BILL_AMT2                0.0000       34.0000  
PAY_5                    0.0005       40.0000
```

```
Hệ số chặn: -0.1705; C = 58.48
```

```
Hệ số  Odds ratio
Đặc trưng (đã chuẩn hoá)                    
EDUCATION_4              -0.9086      0.4031
PAY_0_revolving          -0.5234      0.5925
NUM_MONTHS_DELAYED        0.4229      1.5264
PAY_0                     0.2793      1.3222
MAX_DELAY                 0.2775      1.3198
LIMIT_BAL                -0.2239      0.7994
PAY_2_revolving           0.2062      1.2290
PAY_2_nouse              -0.1993      0.8193
PAY_0_nouse               0.1787      1.1957
PAY_AMT1                 -0.1606      0.8516
PAY_AMT2                 -0.1544      0.8569
MARRIAGE_2               -0.1518      0.8591
PAY_2                    -0.1338      0.8748
CREDIT_UTIL               0.1250      1.1331
BILL_AMT1                 0.1208      1.1284
```

### SHAP cho Random Forest

SHAP (`TreeExplainer`) phân rã xác suất vỡ nợ dự đoán của từng khách hàng thành đóng góp của từng đặc trưng, tính trên mẫu 1.000 dòng của tập test. Summary plot (beeswarm) cho thấy hướng tác động (giá trị đặc trưng cao/thấp đẩy xác suất lên hay xuống); waterfall plot giải thích dự đoán cho 2 khách hàng cụ thể (1 vỡ nợ, 1 không vỡ nợ).

```
SHAP cho 1000 dòng trong 0 giây (shap 0.52.0)
```

```
NUM_MONTHS_DELAYED   0.0740
MAX_DELAY            0.0578
PAY_0                0.0481
AVG_PAY_AMT          0.0211
CREDIT_UTIL          0.0187
LIMIT_BAL            0.0160
AVG_BILL_AMT         0.0158
PAY_2                0.0146
PAY_AMT2             0.0093
BILL_AMT1            0.0092
```

Hai khách hàng minh hoạ được chọn là khách hàng vỡ nợ có xác suất dự đoán cao nhất và khách hàng không vỡ nợ có xác suất thấp nhất trong mẫu 1.000 dòng.

**Nhận xét 4.6 (CH1):**  

* **Hành vi trả nợ gần đây chi phối dự đoán.** Với RF, top 3 theo permutation importance là **PAY_0** (PR-AUC giảm 0,0847 khi xáo trộn), **NUM_MONTHS_DELAYED** (0,0550) và **MAX_DELAY** (0,0388). Theo MDI và SHAP (mean |SHAP|: NUM_MONTHS_DELAYED 0,074; MAX_DELAY 0,058; PAY_0 0,048), cùng 3 đặc trưng này vẫn đứng đầu, chỉ khác thứ tự. Với LR, top đầu là NUM_MONTHS_DELAYED (0,0930), PAY_0_revolving (0,0588), PAY_0 (0,0295) và MAX_DELAY (0,0221).
* **Nhóm thứ hai là năng lực tài chính và mức sử dụng tín dụng:** PAY_RATIO, CREDIT_UTIL, AVG_PAY_AMT, LIMIT_BAL, BILL_AMT1. Beeswarm SHAP cho thấy hướng tác động phù hợp nghiệp vụ: CREDIT_UTIL cao làm tăng rủi ro; LIMIT_BAL cao, AVG_PAY_AMT cao và PAY_RATIO cao làm giảm rủi ro.
* **Biến nhân khẩu học ảnh hưởng yếu.** Với RF, AGE, SEX, MARRIAGE, EDUCATION không có trong top 15 permutation importance. Hệ số âm lớn của EDUCATION_4 trong LR (−0,91; odds ratio 0,40) thuộc về nhóm "khác" chỉ chiếm 1,6% dữ liệu, nên không nên diễn giải như một yếu tố chính.
* **Permutation importance so với MDI (RF):** tương quan hạng Spearman 0,686, 12/15 đặc trưng chung trong top 15. MDI đánh giá cao các đặc trưng liên tục nhiều giá trị: AVG_PAY_AMT xếp hạng 5 theo MDI nhưng 12 theo permutation, AVG_BILL_AMT hạng 6 so với 10. Ngược lại, MDI đánh giá thấp PAY_6 (hạng 28 so với 8). Permutation importance trên test đáng tin cậy hơn để trả lời CH1.
* **Hệ số LR (đã chuẩn hoá):** tăng 1 độ lệch chuẩn NUM_MONTHS_DELAYED làm odds vỡ nợ tăng 1,53 lần; PAY_0 và MAX_DELAY mỗi biến ×1,32; LIMIT_BAL ×0,80; PAY_AMT1/PAY_AMT2 ×0,85–0,86. Khách hàng dùng tín dụng quay vòng ở tháng gần nhất (PAY_0_revolving = 1) có odds thấp hơn (×0,59) khi giữ các biến khác không đổi. Điều này khẳng định lại giá trị của cờ nhị phân tạo ở Chương 3.
* **Waterfall SHAP:** khách hàng vỡ nợ minh hoạ (P = 0,934) trễ hạn cả 6 tháng (NUM_MONTHS_DELAYED = 6, MAX_DELAY = 3, PAY_0 = 3). Ba đặc trưng này đẩy xác suất từ mức nền 0,5 lên thêm +0,14, +0,11 và +0,10. Khách hàng không vỡ nợ (P = 0,149) không trễ hạn tháng nào, trả trung bình 16.500/tháng và chỉ dùng 16,8% hạn mức. Mọi đặc trưng chính đều kéo xác suất xuống. (Mức nền 0,5 là do `class_weight='balanced'`.)

**Trả lời CH1:** các yếu tố ảnh hưởng mạnh nhất đến khả năng vỡ nợ là **lịch sử trễ hạn gần đây**: tình trạng trả nợ tháng gần nhất (PAY_0), số tháng trễ hạn trong 6 tháng (NUM_MONTHS_DELAYED) và mức trễ hạn lớn nhất (MAX_DELAY). Tiếp theo là **khả năng và mức độ thanh toán** (PAY_RATIO, AVG_PAY_AMT, PAY_AMT) và **mức sử dụng/hạn mức tín dụng** (CREDIT_UTIL, LIMIT_BAL). Kết luận này nhất quán giữa hai mô hình (RF, LR) và ba phương pháp (permutation importance, MDI, SHAP). Các đặc trưng tổng hợp tạo ở bước Feature Engineering (NUM_MONTHS_DELAYED, MAX_DELAY, CREDIT_UTIL, PAY_RATIO) chiếm phần lớn nhóm đầu.

<a id='ch4_6b'></a>
### 4.6b. Diễn giải mô hình cuối sau mục 4.5b

Nếu mô hình cuối chuyển sang **RF A + ngưỡng chi phí**, permutation importance (`scoring='average_precision'`, 30 lần) và SHAP được tính lại cho mô hình này. Mẫu 1.000 dòng và **hai khách hàng minh hoạ giữ nguyên như mục 4.6**, để so sánh trực tiếp với RF B. Với RF A, mức nền SHAP E[f(X)] (xác suất trung bình của mô hình) phải xấp xỉ tỷ lệ vỡ nợ thực tế ≈ 0,221 thay vì 0,5.

```
Tương quan hạng Spearman permutation importance RF A vs RF B = 0.838; top 15 chung: 13/15
                    RF A perm mean  RF A perm std  RF A hạng  RF B perm mean  RF B hạng
PAY_0                       0.1480         0.0084     1.0000          0.0847     1.0000
NUM_MONTHS_DELAYED          0.0256         0.0040     2.0000          0.0550     2.0000
MAX_DELAY                   0.0188         0.0046     3.0000          0.0388     3.0000
PAY_RATIO                   0.0138         0.0038     4.0000          0.0137     4.0000
CREDIT_UTIL                 0.0083         0.0017     5.0000          0.0102     5.0000
AVG_BILL_AMT                0.0049         0.0012     6.0000          0.0034    10.0000
BILL_AMT1                   0.0040         0.0015     7.0000          0.0040     9.0000
LIMIT_BAL                   0.0040         0.0007     8.0000          0.0028    13.0000
AVG_PAY_AMT                 0.0032         0.0011     9.0000          0.0029    12.0000
PAY_2                       0.0028         0.0013    10.0000          0.0099     6.0000
PAY_6                       0.0022         0.0019    11.0000          0.0043     8.0000
PAY_0_revolving             0.0016         0.0016    12.0000          0.0068     7.0000
PAY_3                       0.0011         0.0009    13.0000          0.0032    11.0000
AGE                         0.0011         0.0006    14.0000          0.0004    24.0000
PAY_AMT6                    0.0008         0.0014    15.0000          0.0001    34.0000
```

```
SHAP base value E[f(X)]: RF A = 0.2213, RF B = 0.5001 (tỷ lệ vỡ nợ train = 0.2213)
```

```
Khách hàng 19965: P(default) RF A = 0.8470, RF B = 0.9338; dự đoán RF A = 1 (ngưỡng 0.17)
```

```
Khách hàng 20558: P(default) RF A = 0.0615, RF B = 0.1487; dự đoán RF A = 0 (ngưỡng 0.17)
PAY_0                0.0627
MAX_DELAY            0.0357
NUM_MONTHS_DELAYED   0.0349
AVG_PAY_AMT          0.0143
PAY_2                0.0117
AVG_BILL_AMT         0.0108
LIMIT_BAL            0.0106
CREDIT_UTIL          0.0070
PAY_RATIO            0.0056
BILL_AMT1            0.0042
```

**Nhận xét 4.6b:** 

* **Mức nền SHAP của RF A là 0,2213**, đúng bằng tỷ lệ vỡ nợ trên train, so với 0,5001 của RF B. Đóng góp SHAP giờ là độ lệch so với xác suất vỡ nợ trung bình thực tế, nên dễ giải thích với người dùng nghiệp vụ.
* **Thứ tự quan trọng gần như giữ nguyên:** tương quan hạng Spearman của permutation importance giữa RF A và RF B là 0,838, với 13/15 đặc trưng chung trong top 15. Năm đặc trưng đầu giống hệt: PAY_0, NUM_MONTHS_DELAYED, MAX_DELAY, PAY_RATIO, CREDIT_UTIL. Khác biệt chính là RF A dựa **nhiều hơn vào PAY_0**: xáo trộn PAY_0 làm PR-AUC giảm 0,1480, so với 0,0847 ở RF B. RF A cũng dựa ít hơn vào NUM_MONTHS_DELAYED (0,0256 so với 0,0550) và MAX_DELAY (0,0188 so với 0,0388). Theo mean |SHAP|, PAY_0 cũng đứng đầu (0,0627), tiếp theo là MAX_DELAY (0,0357) và NUM_MONTHS_DELAYED (0,0349).
* **Hai khách hàng minh hoạ (giống mục 4.6):**
  * Khách hàng 19965 (vỡ nợ) có P(default) = 0,847 theo RF A (RF B: 0,934). Xuất phát từ mức nền 0,221, PAY_0 = 3 cộng +0,27, NUM_MONTHS_DELAYED = 6 cộng +0,11, MAX_DELAY = 3 cộng +0,10.
  * Khách hàng 20558 (không vỡ nợ) có P(default) = 0,062 theo RF A, thuộc nhóm rủi ro Thấp. RF B cho 0,149, dễ gây hiểu nhầm là rủi ro trung bình.
* Kết luận cho CH1 ở mục 4.6 không đổi: lịch sử trễ hạn gần đây, đặc biệt là tình trạng tháng gần nhất PAY_0, là yếu tố quyết định, tiếp theo là khả năng thanh toán và mức sử dụng hạn mức.

<a id='ch4_7'></a>
## 4.7. Chọn mô hình và lưu

### Nhóm rủi ro

Xác suất dự đoán của mô hình cuối được chia thành 3 nhóm rủi ro: **Thấp (< 0,10)**, **Trung bình (0,10–0,30)**, **Cao (≥ 0,30)**. Bảng dưới cho số khách hàng và tỷ lệ vỡ nợ thực tế từng nhóm trên tập test, kèm phân bố trên xác suất OOF của train để đối chiếu. Để có phương án ngưỡng nhóm khác, ta lấy **phân vị 50% và 80% của xác suất OOF trên train**: nửa số khách hàng được xếp Thấp, 30% Trung bình, 20% Cao. Phương án này chỉ dựa trên train, không dùng tập test để chọn.

```
Mô hình cuối: RF A + ngưỡng chi phí; ngưỡng nhóm theo phân vị 50%/80% của OOF train: [0.14, 0.33]
```

```
Số KH  Tỷ trọng  Số vỡ nợ  Tỷ lệ vỡ nợ thực tế  \
Phương án ngưỡng nhóm   Tập       Nhóm                                                         
Cố định 0,10 / 0,30     OOF train Thấp         7558    0.3153       519               0.0687   
                                  Trung bình  11236    0.4687      1922               0.1711   
                                  Cao          5178    0.2160      2863               0.5529   
                        Test      Thấp         1882    0.3140       143               0.0760   
                                  Trung bình   2819    0.4704       484               0.1717   
                                  Cao          1292    0.2156       699               0.5410   
Phân vị OOF 0.14 / 0.33 OOF train Thấp        11741    0.4898      1050               0.0894   
                                  Trung bình   7458    0.3111      1532               0.2054   
                                  Cao          4773    0.1991      2722               0.5703   
                        Test      Thấp         2930    0.4889       283               0.0966   
                                  Trung bình   1891    0.3155       390               0.2062   
                                  Cao          1172    0.1956       653               0.5572   

                                              P(default) TB dự đoán  
Phương án ngưỡng nhóm   Tập       Nhóm                               
Cố định 0,10 / 0,30     OOF train Thấp                       0.0779  
                                  Trung bình                 0.1681  
                                  Cao                        0.5464  
                        Test      Thấp                       0.0785  
                                  Trung bình                 0.1671  
                                  Cao                        0.5442  
Phân vị OOF 0.14 / 0.33 OOF train Thấp                       0.0922  
                                  Trung bình                 0.2042  
                                  Cao                        0.5660  
                        Test      Thấp                       0.0923  
                                  Trung bình                 0.2041  
                                  Cao                        0.5678
```

**Nhận xét nhóm rủi ro:** với ngưỡng cố định 0,10 / 0,30, ba nhóm trên test có tỷ trọng **31,4% / 47,0% / 21,6%** và tỷ lệ vỡ nợ thực tế **7,6% / 17,2% / 54,1%**. Tỷ lệ tăng đơn điệu và cách xa nhau: nhóm Cao có rủi ro gấp ≈ 7 lần nhóm Thấp. Xác suất dự đoán trung bình từng nhóm (7,9% / 16,7% / 54,4%) khớp với tỷ lệ thực tế, xác nhận mô hình được hiệu chỉnh tốt. Phân bố trên OOF train gần như giống hệt (31,5% / 46,9% / 21,6%; 6,9% / 17,1% / 55,3%). Các nhóm không lệch quá mức nên **giữ ngưỡng nhóm 0,10 / 0,30** vì dễ giải thích. Phương án theo phân vị OOF train (0,14 / 0,33, tỷ trọng ≈ 50% / 30% / 20%) cho tỷ lệ vỡ nợ 9,7% / 20,6% / 55,7% trên test. Phương án này chỉ nên dùng nếu ngân hàng muốn nhóm Thấp chiếm khoảng một nửa danh mục.

Lưu ý: ngưỡng quyết định 0,17 nằm **trong** nhóm Trung bình. Khách hàng có xác suất 0,17–0,30 được mô hình gắn nhãn "có nguy cơ vỡ nợ" nhưng thuộc nhóm rủi ro Trung bình. Ứng dụng nên hiển thị cả xác suất, nhóm rủi ro và nhãn cảnh báo.

### Tổng hợp và chọn mô hình cuối

* **CH1 – Yếu tố ảnh hưởng:** lịch sử trễ hạn gần đây (PAY_0, NUM_MONTHS_DELAYED, MAX_DELAY) chi phối dự đoán. Với mô hình cuối, xáo trộn PAY_0 làm PR-AUC test giảm 0,148. Tiếp theo là khả năng thanh toán (PAY_RATIO, AVG_PAY_AMT) và mức sử dụng/hạn mức tín dụng (CREDIT_UTIL, LIMIT_BAL). Biến nhân khẩu học ảnh hưởng yếu. Kết luận nhất quán giữa RF A, RF B, LR và giữa permutation importance, MDI, SHAP.
* **CH2 – So sánh mô hình:** trên 50 fold, Random Forest đạt PR-AUC 0,5636, cao hơn DT (0,5350) và LR (0,5236). Chênh lệch có ý nghĩa theo corrected resampled t-test với hiệu chỉnh Bonferroni (p < 10⁻⁷). Trên test, RF vẫn đứng đầu (PR-AUC 0,5559–0,5586).
* **CH3 – Mất cân bằng và ngưỡng:** các kỹ thuật xử lý mất cân bằng không thay đổi khả năng phân biệt (PR-AUC/ROC-AUC) mà chỉ dịch điểm làm việc. Trong 4 cấu hình ở ngưỡng mặc định, `class_weight='balanced'` tốt nhất theo F2 (Recall ≈ 0,61–0,66 so với ≈ 0,36 khi không xử lý), còn SMOTE không tốt hơn. Tuy nhiên, **hạ ngưỡng theo chi phí trên mô hình không trọng số (RF A, ngưỡng 0,17)** đạt cùng mục tiêu tốt hơn: trên CV có F2 cao hơn (0,6249 so với 0,5913), chi phí 5:1 thấp hơn (0,5490 so với 0,5641), cả hai có ý nghĩa, và giữ được xác suất đã hiệu chỉnh (Brier 0,1334 so với 0,1783). Trọng số lớp phóng đại xác suất (trung bình 0,43 so với tỷ lệ thực 0,22). Dịch ngưỡng làm được việc tương tự mà không làm sai xác suất.

**Mô hình cuối: Random Forest cấu hình A + ngưỡng chi phí** (200 cây, `max_depth=6`, `min_samples_leaf=21`, `max_features=0,5`, `class_weight=None`, ngưỡng **0,17** chọn theo chi phí 5:1 trên OOF train). Lý do:

1. **Hiệu năng:** khả năng phân biệt tốt nhất trong 3 thuật toán, bằng RF B (PR-AUC test 0,5586, ROC-AUC 0,7733). F2 và chi phí 5:1 trên CV tốt hơn RF B có ý nghĩa thống kê.
2. **Xác suất có ý nghĩa**, đúng yêu cầu của ứng dụng Chương 5: calibration bám đường chéo, xác suất trung bình 0,2206 so với tỷ lệ thực 0,2213, và nhóm rủi ro có tỷ lệ vỡ nợ đúng như dự đoán.
3. **Diễn giải:** SHAP có mức nền 0,221 bằng tỷ lệ vỡ nợ trung bình, nên đóng góp của từng đặc trưng dễ đọc. Thứ tự đặc trưng quan trọng nhất quán với các mô hình khác.

**Đánh đổi cần nêu rõ:** mô hình cuối ưu tiên Recall (70,9% trên test so với 61,6% của RF B) và xác suất đúng. Cái giá là nhiều cảnh báo nhầm hơn: Precision 0,379 so với 0,469, Accuracy 0,679 so với 0,761. Ngưỡng 0,17 chỉ tối ưu khi tỷ lệ chi phí C<sub>FN</sub>:C<sub>FP</sub> ≈ 5:1. Nếu tỷ lệ chi phí thực tế khác, đặt lại ngưỡng theo phân tích độ nhạy ở mục 4.3 (3:1 → ≈ 0,26; 10:1 → ≈ 0,09) mà **không cần huấn luyện lại** mô hình. RF B vẫn được lưu để đối chiếu.

Pipeline cuối (tiền xử lý, mô hình và ngưỡng, bọc trong `FixedThresholdClassifier`) được lưu vào `models/final_model.joblib`, metadata vào `models/metadata.json`, và bản RF B vào `models/rf_config_B.joblib`.

```
Đã lưu models/final_model.joblib, models/rf_config_B.joblib và models/metadata.json
{
  "model_name": "Random Forest — RF A + ngưỡng chi phí",
  "imbalance_config": "A: none",
  "hyperparameters": {
    "bootstrap": true,
    "ccp_alpha": 0.0,
    "class_weight": null,
    "criterion": "gini",
    "max_depth": 6,
    "max_features": 0.5,
    "max_leaf_nodes": null,
    "max_samples": null,
    "min_impurity_decrease": 0.0,
    "min_samples_leaf": 21,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "monotonic_cst": null,
    "n_estimators": 200,
    "n_jobs": null,
    "oob_score": false,
    "random_state": 42,
    "verbose": 0,
    "warm_start": false
  },
  "threshold": 0.17,
  "threshold_rule": "min expected cost, CFN:CFP=5:1, chọn trên OOF train",
  "calibrated_probabilities": true,
  "risk_bands": {
    "edges": [
      0.1,
      0.3
    ],
    "labels": [
      "Thấp",
      "Trung bình",
      "Cao"
    ],
    "rule": "Thấp: p < edges[0]; Trung bình: edges[0] <= p < edges[1]; Cao: p >= edges[1]",
    "test_default_rate": {
      "Thấp": 0.076,
      "Trung bình": 0.1717,
      "Cao": 0.541
    }
  },
  "note": "Đầu vào là các cột sau Feature Engineering của Chương 3 cộng BILL_TREND (hàm add_bill_trend); đối tượng là FixedThresholdClassifier nên predict() đã áp dụng ngưỡng, predict_proba() trả xác suất vỡ nợ.",
  "cv_repeated_5x10_mean": {
    "Precision": 0.3862,
    "Recall": 0.7412,
    "F1": 0.5068,
    "F2": 0.6249,
    "Chi phí/KH (5:1)": 0.549,
    "PR-AUC": 0.5637,
    "ROC-AUC": 0.7853,
    "Brier": 0.1334,
    "Log loss": 0.4261
  },
  "test_metrics": {
    "Ngưỡng": 0.17,
    "Precision": 0.3793,
    "Recall": 0.7089,
    "F1": 0.4942,
    "F2": 0.604,
    "Chi phí/KH (5:1)": 0.5787,
    "PR-AUC": 0.5586,
    "ROC-AUC": 0.7733,
    "Gini": 0.5466,
    "KS": 0.4239,
    "Brier": 0.1359,
    "Log loss": 0.4332,
    "P(default) trung bình": 0.2206
  },
  "test_bootstrap_ci95": {
    "ROC-AUC": [
      0.7586,
      0.7879
    ],
    "PR-AUC": [
      0.5299,
      0.5848
    ],
    "Brier": [
      0.1299,
      0.1417
    ]
  },
  "random_state": 42,
  "sklearn_version": "1.9.1",
  "imblearn_version": "0.14.2"
}
```

<a id='ch4_8'></a>
## 4.8. Kết quả huấn luyện (OOF) và tổng hợp Accuracy

Mục này bổ sung **kết quả huấn luyện** cho từng mô hình (Accuracy, ma trận nhầm lẫn, Precision, Recall, F1) theo khung báo cáo. Các mục trước chỉ báo cáo trung bình CV.

* **Trên train:** dùng dự đoán **out-of-fold (OOF)**, `cross_val_predict(method='predict_proba')` với `skf` (StratifiedKFold 5, shuffle, random_state = 42). Mỗi khách hàng trong train được dự đoán bởi mô hình **không** học từ chính khách hàng đó, nên đây là ước lượng trung thực của hiệu năng huấn luyện, không phải điểm "học thuộc" khi dự đoán lại dữ liệu đã fit. Nhãn dự đoán = 1 nếu xác suất ≥ ngưỡng của mô hình.
* **Trên test:** dùng đúng các mô hình đã fit ở mục 4.5 (LR, DT, RF cấu hình B) và 4.5b (RF A + ngưỡng chi phí), **không fit lại**.
* Bốn mô hình với siêu tham số và cấu hình đọc từ các biến có sẵn (`final_cfg`, `final_thresholds`, `best_params`, `RF_COST_THRESHOLD`): LR, DT, RF cấu hình B với ngưỡng 0,5; RF cấu hình A với ngưỡng chi phí (mô hình cuối).

```
Tập train: 23972 khách hàng, 5304 vỡ nợ (22.13%)

===== Logistic Regression — B: balanced, ngưỡng 0.50 (OOF trên train) =====
              precision    recall  f1-score   support

           0     0.8787    0.7989    0.8369     18668
           1     0.4637    0.6120    0.5276      5304

    accuracy                         0.7576     23972
   macro avg     0.6712    0.7054    0.6823     23972
weighted avg     0.7869    0.7576    0.7685     23972

Accuracy = 0.7576 | lớp vỡ nợ: Precision = 0.4637, Recall = 0.6120, F1 = 0.5276
Ma trận nhầm lẫn: TN = 14914, FP = 3754, FN = 2058, TP = 3246

===== Decision Tree — B: balanced, ngưỡng 0.50 (OOF trên train) =====
              precision    recall  f1-score   support

           0     0.8851    0.7504    0.8122     18668
           1     0.4279    0.6571    0.5183      5304

    accuracy                         0.7297     23972
   macro avg     0.6565    0.7037    0.6652     23972
weighted avg     0.7839    0.7297    0.7471     23972

Accuracy = 0.7297 | lớp vỡ nợ: Precision = 0.4279, Recall = 0.6571, F1 = 0.5183
Ma trận nhầm lẫn: TN = 14008, FP = 4660, FN = 1819, TP = 3485

===== Random Forest — B: balanced, ngưỡng 0.50 (OOF trên train) =====
              precision    recall  f1-score   support

           0     0.8840    0.7983    0.8389     18668
           1     0.4706    0.6312    0.5392      5304

    accuracy                         0.7613     23972
   macro avg     0.6773    0.7147    0.6891     23972
weighted avg     0.7925    0.7613    0.7726     23972

Accuracy = 0.7613 | lớp vỡ nợ: Precision = 0.4706, Recall = 0.6312, F1 = 0.5392
Ma trận nhầm lẫn: TN = 14902, FP = 3766, FN = 1956, TP = 3348

===== Random Forest — A: none + ngưỡng chi phí, ngưỡng 0.17 (OOF trên train) =====
              precision    recall  f1-score   support

           0     0.8985    0.6825    0.7757     18668
           1     0.3947    0.7287    0.5120      5304

    accuracy                         0.6927     23972
   macro avg     0.6466    0.7056    0.6439     23972
weighted avg     0.7870    0.6927    0.7174     23972

Accuracy = 0.6927 | lớp vỡ nợ: Precision = 0.3947, Recall = 0.7287, F1 = 0.5120
Ma trận nhầm lẫn: TN = 12740, FP = 5928, FN = 1439, TP = 3865
```

```
Accuracy  Precision  Recall     F1     F2  \
Logistic Regression (B: balanced, ngưỡng 0.50)        0.7576     0.4637  0.6120 0.5276 0.5752   
Decision Tree (B: balanced, ngưỡng 0.50)              0.7297     0.4279  0.6571 0.5183 0.5935   
Random Forest (B: balanced, ngưỡng 0.50)              0.7613     0.4706  0.6312 0.5392 0.5909   
Random Forest (A: none + ngưỡng chi phí, ngưỡng...    0.6927     0.3947  0.7287 0.5120 0.6232   

                                                           TN        FP        FN        TP  
Logistic Regression (B: balanced, ngưỡng 0.50)     14914.0000 3754.0000 2058.0000 3246.0000  
Decision Tree (B: balanced, ngưỡng 0.50)           14008.0000 4660.0000 1819.0000 3485.0000  
Random Forest (B: balanced, ngưỡng 0.50)           14902.0000 3766.0000 1956.0000 3348.0000  
Random Forest (A: none + ngưỡng chi phí, ngưỡng... 12740.0000 5928.0000 1439.0000 3865.0000
```

```
Accuracy  Precision  Recall     F1     F2  \
Logistic Regression (B: balanced, ngưỡng 0.50)        0.7567     0.4611  0.5905 0.5179 0.5591   
Decision Tree (B: balanced, ngưỡng 0.50)              0.7265     0.4199  0.6192 0.5005 0.5655   
Random Forest (B: balanced, ngưỡng 0.50)              0.7606     0.4687  0.6161 0.5324 0.5797   
Random Forest (A: none + ngưỡng chi phí, ngưỡng...    0.6790     0.3793  0.7089 0.4942 0.6040   

                                                    ROC-AUC  PR-AUC        TN        FP       FN  \
Logistic Regression (B: balanced, ngưỡng 0.50)       0.7624  0.5275 3752.0000  915.0000 543.0000   
Decision Tree (B: balanced, ngưỡng 0.50)             0.7535  0.5272 3533.0000 1134.0000 505.0000   
Random Forest (B: balanced, ngưỡng 0.50)             0.7736  0.5559 3741.0000  926.0000 509.0000   
Random Forest (A: none + ngưỡng chi phí, ngưỡng...   0.7733  0.5586 3129.0000 1538.0000 386.0000   

                                                         TP  
Logistic Regression (B: balanced, ngưỡng 0.50)     783.0000  
Decision Tree (B: balanced, ngưỡng 0.50)           821.0000  
Random Forest (B: balanced, ngưỡng 0.50)           817.0000  
Random Forest (A: none + ngưỡng chi phí, ngưỡng... 940.0000
```

```
Train (OOF): RF A phát hiện thêm 517 khách hàng vỡ nợ (TP 3348 → 3865), thêm 2162 cảnh báo nhầm (FP 3766 → 5928); Accuracy 0.7613 → 0.6927
Test: RF A phát hiện thêm 123 khách hàng vỡ nợ (TP 817 → 940), thêm 612 cảnh báo nhầm (FP 926 → 1538); Accuracy 0.7606 → 0.6790
```

**Nhận xét 4.8:**

* **Kết quả huấn luyện (OOF trên 23.972 khách hàng train, 5.304 vỡ nợ):**
  * **Logistic Regression** (B, ngưỡng 0,5): Accuracy **0,7576**, Precision 0,4637, Recall 0,6120, F1 0,5276 (TN 14.914, FP 3.754, FN 2.058, TP 3.246).
  * **Decision Tree** (B, ngưỡng 0,5): Accuracy **0,7297**, Precision 0,4279, Recall 0,6571, F1 0,5183 (TN 14.008, FP 4.660, FN 1.819, TP 3.485).
  * **Random Forest** (B, ngưỡng 0,5): Accuracy **0,7613**, Precision 0,4706, Recall 0,6312, F1 0,5392 (TN 14.902, FP 3.766, FN 1.956, TP 3.348).
  * **Random Forest A + ngưỡng chi phí 0,17** (mô hình cuối): Accuracy **0,6927**, Precision 0,3947, Recall 0,7287, F1 0,5120 (TN 12.740, FP 5.928, FN 1.439, TP 3.865).
* **Train (OOF) và test chênh lệch nhỏ**, không có dấu hiệu overfit. Accuracy trên test thấp hơn OOF từ 0,0008 (LR) đến 0,0137 (RF A). F1 chênh 0,007–0,018. Recall chênh nhiều nhất ở DT (0,6571 → 0,6192, −0,038), các mô hình còn lại ≤ 0,022. Kết quả OOF là dự đoán trên dữ liệu mà mô hình chưa học, nên đây là ước lượng trung thực. Ngược lại, Accuracy khi dự đoán lại chính dữ liệu đã fit sẽ lạc quan hơn nhiều, nhất là với cây và rừng.
* **So sánh 3 thuật toán trong cùng điều kiện (cấu hình B, ngưỡng 0,5):** trên test, Random Forest có Accuracy cao nhất (0,7606) và F1 cao nhất (0,5324), tiếp theo là Logistic Regression (0,7567; 0,5179) và Decision Tree (0,7265; 0,5005). DT có Recall cao nhất (0,6192) nhưng nhiều cảnh báo nhầm nhất (FP 1.134). Kết quả khớp với so sánh CV và kiểm định ở mục 4.4. **RF A + ngưỡng 0,17 không tham gia so sánh thuật toán.** Đây là bước chọn ngưỡng vận hành (và bỏ trọng số lớp để xác suất đúng) sau khi RF đã được chọn là thuật toán tốt nhất.
* **Vì sao Accuracy của RF A (0,6790 trên test) thấp hơn RF B (0,7606) dù PR-AUC/ROC-AUC tương đương (0,5586/0,7733 so với 0,5559/0,7736):** hai mô hình xếp hạng khách hàng gần như giống nhau. Chỉ khác ngưỡng: 0,17 thấp hơn nhiều nên nhiều khách hàng hơn bị gắn nhãn vỡ nợ. Trên test, RF A **phát hiện thêm 123 khách hàng vỡ nợ** (TP 817 → 940, Recall 0,6161 → 0,7089), đổi lại **thêm 612 cảnh báo nhầm** (FP 926 → 1.538). Trên train OOF: thêm 517 TP và 2.162 FP. Accuracy coi mỗi lỗi như nhau nên giảm. Theo giả định chi phí 5:1, 123 × 5 = 615 đơn vị tiết kiệm từ FN gần như bù đúng 612 đơn vị chi phí FP tăng thêm.
* **Accuracy không phải độ đo phù hợp để chọn mô hình ở bài toán này.** Một mô hình luôn dự đoán "không vỡ nợ" đạt Accuracy 1 − 0,2213 = **0,7787** trên test, cao hơn mọi mô hình ở trên, nhưng không phát hiện được khách hàng vỡ nợ nào (Recall = 0). Vì vậy đồ án chọn mô hình theo PR-AUC, F2 và chi phí kỳ vọng. Accuracy chỉ được báo cáo cho đủ khung báo cáo.


# PHẦN B. BẢNG KẾT QUẢ CHƯƠNG 4 (results/ch4*.csv, số đầy đủ)


## ch4_baseline_vs_tuned_test.csv

```csv
,,Baseline (Chương 3),Baseline (Chương 3),Baseline (Chương 3),Baseline (Chương 3),Baseline (Chương 3),Sau tinh chỉnh (Chương 4),Sau tinh chỉnh (Chương 4),Sau tinh chỉnh (Chương 4),Sau tinh chỉnh (Chương 4),Sau tinh chỉnh (Chương 4),Chênh lệch,Chênh lệch,Chênh lệch,Chênh lệch,Chênh lệch
,Phương án baseline,Precision,Recall,F1,ROC-AUC,PR-AUC,Precision,Recall,F1,ROC-AUC,PR-AUC,Precision,Recall,F1,ROC-AUC,PR-AUC
Logistic Regression,balanced,0.4609,0.5905,0.5177,0.7616,0.5275,0.4611,0.5905,0.5179,0.7624,0.5275,0.0003,0.0,0.0002,0.0008,0.0
Decision Tree,smote,0.3665,0.4804,0.4158,0.6221,0.291,0.4199,0.6192,0.5005,0.7535,0.5272,0.0534,0.1388,0.0847,0.1313,0.2362
Random Forest,balanced,0.5441,0.4789,0.5094,0.7607,0.5279,0.4687,0.6161,0.5324,0.7736,0.5559,-0.0754,0.1373,0.023,0.0129,0.028
```


## ch4_cost_threshold.csv

```csv
Mô hình,CFN:CFP,Ngưỡng tối ưu,Ngưỡng Bayes 1/(1+CFN),Chi phí/KH tại ngưỡng tối ưu,"Chi phí/KH tại 0,5",Recall,Precision,F2
Logistic Regression,3:1,0.25,0.25,0.4116,0.4615,0.5777,0.4933,0.5586
Logistic Regression,5:1,0.15,0.167,0.5688,0.7401,0.7376,0.3695,0.615
Logistic Regression,10:1,0.11,0.091,0.7203,1.4366,0.9112,0.2779,0.6259
Decision Tree,3:1,0.29,0.25,0.4019,0.4631,0.5722,0.5177,0.5604
Decision Tree,5:1,0.16,0.167,0.566,0.7432,0.767,0.3551,0.6226
Decision Tree,10:1,0.08,0.091,0.7211,1.4434,0.9146,0.2755,0.6247
Random Forest,3:1,0.26,0.25,0.3955,0.4652,0.585,0.5188,0.5705
Random Forest,5:1,0.17,0.167,0.5474,0.751,0.7287,0.3947,0.6232
Random Forest,10:1,0.09,0.091,0.7012,1.4656,0.9282,0.2747,0.6289
```


## ch4_dt_max_depth_prauc.csv

```csv
max_depth,count,mean,max
3,2,0.5013,0.5017
4,2,0.5164,0.5175
5,5,0.524,0.5288
6,1,0.5199,0.5199
7,3,0.5215,0.5356
8,3,0.5143,0.5374
9,3,0.508,0.527
10,3,0.5193,0.5391
11,5,0.5062,0.5382
12,2,0.4876,0.5328
13,3,0.514,0.5234
14,5,0.5187,0.5391
15,2,0.4514,0.5118
16,1,0.4986,0.4986
17,2,0.5166,0.5366
18,2,0.4656,0.4656
19,3,0.5278,0.5375
20,1,0.5059,0.5059
None,2,0.5316,0.5331
```


## ch4_feature_importance.csv

```csv
,RF perm mean,RF perm std,RF perm hạng,RF MDI,RF MDI hạng,LR perm mean,LR perm std,LR perm hạng
PAY_0,0.0847,0.00729,1.0,0.17534,3.0,0.02952,0.00596,3.0
NUM_MONTHS_DELAYED,0.05505,0.00643,2.0,0.28359,1.0,0.09299,0.00594,1.0
MAX_DELAY,0.03876,0.00576,3.0,0.22212,2.0,0.02209,0.00461,4.0
PAY_RATIO,0.01372,0.00425,4.0,0.0213,9.0,0.00089,0.00042,19.0
CREDIT_UTIL,0.01021,0.00181,5.0,0.02472,7.0,0.00153,0.00168,16.0
PAY_2,0.00994,0.00278,6.0,0.05286,4.0,0.00597,0.00163,9.0
PAY_0_revolving,0.00684,0.00297,7.0,0.01121,12.0,0.0588,0.00663,2.0
PAY_6,0.00426,0.00202,8.0,0.0011,28.0,-0.0004,0.00108,41.0
BILL_AMT1,0.00396,0.00164,9.0,0.01772,10.0,0.00457,0.00166,10.0
AVG_BILL_AMT,0.00339,0.00094,10.0,0.02889,6.0,0.0,0.0,34.0
PAY_3,0.00318,0.00215,11.0,0.0079,15.0,0.00053,0.00047,21.0
AVG_PAY_AMT,0.00293,0.00178,12.0,0.0393,5.0,0.0,0.0,34.0
LIMIT_BAL,0.00277,0.00058,13.0,0.02153,8.0,0.00725,0.00294,7.0
BILL_AMT2,0.00227,0.00143,14.0,0.00597,18.0,0.0,0.0,34.0
PAY_5,0.00217,0.00139,15.0,0.00249,26.0,-0.00038,0.00048,40.0
PAY_6_revolving,0.0019,0.00085,16.0,0.00033,36.0,0.00156,0.00108,15.0
BILL_AMT6,0.00111,0.0012,17.0,0.00411,23.0,0.0,0.0,34.0
PAY_AMT3,0.001,0.00148,18.0,0.00721,16.0,0.00012,0.00037,26.0
BILL_AMT3,0.00098,0.00102,19.0,0.00473,21.0,0.0,0.0,34.0
PAY_5_revolving,0.00059,0.00074,20.0,0.00022,38.0,-1e-05,6e-05,38.0
BILL_AMT5,0.00056,0.00033,21.0,0.00492,19.0,0.0,0.0,34.0
BILL_AMT4,0.00047,0.0004,22.0,0.00336,24.0,0.0,0.0,34.0
PAY_AMT5,0.00044,0.00089,23.0,0.00305,25.0,0.00045,0.00055,22.0
AGE,0.00043,0.00059,24.0,0.00474,20.0,0.00177,0.0005,14.0
EDUCATION,0.00032,0.00072,25.0,0.0014,27.0,0.00261,0.00149,12.0
PAY_4_revolving,0.0003,0.00021,26.0,0.00034,35.0,0.00084,0.00039,20.0
PAY_4,0.00028,0.00026,27.0,0.00415,22.0,-0.0001,0.00011,39.0
PAY_AMT2,0.00025,0.00064,28.0,0.01546,11.0,0.00131,0.00117,17.0
PAY_6_nouse,0.00016,0.0001,29.0,0.00044,33.0,8e-05,0.00018,27.0
SEX,0.00014,7e-05,30.0,0.00052,32.0,0.00124,0.00111,18.0
PAY_5_nouse,9e-05,0.00026,31.0,0.00029,37.0,0.0,1e-05,30.0
PAY_AMT4,8e-05,0.0004,32.0,0.00907,14.0,0.00031,0.00074,23.0
MARRIAGE,6e-05,7e-05,33.0,0.00065,30.0,0.00322,0.00129,11.0
PAY_AMT6,6e-05,0.00095,34.0,0.00631,17.0,1e-05,0.00033,29.0
PAY_3_nouse,4e-05,1e-05,35.0,0.00019,39.0,0.0003,0.00108,24.0
PAY_4_nouse,4e-05,1e-05,36.0,0.00018,40.0,7e-05,0.0005,28.0
PAY_0_nouse,2e-05,1e-05,37.0,0.00035,34.0,0.00625,0.00348,8.0
PAY_2_nouse,0.0,1e-05,38.0,0.00013,41.0,0.00902,0.0028,6.0
BILL_TREND,0.0,0.0,39.0,,,0.00014,0.00041,25.0
PAY_3_revolving,-4e-05,0.00029,40.0,0.00064,31.0,-0.00152,0.00088,42.0
PAY_2_revolving,-0.00013,0.00034,41.0,0.00092,29.0,0.01298,0.00242,5.0
PAY_AMT1,-0.00048,0.00046,42.0,0.01025,13.0,0.00195,0.00166,13.0
```


## ch4_hyperparameter_tuning.csv

```csv
Mô hình,Siêu tham số tốt nhất,PR-AUC,ROC-AUC,F1,Recall,Precision,PR-AUC baseline tốt nhất (Chương 3),Chênh lệch PR-AUC
Logistic Regression,"C=58.483337, class_weight=None",0.5258 ± 0.0155,0.7679 ± 0.0106,0.4726 ± 0.0109,0.3705 ± 0.0103,0.6526 ± 0.0162,0.5254 (none),+0.0003
Decision Tree,"ccp_alpha=8e-06, class_weight=None, criterion=entropy, max_depth=14, min_samples_leaf=152",0.5391 ± 0.0134,0.7719 ± 0.0071,0.4639 ± 0.0152,0.3582 ± 0.0180,0.6598 ± 0.0238,0.2924 (balanced),+0.2467
Random Forest,"class_weight=balanced, max_depth=6, max_features=0.5, min_samples_leaf=21, n_estimators=200",0.5645 ± 0.0101,0.7858 ± 0.0077,0.5392 ± 0.0087,0.6312 ± 0.0157,0.4708 ± 0.0097,0.5367 (none),+0.0278
```


## ch4_imbalance_configs.csv

```csv
,,Precision,Recall,F1,F2,PR-AUC,ROC-AUC,Precision std,Recall std,F1 std,F2 std,PR-AUC std,ROC-AUC std
Logistic Regression,A: none,0.6526,0.3705,0.4726,0.4055,0.5258,0.7679,0.0162,0.0103,0.0109,0.0105,0.0155,0.0106
Logistic Regression,B: balanced,0.4638,0.612,0.5277,0.5752,0.5219,0.7685,0.0104,0.0119,0.0099,0.0105,0.0154,0.0102
Logistic Regression,C: SMOTE,0.467,0.6084,0.5283,0.5736,0.5226,0.7686,0.0075,0.0129,0.0071,0.0097,0.0151,0.0105
Logistic Regression,D: none + ngưỡng F1,0.5124,0.5509,0.5304,0.5424,0.5258,0.7679,0.0213,0.0165,0.0075,0.0098,0.0155,0.0106
Decision Tree,A: none,0.6598,0.3582,0.4639,0.3941,0.5391,0.7719,0.0238,0.018,0.0152,0.0173,0.0134,0.0071
Decision Tree,B: balanced,0.4283,0.657,0.5183,0.5934,0.5369,0.7681,0.0105,0.017,0.0064,0.0095,0.0114,0.0052
Decision Tree,C: SMOTE,0.4289,0.6163,0.5054,0.5664,0.5044,0.7526,0.0185,0.015,0.0099,0.0078,0.0182,0.0072
Decision Tree,D: none + ngưỡng F1,0.5332,0.5558,0.5437,0.5508,0.5391,0.7719,0.0223,0.0128,0.0084,0.0073,0.0134,0.0071
Random Forest,A: none,0.6834,0.3541,0.4662,0.3917,0.5649,0.7848,0.0298,0.0175,0.0192,0.018,0.0104,0.0075
Random Forest,B: balanced,0.4708,0.6312,0.5392,0.5908,0.5645,0.7858,0.0097,0.0157,0.0087,0.0116,0.0101,0.0077
Random Forest,C: SMOTE,0.4657,0.6218,0.5324,0.5826,0.5565,0.7823,0.0103,0.0154,0.0099,0.0121,0.0089,0.0064
Random Forest,D: none + ngưỡng F1,0.5338,0.5705,0.5515,0.5627,0.5649,0.7848,0.0103,0.0101,0.0082,0.0088,0.0104,0.0075
```


## ch4_lr_coefficients.csv

```csv
Đặc trưng (đã chuẩn hoá),Hệ số,Odds ratio
EDUCATION_4,-0.9086,0.4031
PAY_0_revolving,-0.5234,0.5925
NUM_MONTHS_DELAYED,0.4229,1.5264
PAY_0,0.2793,1.3222
MAX_DELAY,0.2775,1.3198
LIMIT_BAL,-0.2239,0.7994
PAY_2_revolving,0.2062,1.229
PAY_2_nouse,-0.1993,0.8193
PAY_0_nouse,0.1787,1.1957
PAY_AMT1,-0.1606,0.8516
PAY_AMT2,-0.1544,0.8569
MARRIAGE_2,-0.1518,0.8591
PAY_2,-0.1338,0.8748
CREDIT_UTIL,0.125,1.1331
BILL_AMT1,0.1208,1.1284
PAY_6,-0.097,0.9075
PAY_6_revolving,-0.0885,0.9153
MARRIAGE_3,-0.0781,0.9249
SEX,-0.0624,0.9395
PAY_AMT4,-0.0613,0.9405
PAY_3_revolving,-0.0588,0.9429
PAY_3_nouse,0.0533,1.0547
EDUCATION_2,0.0467,1.0479
PAY_AMT5,-0.0388,0.9619
PAY_RATIO,-0.0332,0.9673
PAY_AMT6,-0.0325,0.968
AGE,0.0304,1.0309
PAY_4_revolving,0.0303,1.0307
PAY_5,-0.03,0.9704
PAY_3,-0.0296,0.9709
EDUCATION_3,-0.029,0.9714
PAY_4_nouse,0.0282,1.0286
PAY_AMT3,-0.0259,0.9744
BILL_TREND,-0.022,0.9782
PAY_6_nouse,0.0127,1.0128
PAY_4,-0.0062,0.9938
PAY_5_revolving,0.0032,1.0032
PAY_5_nouse,-0.0003,0.9997
```


## ch4_lr_feature_sets.csv

```csv
LR (balanced),PR-AUC,ROC-AUC,F1,Recall
Chương 3 (không log1p),0.5216 ± 0.0157,0.7693 ± 0.0099,0.5285 ± 0.0097,0.6135 ± 0.0123
(a) log1p PAY_RATIO,0.5219 ± 0.0154,0.7689 ± 0.0099,0.5276 ± 0.0098,0.6124 ± 0.0120
(b) log1p PAY_RATIO + BILL_TREND,0.5219 ± 0.0154,0.7685 ± 0.0102,0.5277 ± 0.0099,0.6118 ± 0.0123
```


## ch4_lr_vif_a_vs_b.csv

```csv
,VIF (a),VIF (b)
NUM_MONTHS_DELAYED,21.56,21.5
PAY_5_nouse,7.4,7.35
PAY_4_nouse,7.12,7.07
PAY_2_nouse,6.91,6.89
PAY_5,6.52,6.49
PAY_3_nouse,6.52,6.47
PAY_2_revolving,6.44,6.4
PAY_2,6.13,6.09
PAY_4,6.04,6.01
PAY_3,5.97,5.91
PAY_6,5.5,5.45
PAY_6_nouse,5.1,5.17
PAY_0_revolving,5.06,5.03
PAY_0,4.68,4.67
MAX_DELAY,4.62,4.62
PAY_0_nouse,4.41,4.41
PAY_3_revolving,4.43,4.38
PAY_4_revolving,3.95,3.91
PAY_5_revolving,3.79,3.76
CREDIT_UTIL,3.45,3.16
PAY_6_revolving,3.01,2.98
BILL_AMT1,15.72,2.38
LIMIT_BAL,2.36,2.23
EDUCATION_3,1.41,1.41
AGE,1.39,1.39
EDUCATION_2,1.38,1.38
PAY_RATIO,1.35,1.35
MARRIAGE_2,1.34,1.34
PAY_AMT1,1.82,1.27
PAY_AMT3,1.88,1.27
PAY_AMT4,1.75,1.24
PAY_AMT2,2.41,1.24
PAY_AMT5,1.71,1.16
BILL_TREND,,1.16
PAY_AMT6,1.22,1.15
EDUCATION_4,1.04,1.04
SEX,1.03,1.03
MARRIAGE_3,1.03,1.03
BILL_AMT2,28.29,
BILL_AMT5,25.35,
BILL_AMT3,22.47,
BILL_AMT4,21.11,
BILL_AMT6,15.24,
```


## ch4_repeated_cv_summary.csv

```csv
Mô hình (mean ± std trên 50 fold),PR-AUC,ROC-AUC,Gini,KS,F1,Recall,Brier
Logistic Regression (B: balanced),0.5236 ± 0.0150,0.7693 ± 0.0085,0.5386 ± 0.0169,0.4186 ± 0.0154,0.5284 ± 0.0110,0.6132 ± 0.0153,0.1870 ± 0.0024
Decision Tree (B: balanced),0.5350 ± 0.0139,0.7691 ± 0.0079,0.5381 ± 0.0159,0.4204 ± 0.0125,0.5164 ± 0.0087,0.6623 ± 0.0178,0.1871 ± 0.0026
Random Forest (B: balanced),0.5636 ± 0.0146,0.7859 ± 0.0073,0.5718 ± 0.0147,0.4390 ± 0.0129,0.5404 ± 0.0088,0.6309 ± 0.0137,0.1783 ± 0.0023
```


## ch4_shap_mean_abs.csv

```csv
,mean |SHAP|
NUM_MONTHS_DELAYED,0.07401
MAX_DELAY,0.05777
PAY_0,0.04812
AVG_PAY_AMT,0.02114
CREDIT_UTIL,0.01872
LIMIT_BAL,0.01598
AVG_BILL_AMT,0.01577
PAY_2,0.01462
PAY_AMT2,0.00933
BILL_AMT1,0.00917
PAY_RATIO,0.00908
PAY_AMT4,0.00636
PAY_0_revolving,0.00611
PAY_AMT1,0.00595
PAY_AMT3,0.00375
PAY_AMT6,0.00366
BILL_AMT5,0.00251
PAY_3,0.0025
BILL_AMT2,0.00233
BILL_AMT3,0.00179
AGE,0.0016
PAY_4,0.0012
PAY_5,0.00104
BILL_AMT4,0.00094
BILL_AMT6,0.00091
PAY_AMT5,0.00084
PAY_6,0.00078
EDUCATION,0.00065
MARRIAGE,0.00051
PAY_6_nouse,0.00037
SEX,0.00033
PAY_6_revolving,0.00031
PAY_0_nouse,0.0003
PAY_5_nouse,0.00026
PAY_3_revolving,0.00025
PAY_2_revolving,0.00021
PAY_5_revolving,0.00017
PAY_3_nouse,0.00016
PAY_4_revolving,0.00015
PAY_4_nouse,0.00015
PAY_2_nouse,0.00011
```


## ch4_significance_tests.csv

```csv
Độ đo,Cặp,Chênh lệch TB,t (corrected),p (corrected),Có ý nghĩa (p < 0.0167),t (ttest_rel),p (ttest_rel)
PR-AUC,Logistic Regression – Decision Tree,-0.011387887919099837,-1.5755742287656682,0.1215591008383999,Không,-5.789029368532532,4.938692079405536e-07
PR-AUC,Logistic Regression – Random Forest,-0.04002697322162385,-8.38059152407735,5.0323094216458254e-11,Có,-30.79225946502466,1.011951042580099e-33
PR-AUC,Decision Tree – Random Forest,-0.028639085302524014,-6.429775769618569,5.073352594872524e-08,Có,-23.62450469411475,1.9578607220103396e-28
F1,Logistic Regression – Decision Tree,0.012004849355033431,2.4795113421695585,0.01664357077537475,Có,9.110306399638327,4.0402727967679086e-12
F1,Logistic Regression – Random Forest,-0.012011633184965231,-3.4512837643484815,0.001158051340039814,Có,-12.68082627030858,4.3174337318302403e-17
F1,Decision Tree – Random Forest,-0.024016482539998663,-5.587395832388334,1.0047748650769429e-06,Có,-20.529403170457055,1.0247178060982361e-25
```


## ch4_test_bootstrap_ci.csv

```csv
Mô hình,Độ đo,Giá trị (test),CI 95% dưới,CI 95% trên
Logistic Regression,ROC-AUC,0.7624,0.7472,0.7779
Logistic Regression,PR-AUC,0.5275,0.4999,0.558
Decision Tree,ROC-AUC,0.7535,0.7387,0.7688
Decision Tree,PR-AUC,0.5272,0.4997,0.5551
Random Forest,ROC-AUC,0.7736,0.7593,0.7884
Random Forest,PR-AUC,0.5559,0.5279,0.5822
```


## ch4_test_metrics.csv

```csv
Mô hình,Cấu hình,Ngưỡng,Accuracy,Precision,Recall,F1,F2,ROC-AUC,PR-AUC,Gini,KS,Brier
Logistic Regression,B: balanced,0.5,0.7567,0.4611,0.5905,0.5179,0.5591,0.7624,0.5275,0.5247,0.403,0.1876
Decision Tree,B: balanced,0.5,0.7265,0.4199,0.6192,0.5005,0.5655,0.7535,0.5272,0.5069,0.3865,0.1916
Random Forest,B: balanced,0.5,0.7606,0.4687,0.6161,0.5324,0.5797,0.7736,0.5559,0.5471,0.4251,0.1815
```


## ch4b_rf_A_permutation_importance.csv

```csv
,RF A perm mean,RF A perm std,RF A hạng,RF B perm mean,RF B hạng
PAY_0,0.14804894732810492,0.00840510294963793,1.0,0.08470479506243546,1.0
NUM_MONTHS_DELAYED,0.02558473448220928,0.003990325424777602,2.0,0.055049298320806433,2.0
MAX_DELAY,0.018782012631662556,0.004594045240814513,3.0,0.03875732922885203,3.0
PAY_RATIO,0.013776509218780915,0.0037851004813446616,4.0,0.013718097018024641,4.0
CREDIT_UTIL,0.008263316824421333,0.0017205095741071347,5.0,0.01020930845613183,5.0
AVG_BILL_AMT,0.004940358896303521,0.0011608339668661859,6.0,0.0033868146413443823,10.0
BILL_AMT1,0.004048267062603283,0.0014703298525360453,7.0,0.003962094653389992,9.0
LIMIT_BAL,0.003970984297611876,0.000739455181484416,8.0,0.0027656345283344447,13.0
AVG_PAY_AMT,0.0032275633643958136,0.0011395980039285833,9.0,0.0029319699463248713,12.0
PAY_2,0.0028165981422279885,0.0013082700570952262,10.0,0.009938535508285114,6.0
PAY_6,0.0022391423354972168,0.0019337331374947799,11.0,0.0042569236058058475,8.0
PAY_0_revolving,0.0016320708462809455,0.0015657634881977428,12.0,0.0068412947826472555,7.0
PAY_3,0.0011421323160706986,0.0009351625994030266,13.0,0.0031837017634916715,11.0
AGE,0.0011293680422351645,0.0005916491332243471,14.0,0.0004269976819952448,24.0
PAY_AMT6,0.0008186035002577419,0.0013863516922352976,15.0,5.587766809656743e-05,34.0
BILL_AMT6,0.0007843637283552862,0.0005104465890537795,16.0,0.0011117471730084593,17.0
EDUCATION,0.0006808768544057838,0.0010538912619175622,17.0,0.00032420485518832215,25.0
BILL_AMT2,0.0004828517820321343,0.0006985025544912206,18.0,0.0022739168898947967,14.0
PAY_AMT2,0.0004386107389836839,0.000326826111484645,19.0,0.0002495741361722037,28.0
PAY_AMT3,0.0003637952193837779,0.001257936773527703,20.0,0.000995628728336606,18.0
BILL_AMT5,0.00035117111142211795,0.00021679033288095085,21.0,0.0005572813778527919,21.0
BILL_AMT3,0.0003478360605042215,0.0003696711388401483,22.0,0.0009751630230261746,19.0
PAY_AMT5,0.00027428296557227376,0.00040756433033649293,23.0,0.0004386284717325495,23.0
MARRIAGE,0.0002728201998562126,0.00018128767315176282,24.0,6.477779544934077e-05,33.0
BILL_AMT4,0.00024721656569332375,0.00019615584975700808,25.0,0.00047260127867712425,22.0
PAY_6_revolving,0.00016852455672041375,0.000515213015954525,26.0,0.0018991620196317023,16.0
PAY_5,9.09840424149951e-05,0.0009092433964755663,27.0,0.0021748633852878506,15.0
PAY_6_nouse,7.739816408602144e-05,2.8789153375803765e-05,28.0,0.00015753790496515855,29.0
PAY_2_revolving,6.927797934025056e-05,9.36199399935397e-05,29.0,-0.00012888227709671277,41.0
PAY_5_nouse,6.51579665551738e-05,2.8027275522219746e-05,30.0,8.752027120690157e-05,31.0
PAY_3_nouse,5.644722251760085e-05,1.0457098622370673e-05,31.0,4.3232344981628905e-05,35.0
PAY_4_nouse,4.313199176789739e-05,1.4963431415592914e-05,32.0,4.113269207628075e-05,36.0
PAY_0_nouse,3.931724418021106e-05,1.6780569942281103e-05,33.0,2.1234443113604674e-05,37.0
SEX,3.477882276186062e-05,0.0001209366628767512,34.0,0.00014184867091353952,30.0
PAY_2_nouse,7.01536531267181e-06,6.483876719145917e-06,35.0,2.4922449669636155e-06,38.0
BILL_TREND,0.0,0.0,36.0,0.0,39.0
PAY_4,-1.877423667769514e-05,0.00015411979926535698,37.0,0.0002834772107895141,27.0
PAY_5_revolving,-2.2112071852102103e-05,0.00023035289618741336,38.0,0.0005905843881600388,20.0
PAY_4_revolving,-4.0992422944981694e-05,0.0001236029331809003,39.0,0.00030311762319124393,26.0
PAY_3_revolving,-7.082102456965612e-05,7.660573306605739e-05,40.0,-4.453174996183288e-05,40.0
PAY_AMT1,-0.00010345146863923737,0.00039059164552679307,41.0,-0.00048144024601206594,42.0
PAY_AMT4,-0.00013246808744243888,0.00039174625347661253,42.0,8.497413874786937e-05,32.0
```


## ch4b_rf_A_shap_mean_abs.csv

```csv
,mean |SHAP|
PAY_0,0.06272608412681056
MAX_DELAY,0.035668891021323174
NUM_MONTHS_DELAYED,0.034882858051891036
AVG_PAY_AMT,0.014284546828311455
PAY_2,0.011720704226705934
AVG_BILL_AMT,0.010804482670302078
LIMIT_BAL,0.010577765518157011
CREDIT_UTIL,0.006990518235727966
PAY_RATIO,0.005562343300708281
BILL_AMT1,0.004229743472210889
PAY_AMT2,0.0033793916570696806
PAY_AMT1,0.003114678447227496
PAY_0_revolving,0.0029171280618182564
PAY_3,0.002671625775815937
PAY_AMT4,0.0022637265386475274
PAY_AMT3,0.0019172629968095523
PAY_AMT6,0.0016239777928929355
PAY_6,0.0015924049966357645
PAY_5,0.0014172914473428635
AGE,0.0013065233719165286
BILL_AMT5,0.0010495527636414054
BILL_AMT2,0.0009771292070873963
BILL_AMT3,0.0009105509473825369
PAY_4,0.0008138912586044241
EDUCATION,0.0007712625348884138
BILL_AMT6,0.0005784336550681403
PAY_AMT5,0.0004923582893534216
BILL_AMT4,0.0004899142831984352
PAY_6_revolving,0.0003880836301578323
SEX,0.0003009124051201351
MARRIAGE,0.0002999194154879991
PAY_4_revolving,0.00018296694705499927
PAY_2_revolving,0.00017963266526593966
PAY_0_nouse,0.00016552633369551825
PAY_5_revolving,0.00015726933623714398
PAY_3_revolving,0.00013887751246218708
PAY_4_nouse,0.00010777744221759122
PAY_6_nouse,9.57688624645027e-05
PAY_3_nouse,8.827878815206149e-05
PAY_5_nouse,6.562363173988406e-05
PAY_2_nouse,3.2905711834651334e-05
```


## ch4b_rf_A_vs_B_cv_summary.csv

```csv
Mô hình (mean ± std trên 50 fold),Precision,Recall,F1,F2,Chi phí/KH (5:1),PR-AUC,ROC-AUC,Brier,Log loss
"RF B (balanced, ngưỡng 0,5)",0.4728 ± 0.0096,0.6309 ± 0.0137,0.5404 ± 0.0088,0.5913 ± 0.0106,0.5641 ± 0.0142,0.5636 ± 0.0146,0.7859 ± 0.0073,0.1783 ± 0.0023,0.5430 ± 0.0060
RF A + ngưỡng chi phí,0.3862 ± 0.0207,0.7412 ± 0.0317,0.5068 ± 0.0122,0.6249 ± 0.0110,0.5490 ± 0.0137,0.5637 ± 0.0142,0.7853 ± 0.0073,0.1334 ± 0.0022,0.4261 ± 0.0056
```


## ch4b_rf_A_vs_B_cv_tests.csv

```csv
Độ đo,RF A + ngưỡng chi phí,RF B,Chênh lệch (A − B),t (corrected),p (corrected),t (ttest_rel)
F2,0.6249357729185511,0.5912560491544201,0.033679723764130975,5.47726815189088,1.4783592088251345e-06,20.124768234794526
Chi phí/KH (5:1),0.5489614668666039,0.564095541742089,-0.015134074875485126,-2.2145743445951496,0.03147120733111127,-8.136865712574899
Brier,0.13337496257006806,0.178296242428671,-0.04492127985860295,-51.40842150011029,2.7046526235706897e-44,-188.88660173579154
```


## ch4b_rf_A_vs_B_test.csv

```csv
Mô hình,Ngưỡng,Precision,Recall,F1,F2,Chi phí/KH (5:1),PR-AUC,ROC-AUC,Gini,KS,Brier,Log loss,P(default) trung bình
"RF B (balanced, ngưỡng 0,5)",0.5,0.46873207114170967,0.6161387631975868,0.5324209840338873,0.579679296154392,0.579175704989154,0.5559410088106023,0.7735647356798367,0.5471294713596735,0.4250620753979757,0.1815008415388566,0.5502590166505457,0.4259741782808562
RF A + ngưỡng chi phí,0.17,0.37933817594834546,0.7088989441930619,0.4942166140904311,0.6039578514520689,0.5786751209744703,0.5585660557738747,0.7733145111483634,0.5466290222967267,0.42389829297907294,0.1358940212683522,0.4331840152536803,0.2205622665117245
```


## ch4b_rf_A_vs_B_test_bootstrap_ci.csv

```csv
Mô hình,Độ đo,Giá trị (test),CI 95% dưới,CI 95% trên
"RF B (balanced, ngưỡng 0,5)",ROC-AUC,0.7735647356798367,0.7593159281577468,0.7884122532008497
"RF B (balanced, ngưỡng 0,5)",PR-AUC,0.5559410088106023,0.5278588922417362,0.5822099886805648
"RF B (balanced, ngưỡng 0,5)",Brier,0.1815008415388566,0.17691691325130354,0.1858760442471165
RF A + ngưỡng chi phí,ROC-AUC,0.7733145111483634,0.7586240586888399,0.787945528192116
RF A + ngưỡng chi phí,PR-AUC,0.5585660557738747,0.529944919582075,0.5847687655899081
RF A + ngưỡng chi phí,Brier,0.1358940212683522,0.12991074756204268,0.14170503890535202
```


## ch4b_risk_bands.csv

```csv
Phương án ngưỡng nhóm,Tập,Nhóm,Số KH,Tỷ trọng,Số vỡ nợ,Tỷ lệ vỡ nợ thực tế,P(default) TB dự đoán
"Cố định 0,10 / 0,30",OOF train,Thấp,7558,0.3152844985816786,519,0.06866896004233924,0.07788377917873876
"Cố định 0,10 / 0,30",OOF train,Trung bình,11236,0.4687134990822626,1922,0.1710573157707369,0.16809930296606393
"Cố định 0,10 / 0,30",OOF train,Cao,5178,0.21600200233605873,2863,0.5529161838547701,0.5463610215493637
"Cố định 0,10 / 0,30",Test,Thấp,1882,0.3140330385449691,143,0.07598299681190224,0.07846124347726004
"Cố định 0,10 / 0,30",Test,Trung bình,2819,0.470382112464542,484,0.17169208939340191,0.1671101634115962
"Cố định 0,10 / 0,30",Test,Cao,1292,0.2155848489904889,699,0.5410216718266254,0.5441811550489719
Phân vị OOF 0.14 / 0.33,OOF train,Thấp,11741,0.4897797430335391,1050,0.08943020185674133,0.09216121178837788
Phân vị OOF 0.14 / 0.33,OOF train,Trung bình,7458,0.3111129651259803,1532,0.20541700187717887,0.20421852241785124
Phân vị OOF 0.14 / 0.33,OOF train,Cao,4773,0.19910729184048057,2722,0.5702912214540121,0.5659627515069198
Phân vị OOF 0.14 / 0.33,Test,Thấp,2930,0.48890372100784246,283,0.09658703071672355,0.09229827045257906
Phân vị OOF 0.14 / 0.33,Test,Trung bình,1891,0.3155347905890205,390,0.20624008461131677,0.20409402955369005
Phân vị OOF 0.14 / 0.33,Test,Cao,1172,0.195561488403137,653,0.5571672354948806,0.5677934478606487
```


## ch4c_test_metrics.csv

```csv
,Accuracy,Precision,Recall,F1,F2,ROC-AUC,PR-AUC,TN,FP,FN,TP
"Logistic Regression (B: balanced, ngưỡng 0.50)",0.7567161688636743,0.46113074204946997,0.5904977375565611,0.5178571428571429,0.5591259640102828,0.762365228598733,0.5275331065977676,3752.0,915.0,543.0,783.0
"Decision Tree (B: balanced, ngưỡng 0.50)",0.7265142666444185,0.41994884910485936,0.6191553544494721,0.5004571776897287,0.5655048904807825,0.7534591097403838,0.527237949613,3533.0,1134.0,505.0,821.0
"Random Forest (B: balanced, ngưỡng 0.50)",0.7605539796429167,0.46873207114170967,0.6161387631975868,0.5324209840338873,0.579679296154392,0.7735647356798367,0.5559410088106023,3741.0,926.0,509.0,817.0
"Random Forest (A: none + ngưỡng chi phí, ngưỡng 0.17)",0.6789587852494577,0.37933817594834546,0.7088989441930619,0.4942166140904311,0.6039578514520689,0.7733145111483634,0.5585660557738747,3129.0,1538.0,386.0,940.0
```


## ch4c_train_oof_metrics.csv

```csv
,Accuracy,Precision,Recall,F1,F2,TN,FP,FN,TP
"Logistic Regression (B: balanced, ngưỡng 0.50)",0.7575504755548139,0.4637142857142857,0.6119909502262444,0.5276332899869961,0.575205557130706,14914.0,3754.0,2058.0,3246.0
"Decision Tree (B: balanced, ngưỡng 0.50)",0.7297263474053062,0.42786985880908535,0.657051282051282,0.5182541452896126,0.5934743367051531,14008.0,4660.0,1819.0,3485.0
"Random Forest (B: balanced, ngưỡng 0.50)",0.7613048556649424,0.4706213100927748,0.6312217194570136,0.5392172652601063,0.5908930462407342,14902.0,3766.0,1956.0,3348.0
"Random Forest (A: none + ngưỡng chi phí, ngưỡng 0.17)",0.6926831303187052,0.39466966200347187,0.7286953242835595,0.5120222560773664,0.6232061659518204,12740.0,5928.0,1439.0,3865.0
```


## ch4c_train_vs_test.csv

```csv
,Train (OOF),Train (OOF),Train (OOF),Train (OOF),Test,Test,Test,Test,Chênh lệch (test − train),Chênh lệch (test − train),Chênh lệch (test − train),Chênh lệch (test − train),Số mẫu,Số mẫu
,Accuracy,Precision,Recall,F1,Accuracy,Precision,Recall,F1,Accuracy,Precision,Recall,F1,Train,Test
"Logistic Regression (B: balanced, ngưỡng 0.50)",0.7575504755548139,0.4637142857142857,0.6119909502262444,0.5276332899869961,0.7567161688636743,0.46113074204946997,0.5904977375565611,0.5178571428571429,-0.0008343066911395969,-0.002583543664815724,-0.021493212669683293,-0.009776147129853197,23972,5993
"Decision Tree (B: balanced, ngưỡng 0.50)",0.7297263474053062,0.42786985880908535,0.657051282051282,0.5182541452896126,0.7265142666444185,0.41994884910485936,0.6191553544494721,0.5004571776897287,-0.0032120807608876367,-0.00792100970422599,-0.03789592760180993,-0.017796967599883806,23972,5993
"Random Forest (B: balanced, ngưỡng 0.50)",0.7613048556649424,0.4706213100927748,0.6312217194570136,0.5392172652601063,0.7605539796429167,0.46873207114170967,0.6161387631975868,0.5324209840338873,-0.000750876022025726,-0.0018892389510651308,-0.015082956259426794,-0.006796281226219003,23972,5993
"Random Forest (A: none + ngưỡng chi phí, ngưỡng 0.17)",0.6926831303187052,0.39466966200347187,0.7286953242835595,0.5120222560773664,0.6789587852494577,0.37933817594834546,0.7088989441930619,0.4942166140904311,-0.013724345069247468,-0.015331486055126409,-0.019796380090497667,-0.017805641986935294,23972,5993
```


# PHẦN C. METADATA MÔ HÌNH CUỐI (models/metadata.json)

```json
{
  "model_name": "Random Forest — RF A + ngưỡng chi phí",
  "imbalance_config": "A: none",
  "hyperparameters": {
    "bootstrap": true,
    "ccp_alpha": 0.0,
    "class_weight": null,
    "criterion": "gini",
    "max_depth": 6,
    "max_features": 0.5,
    "max_leaf_nodes": null,
    "max_samples": null,
    "min_impurity_decrease": 0.0,
    "min_samples_leaf": 21,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "monotonic_cst": null,
    "n_estimators": 200,
    "n_jobs": null,
    "oob_score": false,
    "random_state": 42,
    "verbose": 0,
    "warm_start": false
  },
  "threshold": 0.17,
  "threshold_rule": "min expected cost, CFN:CFP=5:1, chọn trên OOF train",
  "calibrated_probabilities": true,
  "risk_bands": {
    "edges": [
      0.1,
      0.3
    ],
    "labels": [
      "Thấp",
      "Trung bình",
      "Cao"
    ],
    "rule": "Thấp: p < edges[0]; Trung bình: edges[0] <= p < edges[1]; Cao: p >= edges[1]",
    "test_default_rate": {
      "Thấp": 0.076,
      "Trung bình": 0.1717,
      "Cao": 0.541
    }
  },
  "input_columns": [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "AVG_BILL_AMT",
    "AVG_PAY_AMT",
    "PAY_RATIO",
    "NUM_MONTHS_DELAYED",
    "MAX_DELAY",
    "CREDIT_UTIL",
    "PAY_0_nouse",
    "PAY_0_revolving",
    "PAY_2_nouse",
    "PAY_2_revolving",
    "PAY_3_nouse",
    "PAY_3_revolving",
    "PAY_4_nouse",
    "PAY_4_revolving",
    "PAY_5_nouse",
    "PAY_5_revolving",
    "PAY_6_nouse",
    "PAY_6_revolving",
    "BILL_TREND"
  ],
  "note": "Đầu vào là các cột sau Feature Engineering của Chương 3 cộng BILL_TREND (hàm add_bill_trend); đối tượng là FixedThresholdClassifier nên predict() đã áp dụng ngưỡng, predict_proba() trả xác suất vỡ nợ.",
  "cv_repeated_5x10_mean": {
    "Precision": 0.3862,
    "Recall": 0.7412,
    "F1": 0.5068,
    "F2": 0.6249,
    "Chi phí/KH (5:1)": 0.549,
    "PR-AUC": 0.5637,
    "ROC-AUC": 0.7853,
    "Brier": 0.1334,
    "Log loss": 0.4261
  },
  "test_metrics": {
    "Ngưỡng": 0.17,
    "Precision": 0.3793,
    "Recall": 0.7089,
    "F1": 0.4942,
    "F2": 0.604,
    "Chi phí/KH (5:1)": 0.5787,
    "PR-AUC": 0.5586,
    "ROC-AUC": 0.7733,
    "Gini": 0.5466,
    "KS": 0.4239,
    "Brier": 0.1359,
    "Log loss": 0.4332,
    "P(default) trung bình": 0.2206
  },
  "test_bootstrap_ci95": {
    "ROC-AUC": [
      0.7586,
      0.7879
    ],
    "PR-AUC": [
      0.5299,
      0.5848
    ],
    "Brier": [
      0.1299,
      0.1417
    ]
  },
  "random_state": 42,
  "sklearn_version": "1.9.1",
  "imblearn_version": "0.14.2"
}
```
