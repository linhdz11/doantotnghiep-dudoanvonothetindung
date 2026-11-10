# Ứng dụng demo: Dự đoán vỡ nợ thẻ tín dụng

Ứng dụng Streamlit minh hoạ mô hình cuối của đồ án (`models/final_model.joblib` — Random Forest cấu hình A,
ngưỡng 0,17). Minh hoạ cho đồ án, dữ liệu Đài Loan 2005, **không dùng cho quyết định tín dụng thực tế**.

## Cài đặt và chạy

```bash
pip install -r app/requirements.txt     # Python 3.12
streamlit run app/app.py                # mở http://localhost:8501
```

Chạy từ thư mục gốc của dự án (app đọc `models/` và `results/` ở thư mục cha của `app/`).

Các lệnh phụ:

```bash
python app/prepare_data.py              # tạo lại app/data/ (tải UCI id=350, kiểm tra khớp notebook)
python -m pytest app/tests -q           # kiểm thử
# ảnh chụp báo cáo (cần: pip install playwright && playwright install chromium)
streamlit run app/app.py --server.port 8599 --server.headless true &
python app/screenshots/take_screenshots.py
```

## Cấu trúc thư mục

```
app/
├── app.py                 # giao diện 3 trang: một khách hàng / hàng loạt / thông tin mô hình
├── features.py            # tiền xử lý giống hệt notebook: build_features(), risk_band(), validate_raw()
├── prepare_data.py        # chạy một lần: tái tạo tập test, kiểm tra khớp số, chọn 20 khách hàng mẫu
├── requirements.txt
├── data/
│   ├── test_raw.csv           # 5.993 dòng: ID + 23 biến gốc + Default
│   ├── test_predictions.csv   # + proba, pred, band
│   └── sample_customers.csv   # 20 khách hàng mẫu + Mô tả, Loại ca, Lý do chọn
├── outputs/ket_qua_20_khach_hang.csv   # ghi khi bấm "Dùng 20 khách hàng mẫu"
├── screenshots/           # ảnh cho báo cáo + take_screenshots.py
└── tests/test_app.py
```

Ứng dụng dùng thêm `../models/final_model.joblib`, `../models/metadata.json` và (nếu có)
`../results/ch4b_rf_A_permutation_importance.csv`.

## Deploy lên Streamlit Community Cloud

1. Đưa dự án lên một repo GitHub, gồm tối thiểu `app/` (kể cả `app/data/`), `models/final_model.joblib`,
   `models/metadata.json`, `results/ch4b_rf_A_permutation_importance.csv` (mô hình ~1,6 MB, không cần Git LFS).
2. Vào <https://share.streamlit.io> → **Create app** → chọn repo, nhánh, **Main file path** = `app/app.py`.
3. **Advanced settings** → Python **3.12**. Cloud tự đọc `app/requirements.txt` (cùng thư mục với file chính).
4. **Deploy**. Phiên bản scikit-learn/imbalanced-learn phải đúng như đã ghim, nếu không joblib có thể không nạp được mô hình.
