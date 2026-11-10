# Lưu mô hình cuối (pipeline + ngưỡng) và metadata
import json
import joblib
import sklearn
from pathlib import Path

MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(final_model, MODEL_DIR / 'final_model.joblib')

metadata = {
    'model_name': 'Random Forest — RF A + ngưỡng chi phí',
    'threshold': 0.17,
    'threshold_rule': 'min expected cost, CFN:CFP=5:1, chọn trên OOF train',
    'calibrated_probabilities': True,
    'risk_bands': {'edges': [0.10, 0.30], 'labels': ['Thấp', 'Trung bình', 'Cao']},
    'input_columns': list(X_train4.columns),
    'sklearn_version': sklearn.__version__,
}
(MODEL_DIR / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                         encoding='utf-8')

# Kiểm tra: nạp lại và dự đoán giống hệt
reloaded = joblib.load(MODEL_DIR / 'final_model.joblib')
assert (reloaded.predict(X_test4) == final_model.predict(X_test4)).all()
