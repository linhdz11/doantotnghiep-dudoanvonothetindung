"""Ứng dụng demo dự đoán vỡ nợ thẻ tín dụng — chạy: streamlit run app/app.py"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from features import (ROOT, METADATA, RAW_COLUMNS, THRESHOLD, BAND_EDGES, BAND_LABELS, PAY_STATUS_COLS,
                      BILL_COLS, PAY_AMT_COLS, build_features, risk_band, validate_raw, describe_customer)

st.set_page_config(page_title="Dự đoán vỡ nợ thẻ tín dụng", layout="wide", page_icon="💳")

# ---------------------------------------------------------------- nhãn & định dạng
MONTHS = ['tháng 9', 'tháng 8', 'tháng 7', 'tháng 6', 'tháng 5', 'tháng 4']  # PAY_0/BILL_AMT1 = tháng 9 ... tháng 4
SEX_LABELS = {1: 'Nam', 2: 'Nữ'}
EDU_LABELS = {1: 'Sau đại học', 2: 'Đại học', 3: 'Trung học', 4: 'Khác'}
MAR_LABELS = {1: 'Đã kết hôn', 2: 'Độc thân', 3: 'Khác'}
PAY_LABELS = {-2: 'Không phát sinh chi tiêu', -1: 'Trả đủ', 0: 'Trả tối thiểu (quay vòng)',
              **{k: f'Trễ {k} tháng' for k in range(1, 9)}}
BAND_COLORS = {'Thấp': '#2e7d32', 'Trung bình': '#f9a825', 'Cao': '#c62828'}
BAND_BG = {'Thấp': '#c8e6c9', 'Trung bình': '#fff59d', 'Cao': '#ffcdd2'}

FEATURE_VI = {
    'LIMIT_BAL': 'Hạn mức tín dụng', 'SEX': 'Giới tính', 'EDUCATION': 'Học vấn', 'MARRIAGE': 'Hôn nhân', 'AGE': 'Tuổi',
    **{c: f'Tình trạng trả nợ {m}' for c, m in zip(PAY_STATUS_COLS, MONTHS)},
    **{c: f'Sao kê {m}' for c, m in zip(BILL_COLS, MONTHS)},
    **{c: f'Số tiền đã trả {m}' for c, m in zip(PAY_AMT_COLS, MONTHS)},
    'AVG_BILL_AMT': 'Sao kê trung bình 6 tháng', 'AVG_PAY_AMT': 'Thanh toán trung bình 6 tháng',
    'PAY_RATIO': 'Tỷ lệ trả / sao kê', 'NUM_MONTHS_DELAYED': 'Số tháng trễ hạn (trong 6 tháng)',
    'MAX_DELAY': 'Mức trễ hạn lớn nhất', 'CREDIT_UTIL': 'Tỷ lệ sử dụng hạn mức',
    **{f'{c}_nouse': f'Không chi tiêu {m}' for c, m in zip(PAY_STATUS_COLS, MONTHS)},
    **{f'{c}_revolving': f'Trả tối thiểu {m}' for c, m in zip(PAY_STATUS_COLS, MONTHS)},
    'BILL_TREND': 'Xu hướng sao kê',
}

DEFAULTS = {'LIMIT_BAL': 140_000, 'SEX': 2, 'EDUCATION': 2, 'MARRIAGE': 2, 'AGE': 34, **{c: 0 for c in PAY_STATUS_COLS},
            **dict(zip(BILL_COLS, [22_000, 20_000, 20_000, 19_000, 18_000, 17_000])),
            **dict(zip(PAY_AMT_COLS, [2_000, 2_000, 2_000, 1_500, 1_500, 1_500]))}


def vn(x, d=0):
    """Số kiểu Việt Nam: 1.234.567,89"""
    return f"{x:,.{d}f}".replace(',', '_').replace('.', ',').replace('_', '.')


def pct(p, d=1):
    return f"{vn(p * 100, d)}%"


def readable(feat, v):
    if feat in PAY_STATUS_COLS:
        return PAY_LABELS.get(int(v), str(v))
    if feat == 'MAX_DELAY':
        return f"trễ {int(v)} tháng" if v > 0 else "không trễ hạn"
    if feat in ('SEX', 'EDUCATION', 'MARRIAGE'):
        return {'SEX': SEX_LABELS, 'EDUCATION': EDU_LABELS, 'MARRIAGE': MAR_LABELS}[feat][int(v)]
    if feat == 'AGE':
        return f"{int(v)} tuổi"
    if feat == 'NUM_MONTHS_DELAYED':
        return f"{int(v)} tháng"
    if feat in ('PAY_RATIO', 'CREDIT_UTIL'):
        return pct(v)
    if feat == 'BILL_TREND':
        return vn(v, 2)
    if feat.endswith(('_nouse', '_revolving')):
        return 'Có' if v else 'Không'
    return f"{vn(v)} NT$"


# ---------------------------------------------------------------- nạp tài nguyên
@st.cache_resource
def load_model():
    model = joblib.load(ROOT / 'models' / 'final_model.joblib')
    pipe = model.estimator_
    return model, pipe.named_steps['prep'], shap.TreeExplainer(pipe.named_steps['clf'])


@st.cache_data
def load_csv(name):
    return pd.read_csv(APP_DIR / 'data' / name)


model, prep, explainer = load_model()
samples = load_csv('sample_customers.csv').set_index('ID', drop=False)
test_pred = load_csv('test_predictions.csv')


def predict_df(raw):
    X = build_features(raw)
    return model.predict_proba(X)[:, 1]


def explain(raw_row):
    Xt = pd.DataFrame(prep.transform(build_features(raw_row)), columns=prep.get_feature_names_out())
    sv = explainer(Xt)[:, :, 1]
    return float(sv.base_values[0]), pd.Series(sv.values[0], index=Xt.columns), Xt.iloc[0]


# ---------------------------------------------------------------- sidebar
st.sidebar.title("💳 Dự đoán vỡ nợ")
st.sidebar.write("Ứng dụng ước lượng xác suất khách hàng vỡ nợ thẻ tín dụng ở kỳ tới "
                 "bằng Random Forest, kèm giải thích SHAP cho từng khách hàng.")
page = st.sidebar.radio("Trang", ["Dự đoán một khách hàng", "Dự đoán hàng loạt", "Thông tin mô hình"])
st.sidebar.caption("Minh hoạ cho đồ án — dữ liệu Đài Loan 2005, không dùng cho quyết định tín dụng thực tế")


# ================================================================ TRANG 1
def load_sample():
    sid = st.session_state.sample_id
    if sid is not None:
        for c in RAW_COLUMNS:
            st.session_state[c] = int(samples.loc[sid, c])


def reset_form():
    st.session_state.sample_id = None
    for c in RAW_COLUMNS:
        st.session_state[c] = DEFAULTS[c]


def page_single():
    for c in RAW_COLUMNS:
        st.session_state.setdefault(c, DEFAULTS[c])

    st.title("Dự đoán một khách hàng")
    top1, top2 = st.columns([5, 1], vertical_alignment='bottom')
    top1.selectbox(
        "Chọn khách hàng mẫu", [None] + list(samples.index), key='sample_id', on_change=load_sample,
        format_func=lambda i: "— Nhập tay —" if i is None else
        f"KH {i} – {samples.loc[i, 'Mô tả']} – Thực tế: {'Vỡ nợ' if samples.loc[i, 'Default'] else 'Không vỡ nợ'}")
    top2.button("Nhập mới", on_click=reset_form, width='stretch')

    left, right = st.columns([11, 9], gap='large')
    with left:
        with st.expander("1. Thông tin cá nhân", expanded=True):
            a, b = st.columns(2)
            c, d = st.columns(2)
            a.selectbox("Giới tính", list(SEX_LABELS), format_func=SEX_LABELS.get, key='SEX')
            b.selectbox("Học vấn", list(EDU_LABELS), format_func=EDU_LABELS.get, key='EDUCATION')
            c.selectbox("Hôn nhân", list(MAR_LABELS), format_func=MAR_LABELS.get, key='MARRIAGE')
            d.number_input("Tuổi", min_value=21, max_value=79, step=1, key='AGE')
        with st.expander("2. Hạn mức tín dụng", expanded=True):
            st.number_input("Hạn mức (NT$)", min_value=10_000, max_value=1_000_000, step=10_000, key='LIMIT_BAL')
            st.caption(f"= {vn(st.session_state.LIMIT_BAL)} NT$")
        with st.expander("3. Lịch sử trả nợ 6 tháng (tháng 9 → tháng 4)", expanded=True):
            cols = st.columns(2)
            for i, (c, m) in enumerate(zip(PAY_STATUS_COLS, MONTHS)):
                cols[i % 2].selectbox(m.capitalize(), list(PAY_LABELS), format_func=PAY_LABELS.get, key=c)
        with st.expander("4. Sao kê và thanh toán 6 tháng (NT$)", expanded=True):
            h = st.columns([1, 2, 2])
            h[1].markdown("**Sao kê (có thể âm)**")
            h[2].markdown("**Đã thanh toán (≥ 0)**")
            for m, bc, pc in zip(MONTHS, BILL_COLS, PAY_AMT_COLS):
                r = st.columns([1, 2, 2])
                r[0].markdown(f"<div style='padding-top:.5rem'>{m.capitalize()}</div>", unsafe_allow_html=True)
                r[1].number_input(f"Sao kê {m}", step=1_000, key=bc, label_visibility='collapsed')
                r[2].number_input(f"Thanh toán {m}", min_value=0, step=1_000, key=pc, label_visibility='collapsed')
                r[1].caption(f"{vn(st.session_state[bc])} NT$")
                r[2].caption(f"{vn(st.session_state[pc])} NT$")
        clicked = st.button("🔍 Dự đoán", type='primary', width='stretch')

    if clicked:
        raw = pd.DataFrame([{c: st.session_state[c] for c in RAW_COLUMNS}])
        errors = validate_raw(raw)
        if errors:
            with left:
                st.error("Dữ liệu chưa hợp lệ:\n\n" + "\n".join(f"- {e}" for e in errors))
        else:
            base, sv, xt = explain(raw)
            st.session_state.prev = st.session_state.get('result')
            st.session_state.result = {'raw': raw.iloc[0].to_dict(), 'proba': float(predict_df(raw)[0]),
                                       'base': base, 'sv': sv, 'xt': xt, 'sample_id': st.session_state.sample_id}
            st.session_state.show_compare = False

    with right:
        res = st.session_state.get('result')
        if res is None:
            st.info("Chọn một khách hàng mẫu hoặc nhập thông tin, rồi bấm **Dự đoán**.")
            return
        show_result(res, st.session_state.get('prev'))


def gauge(p):
    fig = go.Figure(go.Indicator(
        mode='gauge', value=p * 100,
        gauge={'axis': {'range': [0, 100], 'ticksuffix': '%', 'tickvals': [0, 10, 17, 30, 50, 75, 100]},
               'bar': {'color': '#263238', 'thickness': 0.3},
               'steps': [{'range': [0, 10], 'color': BAND_BG['Thấp']}, {'range': [10, 30], 'color': BAND_BG['Trung bình']},
                         {'range': [30, 100], 'color': BAND_BG['Cao']}],
               'threshold': {'line': {'color': 'black', 'width': 4}, 'thickness': 0.9, 'value': THRESHOLD * 100}}))
    fig.update_layout(height=230, margin=dict(l=30, r=30, t=20, b=0), separators=',.')
    return fig


def show_result(res, prev):
    p = res['proba']
    band = risk_band(p)
    edited = res['sample_id'] is not None and any(
        res['raw'][c] != samples.loc[res['sample_id'], c] for c in RAW_COLUMNS)
    who = "Nhập tay" if res['sample_id'] is None else f"KH {res['sample_id']}" + (" (đã chỉnh sửa)" if edited else "")
    st.subheader(f"Kết quả — {who}")

    c1, c2 = st.columns([1, 1])
    diff = None if prev is None else ('+' if p >= prev['proba'] else '') + vn((p - prev['proba']) * 100, 1)
    delta = None if prev is None else f"{diff} điểm %"
    c1.metric("Xác suất vỡ nợ", pct(p), delta=delta, delta_color='inverse',
              help=None if prev is None else "Chênh lệch so với lần dự đoán trước")
    c2.markdown(
        f"<div style='margin-top:.3rem'>Nhóm rủi ro<br><span style='background:{BAND_COLORS[band]};color:white;"
        f"padding:.3rem .9rem;border-radius:1rem;font-weight:700;font-size:1.2rem'>{band}</span></div>",
        unsafe_allow_html=True)
    st.plotly_chart(gauge(p), width='stretch', config={'displayModeBar': False})
    st.caption(f"Vạch đen: ngưỡng cảnh báo {vn(THRESHOLD, 2)} · Nền màu: Thấp < {vn(BAND_EDGES[0], 2)} ≤ "
               f"Trung bình < {vn(BAND_EDGES[1], 2)} ≤ Cao")
    if p >= THRESHOLD:
        st.error(f"⚠️ **CÓ NGUY CƠ VỠ NỢ** (xác suất ≥ ngưỡng {vn(THRESHOLD, 2)})")
    else:
        st.success(f"✅ **KHÔNG CÓ NGUY CƠ** (xác suất < ngưỡng {vn(THRESHOLD, 2)})")
    if res['sample_id'] is not None and not edited:
        actual = int(samples.loc[res['sample_id'], 'Default'])
        ok = actual == int(p >= THRESHOLD)
        mark = "<span style='color:#2e7d32'>✓ đúng</span>" if ok else "<span style='color:#c62828'>✗ sai</span>"
        st.markdown(f"**Thực tế: {'Vỡ nợ' if actual else 'Không vỡ nợ'}** — dự đoán {mark}", unsafe_allow_html=True)
    elif edited:
        st.caption("Đầu vào đã khác khách hàng mẫu nên không đối chiếu với nhãn thực tế.")

    if prev is not None and st.button("So sánh với lần trước"):
        st.session_state.show_compare = not st.session_state.get('show_compare', False)
    if prev is not None and st.session_state.get('show_compare'):
        changed = [c for c in RAW_COLUMNS if res['raw'][c] != prev['raw'][c]]
        st.markdown(f"**Lần trước:** {pct(prev['proba'])} ({risk_band(prev['proba'])}) → **Lần này:** {pct(p)} "
                    f"({band}); chênh lệch **{diff} điểm %**")
        if changed:
            st.table(pd.DataFrame({'Biến': [FEATURE_VI[c] for c in changed],
                                   'Lần trước': [readable(c, prev['raw'][c]) for c in changed],
                                   'Lần này': [readable(c, res['raw'][c]) for c in changed]}))
        else:
            st.caption("Không có biến đầu vào nào thay đổi.")

    shap_section(res)


def shap_section(res, top=8):
    st.markdown("#### Vì sao mô hình cho kết quả này? (SHAP)")
    sv, xt, base = res['sv'], res['xt'], res['base']
    order = sv.abs().sort_values(ascending=False).index
    top_f, rest = list(order[:top]), sv[order[top:]].sum()
    labels = [f"{FEATURE_VI[f]} = {readable(f, xt[f])}" for f in top_f][::-1]
    vals = (sv[top_f].values * 100)[::-1]
    fig = go.Figure(go.Waterfall(
        orientation='h', measure=['absolute'] + ['relative'] * (top + 1) + ['total'],
        y=['Mức nền (trung bình)', f'{len(order) - top} đặc trưng khác'] + labels + ['Xác suất dự đoán'],
        x=[base * 100, rest * 100, *vals, 0],
        text=[pct(base)] + [f"{'+' if v >= 0 else ''}{vn(v, 1)}" for v in [rest * 100, *vals]] + [pct(res['proba'])],
        textposition='outside', cliponaxis=False, increasing={'marker': {'color': '#c62828'}}, decreasing={'marker': {'color': '#1565c0'}},
        totals={'marker': {'color': '#455a64'}}, connector={'line': {'color': '#b0bec5'}}))
    running = np.cumsum([base * 100, rest * 100, *vals])
    fig.update_layout(height=430, margin=dict(l=10, r=60, t=10, b=30), separators=',.',
                      xaxis_range=[0, max(running.max(), res['proba'] * 100) * 1.18],
                      xaxis_title='Xác suất vỡ nợ (%)', showlegend=False)
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    st.caption(f"Đỏ: làm tăng, xanh: làm giảm xác suất (điểm phần trăm). Mức nền {pct(base, 2)} là xác suất "
               "trung bình của mô hình; cộng tất cả đóng góp ra đúng xác suất dự đoán.")
    for f in top_f[:3]:
        v = sv[f] * 100
        st.markdown(f"- **{FEATURE_VI[f]}** = {readable(f, xt[f])} làm **{'tăng' if v >= 0 else 'giảm'}** "
                    f"xác suất vỡ nợ {vn(abs(v), 1)} điểm phần trăm.")


# ================================================================ TRANG 2
def set_batch(df, source):
    st.session_state.batch = df
    st.session_state.batch_source = source


def on_upload():
    f = st.session_state.get('uploader')
    if f is None:
        return
    try:
        df = pd.read_csv(f)
    except Exception as e:  # file hỏng / không phải CSV
        st.session_state.batch_error = [f"Không đọc được file CSV: {e}"]
        return
    df.columns = df.columns.str.strip()
    errors = validate_raw(df)
    if 'Default' in df.columns and not df['Default'].isin([0, 1]).all():
        errors.append("Default: chỉ nhận 0 hoặc 1")
    st.session_state.batch_error = errors
    if not errors:
        set_batch(df, f.name)


def run_batch(df):
    out = pd.DataFrame({'Mã KH': df['ID'] if 'ID' in df.columns else np.arange(1, len(df) + 1)})
    out['Mô tả'] = df['Mô tả'] if 'Mô tả' in df.columns else df.apply(describe_customer, axis=1)
    out['proba'] = predict_df(df)
    out['Xác suất (%)'] = out['proba'] * 100
    out['Nhóm rủi ro'] = out['proba'].map(risk_band)
    out['pred'] = (out['proba'] >= THRESHOLD).astype(int)
    out['Dự đoán'] = out['pred'].map({1: 'Vỡ nợ', 0: 'Không vỡ nợ'})
    if 'Default' in df.columns:
        out['Default'] = df['Default'].astype(int).values
        out['Thực tế'] = out['Default'].map({1: 'Vỡ nợ', 0: 'Không vỡ nợ'})
        out['Đúng/Sai'] = np.where(out['pred'] == out['Default'], '✓', '✗')
    return out.sort_values('proba', ascending=False).reset_index(drop=True)


def page_batch():
    st.title("Dự đoán hàng loạt")
    c1, c2, c3 = st.columns([1, 2, 1], vertical_alignment='center')
    if c1.button("📋 Dùng 20 khách hàng mẫu", type='primary', width='stretch'):
        set_batch(samples.reset_index(drop=True), '20 mẫu')
        st.session_state.batch_error = []
    c2.file_uploader("Hoặc tải lên CSV (23 biến gốc; tuỳ chọn thêm ID, Default)", type='csv',
                     key='uploader', on_change=on_upload)
    template = samples[['ID'] + RAW_COLUMNS + ['Default']].head(2)
    c3.download_button("⬇️ Tải file CSV mẫu", template.to_csv(index=False).encode('utf-8-sig'),
                       'mau_nhap_lieu.csv', 'text/csv', width='stretch')

    for e in st.session_state.get('batch_error', []):
        st.error(e)
    df = st.session_state.get('batch')
    if df is None:
        st.info("Bấm **Dùng 20 khách hàng mẫu** hoặc tải lên file CSV.")
        return

    res = run_batch(df)
    export = res.drop(columns=['proba', 'pred', 'Default'], errors='ignore').round({'Xác suất (%)': 2})
    if st.session_state.batch_source == '20 mẫu':
        (APP_DIR / 'outputs').mkdir(exist_ok=True)
        export.to_csv(APP_DIR / 'outputs' / 'ket_qua_20_khach_hang.csv', index=False, encoding='utf-8-sig')
    st.caption(f"Nguồn dữ liệu: {st.session_state.batch_source} — {len(res)} khách hàng")

    has_y = 'Default' in res.columns
    m = st.columns([1, 1, 1, 1.2, 1.6])
    band_counts = res['Nhóm rủi ro'].value_counts().reindex(BAND_LABELS, fill_value=0)
    if has_y:
        tp = int(((res.pred == 1) & (res.Default == 1)).sum())
        m[0].metric("Dự đoán đúng", f"{int((res.pred == res.Default).sum())}/{len(res)}")
        m[1].metric("Vỡ nợ được phát hiện", f"{tp}/{int(res.Default.sum())}")
        m[2].metric("Cảnh báo nhầm", int(((res.pred == 1) & (res.Default == 0)).sum()))
    else:
        m[0].metric("Số cảnh báo", int(res.pred.sum()))
    m[3].markdown("**Phân bố nhóm rủi ro**<br>" + " · ".join(
        f"<span style='color:{BAND_COLORS[b]};font-weight:700'>{b}: {n}</span>" for b, n in band_counts.items()),
        unsafe_allow_html=True)
    if has_y:
        m[4].plotly_chart(cm_figure(res.Default, res.pred, 170), width='stretch', config={'displayModeBar': False})

    styler = (export.style
              .format({'Xác suất (%)': lambda v: vn(v, 1)})
              .map(lambda b: f"background-color:{BAND_BG[b]};color:{BAND_COLORS[b]};font-weight:600",
                   subset=['Nhóm rủi ro']))
    if has_y:
        styler = styler.map(lambda s: f"color:{'#2e7d32' if s == '✓' else '#c62828'};font-weight:700",
                            subset=['Đúng/Sai'])
    st.dataframe(styler, hide_index=True, width='stretch', height=min(38 + 35 * len(res), 740))
    st.download_button("⬇️ Tải kết quả CSV", export.to_csv(index=False).encode('utf-8-sig'),
                       'ket_qua_du_doan.csv', 'text/csv')


def cm_figure(y, pred, height):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                    x=['Không', 'Vỡ nợ'], y=['Không', 'Vỡ nợ'], labels=dict(x='Dự đoán', y='Thực tế'))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False, separators=',.')
    fig.update_traces(texttemplate='%{z:,d}', textfont_size=16)
    return fig


# ================================================================ TRANG 3
def page_model():
    st.title("Thông tin mô hình")
    tm, ci = METADATA['test_metrics'], METADATA['test_bootstrap_ci95']
    y, proba = test_pred['Default'].values, test_pred['proba'].values
    acc = float((test_pred['pred'] == y).mean())  # metadata không lưu Accuracy → tính từ test_predictions.csv

    st.markdown(f"#### Chỉ số trên tập test ({vn(len(test_pred))} khách hàng, ngưỡng {vn(THRESHOLD, 2)})")
    cols = st.columns(7)
    cards = [('Accuracy', vn(acc, 4), None), ('Precision', vn(tm['Precision'], 4), None),
             ('Recall', vn(tm['Recall'], 4), None), ('F1', vn(tm['F1'], 4), None),
             ('ROC-AUC', vn(tm['ROC-AUC'], 4), ci['ROC-AUC']), ('PR-AUC', vn(tm['PR-AUC'], 4), ci['PR-AUC']),
             ('Brier', vn(tm['Brier'], 4), ci['Brier'])]
    for col, (name, val, c) in zip(cols, cards):
        col.metric(name, val)
        if c:
            col.caption(f"CI 95%: [{vn(c[0], 4)}; {vn(c[1], 4)}]")

    hp = METADATA['hyperparameters']
    n_test = len(test_pred)
    st.markdown(f"""
#### Mô hình
- **Thuật toán:** Random Forest (scikit-learn {METADATA['sklearn_version']}), cấu hình **A** — không xử lý mất cân bằng
  (không class_weight, không SMOTE) nên xác suất đầu ra đã hiệu chỉnh tốt (Brier {vn(tm['Brier'], 4)}).
- **Siêu tham số:** n_estimators = {hp['n_estimators']}, max_depth = {hp['max_depth']},
  max_features = {vn(hp['max_features'], 1)}, min_samples_leaf = {hp['min_samples_leaf']}, random_state = {hp['random_state']}.
- **Ngưỡng cảnh báo {vn(THRESHOLD, 2)}:** chọn trên dự đoán OOF của tập train để cực tiểu chi phí kỳ vọng, với giả định
  bỏ sót một khách hàng vỡ nợ tốn gấp **5 lần** một cảnh báo nhầm (C_FN : C_FP = 5 : 1). Vì vậy ngưỡng thấp hơn nhiều so với 0,5:
  chấp nhận nhiều cảnh báo nhầm để bắt được ~71% ca vỡ nợ.
- **Dữ liệu:** UCI "Default of Credit Card Clients" (Đài Loan, 04–09/2005), 29.965 khách hàng sau khi bỏ trùng lặp;
  chia 80/20 phân tầng → {vn(29965 - n_test)} train / {vn(n_test)} test; tỷ lệ vỡ nợ ≈ 22,1%.
  Đầu vào là 23 biến gốc, ứng dụng tự tạo thêm 19 đặc trưng giống notebook (tổng 42 cột).
""")

    a, b = st.columns(2)
    with a:
        st.markdown("#### Nhóm rủi ro trên tập test")
        g = test_pred.groupby('band', sort=False)['Default'].agg(['size', 'sum', 'mean']).reindex(BAND_LABELS)
        rules = [f"p < {vn(BAND_EDGES[0], 2)}", f"{vn(BAND_EDGES[0], 2)} ≤ p < {vn(BAND_EDGES[1], 2)}", f"p ≥ {vn(BAND_EDGES[1], 2)}"]
        bt = pd.DataFrame({'Nhóm': BAND_LABELS, 'Khoảng xác suất': rules, 'Số KH': [vn(v) for v in g['size']],
                           'Tỷ trọng': [pct(v) for v in g['size'] / n_test], 'Số vỡ nợ': [vn(v) for v in g['sum']],
                           'Tỷ lệ vỡ nợ thực tế': [pct(v) for v in g['mean']]})
        st.dataframe(bt.style.map(lambda v: f"background-color:{BAND_BG[v]};font-weight:600", subset=['Nhóm']),
                     hide_index=True, width='stretch')
    with b:
        perm_path = ROOT / 'results' / 'ch4b_rf_A_permutation_importance.csv'
        if perm_path.exists():
            st.markdown("#### Top 10 yếu tố ảnh hưởng (permutation importance)")
            perm = pd.read_csv(perm_path, index_col=0).sort_values('RF A perm mean', ascending=False).head(10)[::-1]
            fig = go.Figure(go.Bar(x=perm['RF A perm mean'], y=[FEATURE_VI.get(i, i) for i in perm.index],
                                   orientation='h', marker_color='#1565c0',
                                   error_x=dict(type='data', array=perm['RF A perm std'], color='#90a4ae')))
            fig.update_layout(height=340, margin=dict(l=0, r=10, t=0, b=30), separators=',.',
                              xaxis_title='Mức giảm PR-AUC khi xáo trộn cột (test)')
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

    st.markdown("#### Thử thay đổi ngưỡng")
    st.warning(f"Chỉ để minh hoạ sự đánh đổi Precision – Recall. Ngưỡng vận hành của mô hình vẫn là **{vn(THRESHOLD, 2)}**.")
    t = st.slider("Ngưỡng", 0.05, 0.60, THRESHOLD, 0.01, format='%.2f', key='thr')
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    mc = st.columns(6)
    for col, (name, val) in zip(mc, [('Precision', vn(prec, 4)), ('Recall', vn(rec, 4)), ('F1', vn(f1, 4)),
                                     ('Accuracy', vn((tp + tn) / len(y), 4)), ('Bỏ sót (FN)', vn(fn)),
                                     ('Cảnh báo nhầm (FP)', vn(fp))]):
        col.metric(name, val)
    c1, c2 = st.columns([1, 2])
    c1.plotly_chart(cm_figure(y, pred, 320), width='stretch', config={'displayModeBar': False})
    pr, rc, _ = precision_recall_curve(y, proba)
    p0 = (proba >= THRESHOLD)
    fig = go.Figure(go.Scatter(x=rc, y=pr, mode='lines', name='Đường Precision–Recall', line=dict(color='#546e7a')))
    fig.add_scatter(x=[(p0 & (y == 1)).sum() / y.sum()], y=[(p0 & (y == 1)).sum() / p0.sum()], mode='markers',
                    name=f'Ngưỡng vận hành {vn(THRESHOLD, 2)}', marker=dict(size=12, color='black', symbol='x'))
    fig.add_scatter(x=[rec], y=[prec], mode='markers', name=f'Ngưỡng đang chọn {vn(t, 2)}',
                    marker=dict(size=14, color='#c62828'))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=30), xaxis_title='Recall', yaxis_title='Precision',
                      separators=',.', legend=dict(x=0.55, y=0.98))
    c2.plotly_chart(fig, width='stretch', config={'displayModeBar': False})


{"Dự đoán một khách hàng": page_single, "Dự đoán hàng loạt": page_batch, "Thông tin mô hình": page_model}[page]()
