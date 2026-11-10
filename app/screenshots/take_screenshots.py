"""Chụp ảnh cho báo cáo. Chạy app trước: streamlit run app/app.py --server.port 8599 --server.headless true"""
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8599'
OUT = Path(__file__).resolve().parent
W = 1400


def settle(page, ms=1500):
    page.wait_for_timeout(300)
    page.wait_for_function("!document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=60_000)
    page.wait_for_timeout(ms)


def shot(page, name, full=True):
    if full:  # nội dung Streamlit cuộn trong stMain, nên nới viewport bằng chiều cao nội dung
        h = page.evaluate("document.querySelector('[data-testid=\"stMain\"]').scrollHeight")
        page.set_viewport_size({'width': W, 'height': max(900, h)})
        page.wait_for_timeout(800)
    page.screenshot(path=OUT / name)
    page.set_viewport_size({'width': W, 'height': 900})
    print('đã lưu', name)


def choose(page, label, option):
    page.locator('[data-testid="stSelectbox"]').filter(has_text=label).first.click()
    page.get_by_role('option', name=re.compile(option)).first.click()
    settle(page)


def predict(page):
    page.get_by_role('button', name=re.compile('Dự đoán$')).click()
    settle(page, 2500)


def open_app(browser):
    page = browser.new_page(viewport={'width': W, 'height': 900})
    page.goto(URL)
    page.wait_for_selector('text=Dự đoán một khách hàng')
    settle(page, 2500)
    return page


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = open_app(browser)
    shot(page, '01_trang_chu.png', full=False)
    choose(page, 'Chọn khách hàng mẫu', 'KH 19965')
    shot(page, '02_form_nhap_lieu.png')
    predict(page)
    shot(page, '03_ket_qua_KH_19965.png')

    page = open_app(browser)  # phiên mới để 20558 không bị so với 19965
    choose(page, 'Chọn khách hàng mẫu', 'KH 20558')
    predict(page)
    shot(page, '04_ket_qua_KH_20558.png')
    choose(page, 'Tháng 9', '^Trễ 2 tháng$')
    predict(page)
    page.get_by_role('button', name='So sánh với lần trước').click()
    settle(page)
    shot(page, '05_what_if_20558_PAY0_tre2thang.png')

    page.get_by_text('Dự đoán hàng loạt').first.click()
    settle(page)
    page.get_by_role('button', name=re.compile('Dùng 20 khách hàng mẫu')).click()
    settle(page, 2500)
    shot(page, '06_hang_loat_20_mau.png')

    page.get_by_text('Thông tin mô hình').first.click()
    settle(page, 2500)
    shot(page, '07_thong_tin_mo_hinh.png')
    slider = page.get_by_role('slider')
    slider.focus()
    for _ in range(13):  # 0,17 -> 0,30
        slider.press('ArrowRight')
    settle(page, 2500)
    shot(page, '08_thanh_truot_nguong_0.30.png')
    browser.close()
