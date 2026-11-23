# CARE-IPS

نسخه بازسازمان‌دهی‌شده پروژه CARE-IPS بر اساس ساختار پیشنهادی و کدهای مخزن GitHub:
`elmirakarimpour/Care_IPS` (شاخه main).

## وضعیت داده پیوست‌شده
فایل `Global test set.csv` در مسیر `data/raw/CIC-IDS2017/` قرار گرفته است.
این فایل دارای 15,999 ردیف، 78 ستون و دو کلاس `BENIGN` و `FTP-Patator` است.
به دلیل نام و نقش مجموعه، در این بسته به‌عنوان **Global Test Set** نگهداری می‌شود.
این فایل برای fit کردن مدل، scaler، انتخاب ویژگی یا تنظیم آستانه استفاده نمی‌شود.

در حال حاضر مجموعه آموزش مستقل در بسته وجود ندارد؛ بنابراین اجرای آموزش و محاسبه معیارهای نهایی تا زمان افزودن داده آموزش انجام‌پذیر نیست.

## ساختار
- `configs/experiments/`: تنظیمات قابل ردیابی آزمایش‌ها
- `data/raw/CIC-IDS2017/`: داده خام، بدون تغییر
- `data/interim/`: داده‌های میانی
- `data/processed/`: داده‌های پاک‌سازی‌شده و split/viewها
- `src/care_ips/`: کد کتابخانه CARE-IPS
- `scripts/`: ورودی‌های خط فرمان
- `tests/`: آزمون‌های واحد و یکپارچه
- `docs/`: مستندات معماری، روش و بازتولیدپذیری
- `results/runs/`: خروجی مستقل هر اجرای آزمایش

## نصب
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

## اعتبارسنجی مجموعه آزمون
```bash
python scripts/data_preparation/inspect_global_test.py
```
این دستور گزارش کیفیت داده را در `results/summary/` می‌نویسد و داده اصلی را تغییر نمی‌دهد.

## افزودن داده آموزش
داده آموزش باید جداگانه در مسیر زیر قرار گیرد:
`data/raw/CIC-IDS2017/train/`
سپس pipeline آموزش باید فقط روی داده آموزش fit شود و مجموعه Global Test صرفاً transform/evaluate شود.
هر نوع split، feature selection، حذف ویژگی هم‌بسته و scaling باید با پارامترهای استخراج‌شده از train انجام شود.

## نکات علمی و محدودیت‌های فعلی
- بخش‌هایی از CACM/PSD/Agent/Evidence از مخزن اصلی حفظ شده‌اند.
- `consensus/dependency.py`، Safety Gate، reasoning، feedback و evaluation در مخزن اصلی TODO/NotImplemented بودند؛ در این بسته نیز به‌عنوان اجزای ناتمام علامت‌گذاری شده‌اند.
- چهار Feature View به‌صورت جداگانه تعریف شده‌اند، اما استقلال آماری/معماری عامل‌ها هنوز باید با آزمایش و تحلیل هم‌بستگی خطا اثبات شود.
- Confidence خروجی RandomForest فعلاً `predict_proba().max()` است و calibration مستقل ندارد.
- Faithfulness موجود یک آزمون اولیه است و باید احتمال کلاس اولیه را پس از perturbation مقایسه کند.
- هیچ نتیجه تجربی یا عدد عملکردی در این بسته جعل نشده است.

## اجرای تست‌ها
```bash
pytest tests/ -q
```
