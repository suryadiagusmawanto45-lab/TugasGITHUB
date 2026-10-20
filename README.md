# Studi Kasus 1: Make Sense of Data with Analytics and AI

## Latar belakang
Sebuah perusahaan layanan digital di Sumatera ingin menekan jumlah pelanggan yang berhenti
berlangganan (*churn*). Data pelanggan perlu dipahami lalu dipakai untuk memprediksi siapa yang berisiko.

## Empat tingkat analitik pada kode
| Tingkat | Pertanyaan | Implementasi |
|---|---|---|
| Descriptive | Apa yang terjadi? | Churn rate keseluruhan dan per kota |
| Diagnostic | Mengapa terjadi? | Korelasi fitur dengan churn |
| Predictive | Apa yang akan terjadi? | Random Forest + ROC-AUC |
| Prescriptive | Apa yang harus dilakukan? | Daftar 20 pelanggan risiko tertinggi untuk program retensi |

## Menjalankan
`python main.py` -> `output/analytics.png`, `customers.csv`, `top20_risiko_churn.csv`

## Bahan diskusi
- Churn hanya ~20%, sehingga akurasi saja menyesatkan. Perhatikan **recall** kelas churn dan efek `class_weight="balanced"`.
- Coba ubah threshold probabilitas (default 0.5) dan lihat trade-off precision vs recall.
- Pengembangan: tambahkan SHAP untuk penjelasan model, atau uji XGBoost.
