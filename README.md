# 🌍 Automated Seismic Phase Picking & Earthquake Early Warning System (EEWS)

Repositori ini berisi implementasi *End-to-End Deep Learning* untuk sistem peringatan dini gempa bumi berbasis pengolahan sinyal seismik 3-komponen ($Z, N, E$) menggunakan **Stanford Earthquake Dataset (STEAD)**.  

Proyek ini mengintegrasikan dua model kecerdasan buatan:
1. **1D U-Net (Phase Picker):** Mendeteksi kedatangan Gelombang Primer ($P$) dan Gelombang Sekunder ($S$) secara *real-time*.
2. **1D CNN Regressor (Early Warning Forecaster):** Memprediksi perkiraan waktu tiba Gelombang $S$ ($T_s - T_p$) hanya dari cuplikan **3 detik pertama** fase $P$.

---

## 📌 Alur Eksekusi Notebook (Google Colab)

Seluruh alur eksperimen dijalankan menggunakan runtime GPU di **Google Colab** dengan urutan tahapan berikut:

### 1. `01_Data_Extraction.ipynb` — Tahap Akuisisi & Reduksi Data
- Mengunduh subset data Stanford STEAD via KaggleHub API.  
- Melakukan *stratified sampling* seimbang sebanyak 100.000 sampel (50.000 gempa lokal dan 50.000 derau/noise murni).  
- Mengekstrak matriks gelombang ke dalam format *portable* `metadata_subset.csv` dan `stead.hdf5` di Google Drive.  

### 2. `02_Earthquake_P_and_S_Wave_Prediction_1D_UNet.ipynb` — Tahap Deteksi Fase Gelombang
- **Digital Signal Processing (DSP):** Detrending, *Bandpass Filter* Butterworth (1–45 Hz), dan *Z-score Normalization* pada 3 kanal sensor ($Z, N, E$).  
- **Target Generation:** Label target Gaussian ($\sigma = 20$ sampel / 0.2 detik) untuk fase $P$, fase $S$, dan kanal derau (*noise*).  
- **Event-Based Data Splitting:** Partisi data (70% Train, 15% Validation, 15% Test) berbasis `source_id` dengan `GroupShuffleSplit` untuk mencegah *data leakage*.  
- **Training 1D U-Net:** Arsitektur *Encoder-Decoder with Skip Connections* dilatih dengan `BCEDiceLoss`, *optimizer* Adam, penjadwal `ReduceLROnPlateau`, serta *Early Stopping*.  
- **Evaluasi & Peak Detection:** Menentukan waktu kedatangan gelombang dengan threshold 0.5, evaluasi menggunakan F1-Score dan MAE.  

### 3. `03_S_Wave_Arrival_Time_Regression.ipynb` — Tahap Peramalan Waktu Tiba Fase S
- **3-Second Window Slicing:** Memotong matriks gelombang sepanjang 300 sampel (3 detik) sejak fase $P$ terdeteksi pada 49.945 data gempa valid.  
- **Training 1D CNN Regressor:** Arsitektur *Encoder-Only* dengan *Global Average Pooling* dan *Fully Connected Layer* untuk memprediksi selisih waktu ($T_s - T_p$).  
- **Loss Function & Optimasi:** Menggunakan `SmoothL1Loss` (Huber Loss) yang tangguh terhadap *outliers*.  

---

## 📊 Hasil Evaluasi Model (Independent Test Set)

### 1. Model Deteksi Fase (1D U-Net — 14.991 Rekaman Uji)
| Fase Gelombang | F1-Score | MAE |
| :--- | :--- | :--- |
| **Gelombang Primer ($P$)** | **0.9652** | **0.0356 detik (35.6 ms)** |
| **Gelombang Sekunder ($S$)** | **0.9487** | **0.0580 detik (58.0 ms)** |

### 2. Model Regresi Waktu Tiba Fase S (1D CNN — 3-Second Window)
| Metrik Evaluasi | Nilai Uji | Standar Industri EEWS |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **1.113 detik** | ≤ 1.0–1.5 detik (Optimal) |
| **Root Mean Squared Error (RMSE)** | **2.156 detik** | Terkendali (*No Catastrophic Outliers*) |

---

## 🗂️ Struktur Direktori Proyek

```text
seismic-phase-picking-1d-unet/
├── data/
│   ├── metadata_subset.csv
│   ├── stead.hdf5
│   ├── simulasi_gempa.npy
│   └── simulasi_noise.npy
├── models/
│   ├── unet_seismic.pt              # Bobot model 1D U-Net (Phase Picker)
│   └── best_regressor_seismic.pt    # Bobot model 1D CNN (S-Wave Regressor)
├── notebooks/
│   ├── 01_Data_Extraction.ipynb
│   ├── 02_Earthquake_P_and_S_Wave_Prediction_1D_UNet.ipynb
│   └── 03_S_Wave_Arrival_Time_Regression.ipynb
├── app/
│   ├── app.py                       # Logika Dashboard Streamlit
│   └── style.css                    # Styling UI (2D Flat Vector Soft Pastel)
├── requirements.txt
└── README.md
```

## Menjalankan Dashboard Simulasi (Lokal)
### 1. Clone repositori dan pasang dependensi
```
git clone https://github.com/<username-anda>/<nama-repo>.git
cd <nama-repo>
pip install -r requirements.txt
```
### 2. Jalankan aplikasi Streamlit
```
streamlit run app/app.py
```

### 3. Simulasi Aliran Sinyal (Streaming Simulation)
```
1. Unduh data uji simulasi_gempa.npy atau simulasi_noise.npy dari menu ekspander dashboard.
2. Unggah file tersebut ke aplikasi lalu klik tombol "Mulai Simulasi Streaming".
3. Dashboard akan memproses sinyal secara iteratif dengan sliding window 100 Hz, mengunci status siaga saat fase 𝑃
terdeteksi, menampilkan estimasi waktu kemunculan fase 𝑆, dan mengonfirmasi kedatangan fisik fase 𝑆.
```

##  Dataset & Referensi
Publikasi Ilmiah STEAD:  
Mousavi, S. M., Sheng, Y., Zhu, W., & Beroza, G. C. (2019). STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI. IEEE Access, 7, 179464–179476. doi:10.1109/ACCESS.2019.2947848

Repositori Resmi STEAD: smousavi05/STEAD

STEAD Chunk 1 (https://www.kaggle.com/datasets/julianachristamacam/steadchunk1)

STEAD Chunk 2 (https://www.kaggle.com/datasets/alextitu/stead-chunk-2)
