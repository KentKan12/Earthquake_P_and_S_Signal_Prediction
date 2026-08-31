# 🌍 Automated Seismic Phase Picking & Earthquake Early Warning System (1D U-Net)

Repositori ini berisi implementasi _Deep Learning_ untuk deteksi otomatis waktu tiba gelombang gempa bumi (Gelombang P dan Gelombang S) menggunakan arsitektur **1D U-Net** pada dataset **Stanford Earthquake Dataset (STEAD)**.

---

## 📌 Alur Eksekusi Notebook (Google Colab)

Seluruh eksperimen dijalankan menggunakan runtime GPU di **Google Colab** dengan urutan tahapan sebagai berikut:

### 1. `01_Data_Extraction.ipynb` (Tahap Akuisisi & Reduksi Data)

- Mengunduh subset data Stanford STEAD dari repositori Kagglehub.
- Melakukan _stratified sampling_ sebanyak 100.000 sampel (50.000 gempa lokal dan 50.000 derau murni).
- Menyimpan subset gelombang ke dalam format portabel `metadata_subset.csv` dan `stead.hdf5` di Google Drive.

### 2. `02_Earthquake_P_and_S_Wave_Prediction_1D_UNet.ipynb` (Tahap Pemrosesan & Pemodelan)

- **Digital Signal Processing (DSP):** Penerapan _Detrending_, _Bandpass Filter_ Butterworth (1–45 Hz), dan _Z-score Normalization_ pada 3 kanal sensor (Z, N, E).
- **Target Generation:** Pembuatan label target berdistribusi _Gaussian_ ($\sigma = 20$ sampel / 0.2 detik) untuk fase P, fase S, dan derau (_noise_).
- **Event-Based Data Splitting:** Pembagian partisi data (70% Train, 15% Validation, 15% Test) berbasis `source_id` menggunakan `GroupShuffleSplit` guna mencegah _data leakage_.
- **Training 1D U-Net:** Arsitektur _Encoder-Decoder_ konvolusional 1D dilatih menggunakan fungsi rugi `BCEDiceLoss`, _optimizer_ Adam, serta penjadwal `ReduceLROnPlateau` dan _Early Stopping_.
- **Evaluasi & Post-Processing:** Inferensi pada data uji, algoritma _Peak Detection_ ($\text{threshold} = 0.5$), serta kalkulasi metrik F1-Score dan MAE.

---

## 📊 Hasil Evaluasi Model (Test Set)

Pengujian dilakukan secara mandiri pada 14.991 rekaman uji independen (_unseen event-based test set_):

| Fase Gelombang             | F1-Score (Akurasi Deteksi) | MAE (Deviasi Waktu Tiba)   |
| :------------------------- | :------------------------- | :------------------------- |
| **Gelombang Primer (P)**   | **0.9652**                 | **0.0356 detik (35.6 ms)** |
| **Gelombang Sekunder (S)** | **0.9487**                 | **0.0580 detik (58.0 ms)** |

---

## 🛠️ Instalasi Dependensi Lokal

Jika ingin menjalankan kode di lingkungan lokal:

```bash
git clone [https://github.com/](https://github.com/)<username-anda>/<nama-repo>.git
cd <nama-repo>
pip install -r requirements.txt

Catatan: File data stead.hdf5 tidak disertakan di repositori GitHub karena limitasi ukuran file. Jalankan notebook 01_Data_Extraction.ipynb untuk menghasilkan file HDF5 tersebut secara mandiri.

📚 Dataset & Referensi
Publikasi Ilmiah Orisinal:

Mousavi, S. M., Sheng, Y., Zhu, W., & Beroza, G. C. (2019). STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI. IEEE Access, 7, 179464–179476. doi:10.1109/ACCESS.2019.2947848

Repositori Resmi STEAD:

https://github.com/smousavi05/STEAD

Sumber Data Ekstraksi (Kaggle):

STEAD Chunk 1 (https://www.kaggle.com/datasets/julianachristamacam/steadchunk1)

STEAD Chunk 2 (https://www.kaggle.com/datasets/alextitu/stead-chunk-2)
```
