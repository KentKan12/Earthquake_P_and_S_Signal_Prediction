# DATA INGESTION & EXPLORATORY DAYA ANALYSIS

**Inisialisasi & Ekstraksi Dataset STEAD via KaggleHub**

Pada bab ini, kita akan mengunduh dan menyiapkan dataset seismik **STEAD** (*Stanford Earthquake Dataset*) menggunakan pustaka **kagglehub**.

Karena dataset asli berukuran puluhan gigabyte (~28 GB), kita membangun pipeline untuk:
1. Mengunduh data mentah ke *cache* Google Colab secara otomatis.
2. Menyaring **100.000 rekaman seimbang (50:50)** antara gempa dan *noise*.
3. Mengekstrak matriks getaran terpilih ke dalam file ** HDF5 ** (~7 GB) tanpa menurunkan kualitas sinyal.
4. Menyimpan dataset yang telah diproses ke **Google Drive**

---

### **Tujuan Utama**

1. **Unduh Cepat & Otomatis via API Cloud:**  
   Mengambil data mentah `Chunk 1` (Noise) dan `Chunk 2` (Gempa Bumi) langsung dari server Kaggle ke Google Colab tanpa membebani kuota internet lokal.

2. **Penyaringan Data Seimbang (*Stratified Sampling*):**  
   Menyeleksi tepat 100.000 sampel data dengan rasio seimbang 50:50:  
   * **50.000 rekaman gempa bumi lokal** yang memiliki label waktu kedatangan gelombang $P$ (*Primary*) dan $S$ (*Secondary*) yang valid dan lengkap.  
   * **50.000 rekaman kebisingan lingkungan (*ambient noise*)** tanpa sinyal gempa untuk melatih model mengenali sinyal latar belakang.

3. **Pembuatan Dataset Portabel (*Mini HDF5 Extraction*):**  
   Mengekstrak 100.000 rekaman sinyal yang dipilih dari HDF5 raksasa ke dalam satu file baru (`stead_100k.hdf5`) agar ukuran data lebih ringkas, hemat memori, dan mudah dipindahkan.

4. **Penyimpanan Permanen ke Google Drive:**  
   Menyimpan data  (`metadata_100k_ready.csv` dan `stead_100k.hdf5`) ke Google Drive

## Setup Environment & Download via KaggleHub

Kita akan menginstal library yang diperlukan untuk mengunduh dataset. Karena file diunduh ke dalam folder cache sementara, kita akan menggunakan library glob untuk mencari lokasi pasti dari file .csv dan .hdf5 di dalam folder tersebut.


```python
!pip install -q kagglehub h5py
```


```python
# import library
import kagglehub
import glob
import os

print("Download Chunk 1 (Noise)")
path_chunk1 = kagglehub.dataset_download("julianachristamacam/steadchunk1")

print("Download Chunk 2 (Gempa)")
path_chunk2 = kagglehub.dataset_download("alextitu/stead-chunk-2")

print("\nLokasi Direktori Cache")
print(f"Path Chunk 1: {path_chunk1}")
print(f"Path Chunk 2: {path_chunk2}")

def find_files(base_path) -> tuple[str, str]:
    # Mencari file .csv dan .hdf5 di dalam direktori unduhan Kagglehub
    csv_files = glob.glob(os.path.join(base_path, "**", "*.csv"), recursive=True)
    hdf5_files = glob.glob(os.path.join(base_path, "**", "*.hdf5"), recursive=True)

    return csv_files[0], hdf5_files[0]

#  path untuk masing-masing file
csv_noise, hdf5_noise = find_files(path_chunk1)
csv_eq, hdf5_eq = find_files(path_chunk2)

print(f"CSV Noise  : {csv_noise}")
print(f"HDF5 Noise : {hdf5_noise}")
print(f"CSV Gempa  : {csv_eq}")
print(f"HDF5 Gempa : {hdf5_eq}")
```

    Mengunduh Chunk 1 (Noise) dari Kaggle...
    Downloading to /root/.cache/kagglehub/datasets/julianachristamacam/steadchunk1/1.archive...
    

    100%|██████████| 13.6G/13.6G [02:47<00:00, 87.2MB/s]

    Extracting files...
    

    
    

    Mengunduh Chunk 2 (Gempa) dari Kaggle...
    Downloading to /root/.cache/kagglehub/datasets/alextitu/stead-chunk-2/1.archive...
    

    100%|██████████| 12.7G/12.7G [02:30<00:00, 91.0MB/s]

    Extracting files...
    

    
    

    
    --- Lokasi Direktori Cache ---
    Path Chunk 1: /root/.cache/kagglehub/datasets/julianachristamacam/steadchunk1/versions/1
    Path Chunk 2: /root/.cache/kagglehub/datasets/alextitu/stead-chunk-2/versions/1
    
    --- Lokasi File Spesifik ---
    CSV Noise  : /root/.cache/kagglehub/datasets/julianachristamacam/steadchunk1/versions/1/chunk1.csv
    HDF5 Noise : /root/.cache/kagglehub/datasets/julianachristamacam/steadchunk1/versions/1/chunk1.hdf5
    CSV Gempa  : /root/.cache/kagglehub/datasets/alextitu/stead-chunk-2/versions/1/chunk2.csv
    HDF5 Gempa : /root/.cache/kagglehub/datasets/alextitu/stead-chunk-2/versions/1/chunk2.hdf5
    

## Stratified Sampling 100k Data
Setelah mengetahui lokasi file, kita akan memuat kedua tabel CSV menggunakan Pandas. Kita akan ambil  sampel acak 50.000 data per kelas, sehingga total data adalah 100.000


```python
import pandas as pd
import numpy as np

# Set seed untuk reproduksibilitas
np.random.seed(42)

# Load data mentah
df_noise_raw = pd.read_csv(csv_noise, low_memory=False)
df_eq_raw = pd.read_csv(csv_eq, low_memory=False)

def prepare_subset(df_noise, df_eq, path_h5_noise, path_h5_eq, n_samples):
    """
    - df_noise: DataFrame noise
    - df_eq: DataFrame gempa
    - path_h5_noise: path file HDF5 noise
    - path_h5_eq: path file HDF5 gempa
    - n_samples: jumlah sampel per kelas
    """

    # Filter data gempa yang valid
    filter_eq = (
        (df_eq['trace_category'] == 'earthquake_local') &
        (df_eq['p_arrival_sample'].notnull()) &
        (df_eq['s_arrival_sample'].notnull())
    )
    df_eq = df_eq[filter_eq]

    # Sampling acak
    sampled_noise = df_noise.sample(n=n_samples, random_state=42).copy()
    sampled_eq = df_eq.sample(n=n_samples, random_state=42).copy()

    # Tambah informasi file sumber
    sampled_noise['source_file'] = path_h5_noise
    sampled_eq['source_file'] = path_h5_eq

    # Gabungkan dan acak urutan
    df_subset = pd.concat([sampled_noise, sampled_eq])
    df_subset = df_subset.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f" Total data: {len(df_subset)} baris.")
    return df_subset


metadata_subset = prepare_subset(df_noise_raw, df_eq_raw, hdf5_noise, hdf5_eq, 50000)

```

    Membaca tabel CSV ke dalam Pandas DataFrame...
    Mengambil 50000 sampel acak dari masing-masing kelas...
    Sampling selesai! Total data latih: 100000 baris.
    


```python
metadata_subset.info()
```

**Selanjutnya kita akan memproses/filter data hdf5 yang ada di metadata subset saja, sehingga proses pemrosesan data lebih efisien dan tidak mengambil banyak memori**


```python
import h5py
import matplotlib.pyplot as plt
import numpy as np

def get_waveform(trace_name, hdf5_path):
    """
    Mengekstrak array getaran langsung dari file HDF5.
    """

    with h5py.File(hdf5_path, 'r') as f:
        print(f"data untuk trace: {trace_name}")
        data_array = np.array(f.get(f"data/{trace_name}"))

    print(f"Waveform  diekstrak dengan shape: {data_array.shape}")
    return data_array.T

def plot_seismic_signal(metadata_row, path_hdf5_baru):
    """
    Plotting gelombang 3 kanal dengan anotasi titik tiba P dan S.
    Menggunakan data dari path HDF5 hasil proses.
    """
    trace_name = metadata_row['trace_name']
    category = metadata_row['trace_category']

    print("\n--- Plotting Trace ---")
    print(f"Trace Name : {trace_name}")
    print(f"File Path  : {path_hdf5_baru}")
    print(f"Kategori   : {category}")

    # Ambil waveform dari file HDF5 hasil proses
    waveform = get_waveform(trace_name, path_hdf5_baru)

    channels = ['Vertical (Z)', 'North-South (N)', 'East-West (E)']
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    title = f"Waveform: {trace_name} | {category.upper()}"
    if category == 'earthquake_local':
        title += f" | Mag: {metadata_row.get('source_magnitude', 'N/A')}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Buat sumbu waktu 60 detik
    time_axis = np.linspace(0, 60, waveform.shape[1])

    for i in range(3):
        axs[i].plot(time_axis, waveform[i], color='black', linewidth=0.5)
        axs[i].set_ylabel(channels[i], fontsize=10)
        axs[i].grid(True, linestyle='--', alpha=0.5)

        # Tambahkan penanda fase P dan S jika kategori gempa
        if category == 'earthquake_local':
            p_sec = metadata_row['p_arrival_sample'] / 100.0
            s_sec = metadata_row['s_arrival_sample'] / 100.0
            axs[i].axvline(x=p_sec, color='red', linestyle='-', linewidth=2, label='P-wave' if i==0 else "")
            axs[i].axvline(x=s_sec, color='blue', linestyle='-', linewidth=2, label='S-wave' if i==0 else "")

    axs[2].set_xlabel('Waktu (Detik)', fontsize=12)
    if category == 'earthquake_local':
        fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))

    plt.tight_layout()
    plt.show()

# Tentukan alamat file HDF5 hasil proses
lokasi_hdf5_proses = "data/processed/stead_100k.hdf5"

# Plot sampel gempa
print("Plot Sampel Gempa:")
plot_seismic_signal(
    metadata_subset[metadata_subset['trace_category'] == 'earthquake_local'].iloc[0],
    path_hdf5_baru=lokasi_hdf5_proses
)

# Plot sampel noise
print("Plot Sampel Noise:")
plot_seismic_signal(
    metadata_subset[metadata_subset['trace_category'] == 'noise'].iloc[0],
    path_hdf5_baru=lokasi_hdf5_proses
)

```

## Save Dataset
Simpan dataset df_subset dan hdf5 di dalam folder MyDrive


```python
from google.colab import drive
import shutil
import os

def save_to_google_drive(local_folder, drive_folder):
    drive.mount('/content/drive')

    # Buat folder tujuan
    os.makedirs(drive_folder, exist_ok=True)

    # Tentukan path (alamat) file yang ada di Colab saat ini
    file_hdf5_lokal = os.path.join(local_folder, "stead.hdf5")
    file_csv_lokal = os.path.join(local_folder, "metadata_subset.csv")

    # path file tujuan di Google Drive
    file_hdf5_drive = os.path.join(drive_folder, "stead.hdf5")
    file_csv_drive = os.path.join(drive_folder, "metadata_subset.csv")

    print("Copy file CSV...")
    shutil.copy(file_csv_lokal, file_csv_drive)

    print("Copy HDF5 file...")
    shutil.copy(file_hdf5_lokal, file_hdf5_drive)

    print("\nDONE")

save_to_google_drive("data/processed", "/content/drive/MyDrive")
```
