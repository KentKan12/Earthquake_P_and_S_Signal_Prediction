import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import scipy.signal as signal
import time
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. KONFIGURASI HALAMAN & STYLING
# ==========================================
st.set_page_config(page_title="Seismic EEWS Dashboard", layout="wide")

def load_css(file_name):
    css_path = os.path.join(os.path.dirname(__file__), file_name)
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ==========================================
# 2. DEFINISI ARSITEKTUR MODEL
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# Arsitektur Segmentasi U-Net (Deteksi Aktual)
class UNet1D(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet1D, self).__init__()
        self.down1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.down2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.down3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(64, 128)
        self.upconv3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.up3 = DoubleConv(128, 64)
        self.upconv2 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.up2 = DoubleConv(64, 32)
        self.upconv1 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)
        self.up1 = DoubleConv(32, 16)
        self.out_conv = nn.Conv1d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        p1 = self.pool1(x1)
        x2 = self.down2(p1)
        p2 = self.pool2(x2)
        x3 = self.down3(p2)
        p3 = self.pool3(x3)
        bot = self.bottleneck(p3)
        up3 = self.upconv3(bot)
        merge3 = torch.cat([x3, up3], dim=1)
        out3 = self.up3(merge3)
        up2 = self.upconv2(out3)
        merge2 = torch.cat([x2, up2], dim=1)
        out2 = self.up2(merge2)
        up1 = self.upconv1(out2)
        merge1 = torch.cat([x1, up1], dim=1)
        out1 = self.up1(merge1)
        logits = self.out_conv(out1)
        return logits

# Arsitektur Regresi CNN (Prediksi Countdown)
class S_Wave_Regressor_CNN(nn.Module):
    def __init__(self, in_channels=3):
        super(S_Wave_Regressor_CNN, self).__init__()
        self.down1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.down2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.down3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.feature_extractor = DoubleConv(64, 128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1) 

    def forward(self, x):
        x = self.pool1(self.down1(x))
        x = self.pool2(self.down2(x))
        x = self.pool3(self.down3(x))
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1) 
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def signal_to_waveform(waveform_matrix):
    detrended_waveform = signal.detrend(waveform_matrix, axis=1)
    sampling_rate = 100.0
    nyquist_freq = sampling_rate / 2.0
    low_cut = 1.0 / nyquist_freq
    high_cut = 45.0 / nyquist_freq
    b, a = signal.butter(4, [low_cut, high_cut], btype='band')
    filtered_waveform = signal.filtfilt(b, a, detrended_waveform, axis=1)
    mean_waveform = np.mean(filtered_waveform, axis=1, keepdims=True)
    std_waveform = np.std(filtered_waveform, axis=1, keepdims=True)
    std_waveform[std_waveform == 0] = 1e-8
    normalized_waveform = (filtered_waveform - mean_waveform) / std_waveform
    return normalized_waveform

# ==========================================
# 3. FUNGSI CACHE & LOAD MODEL
# ==========================================
@st.cache_resource
def load_models():
    device = torch.device("cpu")
    
    # Load U-Net
    model_unet = UNet1D(in_channels=3, out_channels=3)
    unet_path = os.path.join(os.path.dirname(__file__), "..", "models", "unet_seismic.pt")
    if os.path.exists(unet_path):
        model_unet.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
    else:
        st.warning(f"File model U-Net tidak ditemukan di {unet_path}.")
    model_unet.to(device).eval()

    # Load Regressor
    model_reg = S_Wave_Regressor_CNN(in_channels=3)
    reg_path = os.path.join(os.path.dirname(__file__), "..", "models", "regressor_seismic.pt")
    if os.path.exists(reg_path):
        model_reg.load_state_dict(torch.load(reg_path, map_location=device, weights_only=True))
    else:
        st.warning(f"File model Regressor tidak ditemukan di {reg_path}.")
    model_reg.to(device).eval()

    return model_unet, model_reg, device

model_unet, model_reg, device = load_models()

# ==========================================
# 4. ANTARMUKA EDUKASI & SETUP DATA
# ==========================================
st.title("Automated Seismic Phase Picking & EEWS")
st.markdown("Dashboard interaktif untuk mensimulasikan peringatan dini gempa bumi menggunakan arsitektur **1D U-Net** berbasis analisis sinyal 3-kanal (Z, N, E).")

with st.expander("Informasi Dataset & Unduh Data Simulasi", expanded=True):
    st.markdown("""
    Sistem ini memproses sinyal getaran fisik (*waveform*) berdurasi 60 detik yang terekam pada kecepatan 100 sampel per detik. 
    Anda dapat menguji sistem dengan rekaman gempa asli atau derau (*noise*) dari **Stanford STEAD Dataset**.
    """)
    
    col1, col2, col3 = st.columns([1.5, 1.5, 7])
    gempa_path = os.path.join(os.path.dirname(__file__), "..", "data", "simulasi_gempa.npy")
    noise_path = os.path.join(os.path.dirname(__file__), "..", "data", "simulasi_noise.npy")
    
    with col1:
        if os.path.exists(gempa_path):
            with open(gempa_path, "rb") as file:
                st.download_button(label="Unduh Gempa (.npy)", data=file, file_name="simulasi_gempa.npy", mime="application/octet-stream")
    with col2:
        if os.path.exists(noise_path):
            with open(noise_path, "rb") as file:
                st.download_button(label="Unduh Noise (.npy)", data=file, file_name="simulasi_noise.npy", mime="application/octet-stream")

uploaded_file = st.file_uploader("Unggah file .npy (Gempa / Noise) untuk memulai simulasi", type=["npy"])

# ==========================================
# 5. LOGIKA SIMULASI & EEWS
# ==========================================
if uploaded_file is not None:
    raw_data = np.load(uploaded_file)
    
    if raw_data.shape != (3, 6000):
        st.error(f"Format array salah. Diharapkan (3, 6000), tetapi mendapatkan {raw_data.shape}")
        st.stop()
        
    clean_data = signal_to_waveform(raw_data)
    
    mulai_simulasi = st.button("Mulai Simulasi Streaming")
    st.write("") 
    
    col_status, col_countdown = st.columns(2)
    status_placeholder = col_status.empty()
    countdown_placeholder = col_countdown.empty()
    chart_placeholder = st.empty()
    
    if mulai_simulasi:
        streaming_buffer = np.zeros((3, 6000))
        
        status_placeholder.markdown('<div class="status-aman">STATUS: AMAN (Pemantauan Aktif)</div>', unsafe_allow_html=True)
        countdown_placeholder.markdown('<div class="status-aman">EVAKUASI: -</div>', unsafe_allow_html=True)
        
        # Variabel kontrol regresi
        p_locked = False
        waktu_tiba_P_locked = 0.0
        regresi_selesai = False
        prediksi_waktu_S = 0.0
        estimasi_durasi_S = 0.0
        
        langkah_sampel = 100
        total_langkah = 6000 // langkah_sampel
        sumbu_waktu = np.arange(6000) / 100.0
        
        for i in range(1, total_langkah + 1):
            batas_waktu = i * langkah_sampel
            waktu_sekarang = batas_waktu / 100.0
            streaming_buffer[:, :batas_waktu] = clean_data[:, :batas_waktu]
            
            input_tensor = torch.tensor(streaming_buffer, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model_unet(input_tensor)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
            prob_P = probs[0, :]
            prob_S = probs[1, :]
            
            p_terdeteksi = False
            s_terdeteksi = False
            waktu_tiba_P = 0
            waktu_tiba_S = 0
            
            puncak_P = np.max(prob_P[:batas_waktu])
            if puncak_P >= 0.5:
                p_terdeteksi = True
                waktu_tiba_P = np.argmax(prob_P[:batas_waktu]) / 100.0
                # Kunci waktu tiba P untuk diekstrak 3 detiknya
                if not p_locked:
                    p_locked = True
                    waktu_tiba_P_locked = waktu_tiba_P
                
            puncak_S = np.max(prob_S)
            if puncak_S >= 0.5:
                waktu_tiba_S = np.argmax(prob_S) / 100.0
                if waktu_tiba_S > waktu_tiba_P:
                    s_terdeteksi = True

            # Eksekusi Model Regresi tepat 3 detik setelah P tiba
            if p_locked and not regresi_selesai and (waktu_sekarang >= waktu_tiba_P_locked + 3.0):
                idx_p = int(waktu_tiba_P_locked * 100)
                potongan_3detik = streaming_buffer[:, idx_p : idx_p + 300]
                tensor_reg = torch.tensor(potongan_3detik, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    ts_min_tp = model_reg(tensor_reg).item()
                estimasi_durasi_S = ts_min_tp
                regresi_selesai = True

# Pembaruan UI Peringatan Dini
            if p_terdeteksi:
                status_placeholder.markdown(f'<div class="status-bahaya">STATUS: SIAGA GEMPA!<br>Fase P di {waktu_tiba_P:.2f}s</div>', unsafe_allow_html=True)
                
                if s_terdeteksi:
                    # Sinyal S AKTUAL terdeteksi oleh U-Net
                    if regresi_selesai:
                        countdown_placeholder.markdown(
                            f'<div class="status-bahaya">Prediksi: S muncul dalam {estimasi_durasi_S:.1f}s<br>FASE S TIBA di {waktu_tiba_S:.2f}s!</div>', 
                            unsafe_allow_html=True
                        )
                    else:
                        countdown_placeholder.markdown(
                            f'<div class="status-bahaya">FASE S TIBA di {waktu_tiba_S:.2f}s!<br>Berlindung!</div>', 
                            unsafe_allow_html=True
                        )
                else:
                    # Sinyal S aktual belum tiba
                    if not regresi_selesai:
                        countdown_placeholder.markdown('<div class="status-bahaya">Prediksi Fase S......</div>', unsafe_allow_html=True)
                    else:
                        countdown_placeholder.markdown(
                            f'<div class="status-bahaya">Prediksi:<br>Fase S muncul dalam {estimasi_durasi_S:.1f} detik</div>', 
                            unsafe_allow_html=True
                        )

            # Merender dengan Matplotlib
            fig, ax = plt.subplots(figsize=(12, 5))
            fig.patch.set_facecolor('#FDFBF7')
            ax.set_facecolor('#FDFBF7')
            
            ax.plot(sumbu_waktu[:batas_waktu], streaming_buffer[0, :batas_waktu], color='#2B2B2B', linewidth=1, label='Getaran (Z)')
            ax.plot(sumbu_waktu, prob_P, color='#8BA88E', linewidth=2.5, label='Probabilitas Fase P')
            ax.plot(sumbu_waktu, prob_S, color='#D37D6E', linewidth=2.5, label='Probabilitas Fase S')
            
            # Tambahkan garis penanda hasil regresi di plot jika sudah selesai dihitung
            if regresi_selesai:
                ax.axvline(x=prediksi_waktu_S, color='#D37D6E', linestyle='--', linewidth=1.5, label='Prediksi Tiba S (Regresi)')
            
            ax.set_ylim(-4, 4)
            ax.set_xlim(0, 60)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#E5E5E5')
            ax.spines['bottom'].set_color('#E5E5E5')
            ax.set_xlabel("Waktu (Detik)", fontsize=10, fontweight='bold', color='#2B2B2B')
            ax.set_ylabel("Amplitudo", fontsize=10, fontweight='bold', color='#2B2B2B')
            ax.legend(loc='upper right', ncol=4, frameon=False, fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            chart_placeholder.pyplot(fig)
            plt.close(fig) 
            
            time.sleep(0.05)