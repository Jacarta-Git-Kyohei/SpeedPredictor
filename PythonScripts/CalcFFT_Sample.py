import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import os

# ================================
# ユーザー設定
# ================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_SAVED_FOLDER = os.path.join(SCRIPT_DIR, "..", "RawData")
FILENAME = "RawData_4.csv"   # 生データファイル
CHUNK_SECONDS = 3             # 3秒ごとにFFT
PLOT_FFT = False              # Trueで各区間のFFT波形を表示
# ================================


def load_data(filename):
    df = pd.read_csv(filename)
    return df


def compute_vector_magnitude(df):
    df["accel_abs"] = np.sqrt(df["accel_x"]**2 + df["accel_y"]**2 + df["accel_z"]**2)
    df["gyro_abs"] = np.sqrt(df["gyro_x"]**2 + df["gyro_y"]**2 + df["gyro_z"]**2)
    return df


def compute_sampling_rate(time_ms):
    dt = np.diff(time_ms) / 1000.0
    dt_mean = np.mean(dt)
    fs = 1.0 / dt_mean
    return fs


# ============================================
# ここが改良版（Hanning＋ゼロパディング）
# ============================================
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend

def get_dominant_frequency(signal, fs, min_freq=0.5, zero_pad_len=4096, remove_trend=True):
    """
    改良版：
    - 平均/トレンド除去（option）
    - Hann窓
    - ゼロパディング（長め：zero_pad_len）
    - min_freq 以上の範囲で最大ピークを選択

    パラメータ:
    - signal: 1D numpy array (時間信号)
    - fs: サンプリング周波数 (Hz)
    - min_freq: 最低探索周波数 (Hz). 期待する周波数帯があるなら適宜上げてください。
    - zero_pad_len: FFT 長（ゼロパディング後の長さ）。2のべき乗である必要はありませんが、高いほど周波数分解能が上がる。
    """
    x = np.asarray(signal).astype(float)
    N = len(x)

    # 1) 平均・トレンド除去（推奨）
    if remove_trend:
        # 直流と線形トレンドを除去
        x = detrend(x, type='constant')  # 平均を引くだけにしたければ type='constant'
        # x = detrend(x, type='linear')  # 線形トレンドも除去するならこちら

    # 2) Hann窓
    window = np.hanning(N)
    x_win = x * window

    # 3) ゼロパディング（ユーザー指定長）
    N_pad = int(zero_pad_len)
    if N_pad < N:
        # 最低でも元の長さは使う
        N_pad = 2 ** int(np.ceil(np.log2(N)))
    x_pad = np.zeros(N_pad)
    x_pad[:N] = x_win

    # 4) FFT
    Y = rfft(x_pad)
    freqs = rfftfreq(N_pad, 1.0/fs)

    # 5) DCを確実に無視（安全策）
    # find indices where freq >= min_freq
    idx_min = np.searchsorted(freqs, min_freq, side='left')
    if idx_min <= 0:
        idx_min = 1

    # pick peak in valid range
    magnitude = np.abs(Y)
    # zero out below min index to avoid selecting low-freq bin
    magnitude[:idx_min] = 0

    peak_idx = np.argmax(magnitude)
    dom_freq = freqs[peak_idx]

    return dom_freq, freqs, Y



def main():
    Filename = os.path.join(RAW_DATA_SAVED_FOLDER, FILENAME)
    df = pd.read_csv(Filename)
    df = compute_vector_magnitude(df)

    fs = compute_sampling_rate(df["time"].values)
    print(f"\n推定サンプリング周波数: {fs:.2f} Hz")

    samples_per_chunk = int(fs * CHUNK_SECONDS)
    print(f"1区間のサンプル数: {samples_per_chunk}")

    accel_domfreq_list = []
    gyro_domfreq_list = []

    total_chunks = len(df) // samples_per_chunk
    print(f"\n解析する区間数: {total_chunks}\n")

    for i in range(total_chunks):
        chunk = df.iloc[i * samples_per_chunk : (i+1) * samples_per_chunk]

        accel_signal = chunk["accel_abs"].values
        gyro_signal = chunk["gyro_abs"].values

        accel_dom, xf_a, yf_a = get_dominant_frequency(accel_signal, fs)
        gyro_dom, xf_g, yf_g = get_dominant_frequency(gyro_signal, fs)

        accel_domfreq_list.append(accel_dom)
        gyro_domfreq_list.append(gyro_dom)

        print(f"区間 {i+1}: accel_DomFreq = {accel_dom:.3f} Hz , gyro_DomFreq = {gyro_dom:.3f} Hz")

        if PLOT_FFT:
            plt.figure(figsize=(10,4))
            plt.title(f"Chunk {i+1} Accel FFT")
            plt.plot(xf_a, np.abs(yf_a))
            plt.grid()
            plt.show()

            plt.figure(figsize=(10,4))
            plt.title(f"Chunk {i+1} Gyro FFT")
            plt.plot(xf_g, np.abs(yf_g))
            plt.grid()
            plt.show()

    print("\n=== 解析終了 ===")


if __name__ == "__main__":
    main()
