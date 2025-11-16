import pandas as pd
import numpy as np
import os
from scipy.fft import fft, fftfreq
from datetime import datetime

timestamp = datetime.now().strftime("%m%d_%H%M")

# ------ Change parameters according to the situation ------ #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_SAVED_FOLDER = os.path.join(SCRIPT_DIR, "..", "RawData")
OUTPUT_SAVED_FOLDER = os.path.join(SCRIPT_DIR, "..", "CreatedDataset")
OUTPUT_FILE = os.path.join(OUTPUT_SAVED_FOLDER, f"SpeedPredictorDataset_{timestamp}.csv")
MIN_TARGET_SPEED = 3
MAX_TARGET_SPEED = 10
CHUNK_INTERVAL_MS = 3000   # MS : millisecond
# ---------------------------------------------------------- #

# If the folder does not exist, make it.
os.makedirs(OUTPUT_SAVED_FOLDER, exist_ok=True)

# ===================================================
# Calculate dominant frequency.
# Returns specified DomFreq.
# Signal :
# fps :
# ===================================================
def GetDominantFrequency(signal, fps):

    N = len(signal)

    # DC成分除去
    signal_centered = signal - np.mean(signal)

    # FFT
    fft_vals = fft(signal_centered)
    freqs = fftfreq(N, d=1.0/fps)

    # 正の周波数のみ
    mask = freqs > 0
    fft_positive = fft_vals[mask]
    freqs_positive = freqs[mask]

    # 主周波数
    DomFreq = freqs_positive[np.argmax(np.abs(fft_positive))]

    return DomFreq


# ===================================================
# Extract features for each chunk.
# Returns Features {accel_STD, accel_RMS, accel_DomFreq, ang_STD, ang_RMS, ang_DomFreq}.
# Chunked_Dataset : A dataset divided into specified chunks
# ===================================================
def ExtractFeatures(Chunked_Dataset):

    Features = {}

    # average data acquisition frame rate within the chunk
    duration_sec = (Chunked_Dataset["time"].iloc[-1] - Chunked_Dataset["time"].iloc[0]) / 1000.0
    fps = len(Chunked_Dataset) / duration_sec   # (fps = Frame Per Second)

    # Magnitude of the acceleration vector
    AccelScalar = np.sqrt(Chunked_Dataset["accel_x"]**2 + Chunked_Dataset["accel_y"]**2 + Chunked_Dataset["accel_z"]**2)

    # Magnitude of the angular velocity vector
    AngularScalar  = np.sqrt(Chunked_Dataset["gyro_x"]**2  + Chunked_Dataset["gyro_y"]**2  + Chunked_Dataset["gyro_z"]**2)

    # Acceleration STD, RMS, Dominant Frequency
    Features["accel_STD"]  = AccelScalar.std()
    Features["accel_RMS"]  = np.sqrt(np.mean(AccelScalar**2))
    Features["accel_DomFreq"] = GetDominantFrequency(AccelScalar, fps)

    # Angular STD, RMS, Dominant Frequency
    Features["ang_STD"]  = AngularScalar.std()
    Features["ang_RMS"]  = np.sqrt(np.mean(AngularScalar**2))
    Features["ang_DomFreq"] = GetDominantFrequency(AngularScalar, fps)

    return Features

# ===================================================
# Create a dataset for a single RawData file.
# Returns a dataset {accel_STD, accel_RMS, accel_DomFreq, ang_STD, ang_RMS, ang_DomFreq, TargetSpeed} for the specified correct labels.
# FileName : Raw data file to be converted into a dataset
# TargetSpeed : Correct answer label
# ===================================================
def MakeDatasetPerOneFile(FileName, TargetSpeed):

    print(f"Processing {FileName} (TargetSpeed={TargetSpeed} km/h)...")

    RawDataSet = pd.read_csv(FileName)
    Dataset = []
    StartIndex = 0

    # Just in case, verify that time is sorted in ascending order.
    RawDataSet = RawDataSet.sort_values("time").reset_index(drop=True)

    # Dataset Creation
    while True:

        # Divide the raw dataset into chunks at specified intervals
        TIME_START = RawDataSet.loc[StartIndex, "time"]
        TIME_END = TIME_START + CHUNK_INTERVAL_MS
        EndIndex = (RawDataSet["time"] - TIME_END).abs().idxmin()
        if EndIndex <= StartIndex or EndIndex >= len(RawDataSet) - 1:
            break
        ChunkedDataset = RawDataSet.loc[StartIndex:EndIndex]

        # Add features calculated from raw data within one chunk to the dataset
        FeaturesPerChunk = ExtractFeatures(ChunkedDataset)
        Dataset.append(FeaturesPerChunk)

        # Start index of the next chunk
        StartIndex = EndIndex + 1

        if RawDataSet.loc[StartIndex, "time"] >= RawDataSet["time"].iloc[-1] - CHUNK_INTERVAL_MS:
            break

    DataFrame = pd.DataFrame(Dataset)
    DataFrame["TargetSpeed"] = TargetSpeed

    return DataFrame

# ===================================================
# Main : Creation of a comprehensive dataset ranging from MIN_TARGET_SPEED to MAX_TARGET_SPEED
# ===================================================
def main():

    AllDataSets = []

    for n in range(MIN_TARGET_SPEED, MAX_TARGET_SPEED + 1):
        Filename = os.path.join(RAW_DATA_SAVED_FOLDER, f"RawData_{n}.csv")
        if not os.path.exists(Filename):
            print(f"Warning: {Filename} not found, skipping.")
            continue

        Dataset_N = MakeDatasetPerOneFile(Filename, TargetSpeed=n)
        AllDataSets.append(Dataset_N)

    FinalDataFrame = pd.concat(AllDataSets, ignore_index=True)
    FinalDataFrame.to_csv(OUTPUT_FILE, index=False)
    print(f"\n All datasets saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
