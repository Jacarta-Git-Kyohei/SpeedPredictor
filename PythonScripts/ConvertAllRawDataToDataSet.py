import pandas as pd
import numpy as np
import os
from scipy.fft import fft, fftfreq
from datetime import datetime

TimeStamp = datetime.now().strftime("%m%d_%H%M")

# ------ Change parameters according to the situation ------ #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_SAVED_FOLDER = os.path.join(SCRIPT_DIR, "..", "RawData")
OUTPUT_SAVED_FOLDER = os.path.join(SCRIPT_DIR, "..", "CreatedDataset")
OUTPUT_FILE = os.path.join(OUTPUT_SAVED_FOLDER, f"SpeedPredictorDataset_{TimeStamp}.csv")
MIN_TARGET_SPEED = 3
MAX_TARGET_SPEED = 10
CHUNK_INTERVAL_MS = 3500 # MS : millisecond
RAW_DATA_FOR_ENERGY_ESTIMATE_FILE_NAME = "RawData_EnergyTest.csv"
bCreateDatasetForEnergyEstimation = False
# ---------------------------------------------------------- #

# If the folder does not exist, make it.
os.makedirs(OUTPUT_SAVED_FOLDER, exist_ok=True)

# ===================================================
# Calculate top two dominant frequency.
# Returns specified DomFreq.
# Signal : Time Series Signal Array [time, signal]
# fps : Sampling frequency
# MinFreq : Minimum value of low-frequency noise to be ignored
# ===================================================
def GetTopFrequencies(Signal, fps, MinFreq = 0.5):

    N = len(Signal)

    # Remove DC
    signal_centered = Signal - np.mean(Signal)

    # Hann window
    window = np.hanning(N)
    signal_win = signal_centered * window

    # Zero-padding
    N_fft = 4096
    padded = np.zeros(N_fft)
    padded[:N] = signal_win

    # FFT
    fft_vals = fft(padded)
    freqs = fftfreq(len(padded), d=1.0 / fps)

    # Positive frequencies only
    mask = freqs > 0
    freqs_pos = freqs[mask]
    fft_pos = np.abs(fft_vals[mask])

    # Remove low frequency noise
    valid = freqs_pos >= MinFreq
    freqs_pos = freqs_pos[valid]
    fft_pos = fft_pos[valid]

    if len(fft_pos) == 0:
        return 0.0, 0.0

    # Sort indices by amplitude (descending)
    top_idx = np.argsort(fft_pos)[::-1]

    # 1st & 2nd dominant frequencies
    Freq1 = freqs_pos[top_idx[0]] if len(top_idx) > 0 else 0.0
    Freq2 = freqs_pos[top_idx[1]] if len(top_idx) > 1 else 0.0

    return Freq1, Freq2


# ===================================================
# Extract features for each chunk.
# Returns Features.
# ChunkedDataset : A dataset divided into specified chunks
# ===================================================
def ExtractFeatures(ChunkedDataset):

    Features = {}

    # average data acquisition frame rate within the chunk
    duration_sec = (ChunkedDataset["time"].iloc[-1] - ChunkedDataset["time"].iloc[0]) / 1000.0
    fps = len(ChunkedDataset) / duration_sec   # (fps = Frame Per Second)

    # Magnitude of the acceleration vector
    AccelScalar = np.sqrt(ChunkedDataset["accel_x"]**2 + ChunkedDataset["accel_y"]**2 + ChunkedDataset["accel_z"]**2)

    # Magnitude of the angular velocity vector
    AngularScalar  = np.sqrt(ChunkedDataset["gyro_x"]**2  + ChunkedDataset["gyro_y"]**2  + ChunkedDataset["gyro_z"]**2)

    # Acceleration STD, RMS, Dominant Frequency
    Features["accel_STD"]  = AccelScalar.std()
    Features["accel_RMS"]  = np.sqrt(np.mean(AccelScalar**2))
    accel_domfreq1, accel_domfreq2 = GetTopFrequencies(AccelScalar, fps)
    Features["accel_DomFreq1"] = accel_domfreq1
    Features["accel_DomFreq2"] = accel_domfreq2

    # Angular STD, RMS, Dominant Frequency
    Features["ang_STD"]  = AngularScalar.std()
    Features["ang_RMS"]  = np.sqrt(np.mean(AngularScalar**2))
    ang_domfreq1, ang_domfreq2 = GetTopFrequencies(AngularScalar, fps)
    Features["ang_DomFreq1"] = ang_domfreq1
    Features["ang_DomFreq2"] = ang_domfreq2

    return Features

# ===================================================
# Create a dataset for a single RawData file.
# Returns a dataset for the specified correct labels.
# FileName : Raw data file to be converted into a dataset
# TargetSpeed : Correct answer label
# ===================================================
def MakeDatasetPerOneFile(FileName, TargetSpeed):

    print(f"Processing {FileName} (TargetSpeed={TargetSpeed} km/h)...")

    RawDataSet = pd.read_csv(FileName)

    # Just in case, verify that time is sorted in ascending order.
    RawDataSet = RawDataSet.sort_values("time").reset_index(drop=True)

    Dataset = []
    StartIndex = 0

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

    if bCreateDatasetForEnergyEstimation:
        Filename = os.path.join(RAW_DATA_SAVED_FOLDER, RAW_DATA_FOR_ENERGY_ESTIMATE_FILE_NAME)
        Dataset_ForEnergyEstimation = MakeDatasetPerOneFile(Filename, 0)
        AllDataSets.append(Dataset_ForEnergyEstimation)

    else:
        for n in range(MIN_TARGET_SPEED, MAX_TARGET_SPEED + 1):
            for p in range(0,10):
                Filename = os.path.join(RAW_DATA_SAVED_FOLDER, f"RawData_{n}_{p}.csv")
                if not os.path.exists(Filename):
                    print(f"Warning: RawData_{n}_{p}.csv not found, skipping.")
                    continue
                TargetSpeed = n + p * 0.1
                Dataset_N = MakeDatasetPerOneFile(Filename, TargetSpeed)
                AllDataSets.append(Dataset_N)

    FinalDataFrame = pd.concat(AllDataSets, ignore_index=True)
    FinalDataFrame.to_csv(OUTPUT_FILE, index=False)
    print(f"\n Success! : datasets saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
