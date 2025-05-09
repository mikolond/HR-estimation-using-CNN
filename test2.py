import os
import numpy as np
import mne
import csv

def extract_heart_rate_bpm(bdf_path, csv_path=None):
    """
    Extracts instantaneous heart rate (bpm) from a BioSemi .bdf file
    and writes it to a CSV with header "TimeStamp,HeartRate".
    Any heart rate outside [40, 240] bpm is flagged in the console.

    Parameters
    ----------
    bdf_path : str
        Path to the input .bdf file.
    csv_path : str, optional
        Path to the output .csv file. If None, uses the same base name
        as `bdf_path` with suffix '_hr.csv'.
    """
    # Determine output CSV path
    if csv_path is None:
        base, _ = os.path.splitext(bdf_path)
        csv_path = base + '_hr.csv'

    # 1) Load the BDF file (preload into memory for filtering)  
    raw = mne.io.read_raw_bdf(bdf_path, preload=True)  # :contentReference[oaicite:0]{index=0}

    # 2) Select the ECG channel (here assumed 'EXG1')  
    if 'EXG1' not in raw.ch_names:
        raise ValueError("ECG channel 'EXG1' not found in data.")
    ecg = raw.copy().pick_channels(['EXG1'])            # :contentReference[oaicite:1]{index=1}

    # 3) Band-pass filter to isolate the QRS complex (5–35 Hz)  
    ecg.filter(l_freq=5., h_freq=35., fir_design='firwin')  # :contentReference[oaicite:2]{index=2}

    # 4) Detect R-wave events (returns events, channel index, and avg pulse)  
    events, ch_ecg, avg_pulse = mne.preprocessing.find_ecg_events(
        ecg, ch_name='EXG1', return_ecg=False, l_freq=5., h_freq=35.
    )                                                    # :contentReference[oaicite:3]{index=3}

    # 5) Convert sample indices to times (seconds)
    sfreq = raw.info['sfreq']  # sampling frequency in Hz
    peak_times = events[:, 0] / sfreq

    # 6) Compute instantaneous HR (bpm) from RR intervals
    rr_intervals = np.diff(peak_times)
    hr_bpm = 60.0 / rr_intervals
    hr_times = peak_times[1:]

    # 7) Write out CSV and flag out-of-range values
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['TimeStamp', 'HeartRate'])
        for t, hr in zip(hr_times, hr_bpm):
            if hr < 40 or hr > 240:
                print(f"Warning: heart rate {hr:.1f} bpm at {t:.3f} s is outside [40,240] bpm")
            writer.writerow([f"{t:.3f}", f"{hr:.1f}"])

    print(f"Heart-rate CSV written to: {csv_path}")

if __name__ == '__main__':
    import sys

    extract_heart_rate_bpm("/home/ondrej/Desktop/ptak_download/hci/1/Part_1_N_Trial1_emotion.bdf", "hr_out.csv")
