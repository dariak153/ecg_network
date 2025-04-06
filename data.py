import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt

class ECG:
    def __init__(self, base_path):
        self.base_path = base_path
        self.leads = ["i", "ii", "iii", "avr", "avl", "avf",
                      "v1", "v2", "v3", "v4", "v5", "v6"]

    def load_record(self, record_id):
        rec = wfdb.rdrecord(f"{self.base_path}/{record_id}")
        return rec.p_signal, rec.fs

    def load_annotations(self, record_id, lead, fs):
        try:
            ann = wfdb.rdann(f"{self.base_path}/{record_id}", extension=lead.lower())
        except Exception as e:
            print(f"Error loading record {record_id}, lead {lead}: {e}")
            return []
        anns = []
        for i, sym in enumerate(ann.symbol):
            if sym in ["p", "N", "t"]:
                peak = ann.sample[i]
                onset = ann.sample[i - 1] if i > 0 and ann.symbol[i - 1] == "(" else peak
                offset = ann.sample[i + 1] if i < len(ann.symbol) - 1 and ann.symbol[i + 1] == ")" else peak
                duration = (offset - onset) * 1000 / fs
                anns.append({
                    "label": sym,
                    "peak": peak,
                    "onset": onset,
                    "offset": offset,
                    "duration": duration
                })
        return anns

    def mask_create(self, length, anns):
        mask = np.zeros(length, dtype=np.uint8)
        for ann in anns:
            if ann["label"] == "N":
                start = max(0, ann["onset"])
                end   = min(length - 1, ann["offset"])
                mask[start:end + 1] = 1
        return mask

def process_record(ecg, record_id, lead):
    try:
        signal, fs = ecg.load_record(record_id)
        idx = ecg.leads.index(lead.lower())
        raw_signal = signal[:, idx]
        anns = ecg.load_annotations(record_id, lead, fs)
        mask = ecg.mask_create(len(raw_signal), anns)
        return raw_signal, mask, anns, fs
    except Exception as e:
        print(f"Error processing record {record_id}, lead {lead}: {e}")
        return None, None, None, None

def qrs_dur(anns, fs):
    for ann in anns:
        if ann["label"] == "N":
            onset_t  = ann["onset"]  / fs
            peak_t   = ann["peak"]   / fs
            offset_t = ann["offset"] / fs
            print(f"Onset: {onset_t:.3f}s, Peak: {peak_t:.3f}s, "
                  f"Offset: {offset_t:.3f}s, Duration: {ann['duration']:.2f}ms")

def plot_ann(signal, anns, fs):
    t = np.arange(len(signal)) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal, label="ECG Signal")
    colors = {"QRS": "red", "P": "blue", "T": "green"}
    used = {"QRS": False, "P": False, "T": False}
    for ann in anns:
        if ann["label"] == "N":
            wave = "QRS"
        elif ann["label"].lower() == "p":
            wave = "P"
        elif ann["label"].lower() == "t":
            wave = "T"
        else:
            continue
        onset  = ann["onset"]  / fs
        peak   = ann["peak"]   / fs
        offset = ann["offset"] / fs
        if not used[wave]:
            plt.plot(peak, signal[ann["peak"]], 'o', color=colors[wave], label=f"{wave} Peak")
            plt.axvline(onset,  linestyle="--", color=colors[wave], label=f"{wave} Onset/Offset")
            plt.axvline(offset, linestyle="--", color=colors[wave])
            used[wave] = True
        else:
            plt.plot(peak, signal[ann["peak"]], 'o', color=colors[wave])
            plt.axvline(onset,  linestyle="--", color=colors[wave])
            plt.axvline(offset, linestyle="--", color=colors[wave])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_mask(signal, mask, anns, fs):
    t = np.arange(len(signal)) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal, label="ECG Signal")
    plt.fill_between(t, signal.min(), signal.max(), where=mask == 1, color="red", alpha=0.3, label="QRS Mask")
    for ann in anns:
        if ann["label"] == "N":
            onset  = ann["onset"]  / fs
            offset = ann["offset"] / fs
            plt.axvline(onset,  linestyle="--", color="red")
            plt.axvline(offset, linestyle="--", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("QRS Mask")
    plt.legend()
    plt.tight_layout()
    plt.show()

base_path = "datasets_physionet/data"
ecg = ECG(base_path)
record_ids = range(1, 201)

train_leads = ["ii", "v1", "v4"]
val_leads   = ["i",  "v2", "v5"]
test_leads  = ["iii","v3", "v6"]

train_signals = []
train_masks   = []
val_signals   = []
val_masks     = []
test_signals  = []
test_masks    = []

for i, rec_id in enumerate(record_ids):
    for lead in train_leads:
        raw_signal, mask, anns, fs = process_record(ecg, rec_id, lead)
        if raw_signal is not None:
            train_signals.append(raw_signal)
            train_masks.append(mask)
            if i < 3:
                print(f"[Train] Record {rec_id}, lead {lead}")
                qrs_dur(anns, fs)
                plot_ann(raw_signal, anns, fs)
                plot_mask(raw_signal, mask, anns, fs)

    for lead in val_leads:
        raw_signal, mask, anns, fs = process_record(ecg, rec_id, lead)
        if raw_signal is not None:
            val_signals.append(raw_signal)
            val_masks.append(mask)
            if i < 3:
                print(f"[Val]   Record {rec_id}, lead {lead}")
                qrs_dur(anns, fs)
                plot_ann(raw_signal, anns, fs)
                plot_mask(raw_signal, mask, anns, fs)

    for lead in test_leads:
        raw_signal, mask, anns, fs = process_record(ecg, rec_id, lead)
        if raw_signal is not None:
            test_signals.append(raw_signal)
            test_masks.append(mask)
            if i < 3:
                print(f"[Test]  Record {rec_id}, lead {lead}")
                qrs_dur(anns, fs)
                plot_ann(raw_signal, anns, fs)
                plot_mask(raw_signal, mask, anns, fs)

np.save("train_signals.npy", np.array(train_signals, dtype=object), allow_pickle=True)
np.save("train_masks.npy",   np.array(train_masks,   dtype=object), allow_pickle=True)
np.save("val_signals.npy",   np.array(val_signals,   dtype=object), allow_pickle=True)
np.save("val_masks.npy",     np.array(val_masks,     dtype=object), allow_pickle=True)
np.save("test_signals.npy",  np.array(test_signals,  dtype=object), allow_pickle=True)
np.save("test_masks.npy",    np.array(test_masks,    dtype=object), allow_pickle=True)
