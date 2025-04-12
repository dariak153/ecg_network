import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt

class ECG:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.leads = ["i", "ii", "iii", "avr", "avl", "avf",
                      "v1", "v2", "v3", "v4", "v5", "v6"]

    def read_record(self, rec_id, lead):
      
        path = os.path.join(self.data_dir, str(rec_id))
        record = wfdb.rdrecord(path)
        signal, fs = record.p_signal, record.fs

        ann = wfdb.rdann(path, extension=lead.lower())
        ann_list = []
        for i, sym in enumerate(ann.symbol):
            if sym in ["p", "N", "t"]:
                peak = ann.sample[i]
                start = ann.sample[i - 1] if i > 0 and ann.symbol[i - 1] == "(" else peak
                end = ann.sample[i + 1] if i < len(ann.symbol) - 1 and ann.symbol[i + 1] == ")" else peak
                duration = (end - start) * 1000 / fs
                ann_list.append({
                    "label": sym,
                    "peak": peak,
                    "start": start,
                    "end": end,
                    "duration": duration
                })
        return signal, fs, ann_list

def process_record(ecg, rec_id, lead):
   
    full_signal, fs, ann_list = ecg.read_record(rec_id, lead)
    lead_idx = ecg.leads.index(lead.lower())
    sig = full_signal[:, lead_idx]
    
 
    mask = np.zeros(len(sig), dtype=np.uint8)
    for ann in ann_list:
        if ann["label"] == "N":
            s = max(0, ann["start"])
            e = min(len(sig) - 1, ann["end"])
            mask[s:e+1] = 1
    return sig, mask, ann_list, fs

def plot_record(signal, mask, ann_list, fs):
   
    for ann in ann_list:
        if ann["label"] == "N":
            print(f"Onset: {ann['start']/fs:.3f}s, Peak: {ann['peak']/fs:.3f}s, "
                  f"End: {ann['end']/fs:.3f}s, Duration: {ann['duration']:.2f}ms")
    

    t = np.arange(len(signal)) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal, label="ECG Signal")
    
  
    colors = {"QRS": "red", "P": "blue", "T": "green"}
    for ann in ann_list:
        if ann["label"] == "N":
            wave = "QRS"
        elif ann["label"].lower() == "p":
            wave = "P"
        elif ann["label"].lower() == "t":
            wave = "T"
        else:
            continue
        
        t_start = ann["start"] / fs
        t_peak = ann["peak"] / fs
        t_end = ann["end"] / fs
        
        plt.plot(t_peak, signal[ann["peak"]], 'o', color=colors[wave])
        plt.axvline(t_start, linestyle="--", color=colors[wave])
        plt.axvline(t_end, linestyle="--", color=colors[wave])
    
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(["ECG Signal", "Annotations"])
    plt.tight_layout()
    plt.show()
    

    plt.figure(figsize=(12, 4))
    plt.plot(t, signal, label="ECG Signal")
    plt.fill_between(t, signal.min(), signal.max(), where=mask==1,
                     color="red", alpha=0.3, label="QRS Mask")
    
    for ann in ann_list:
        if ann["label"] == "N":
            t_start = ann["start"] / fs
            t_end = ann["end"] / fs
            plt.axvline(t_start, linestyle="--", color="red")
            plt.axvline(t_end, linestyle="--", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("ECG with QRS Mask")
    plt.legend()
    plt.tight_layout()
    plt.show()


data_dir = "datasets_physionet/data"
ecg = ECG(data_dir)
record_ids = range(1, 201)


train_leads = ["ii", "v1", "v4"]
val_leads   = ["i", "v2", "v5"]
test_leads  = ["iii", "v3", "v6"]

train_signals, train_masks = [], []
val_signals, val_masks = [], []
test_signals, test_masks = [], []

for idx, rec_id in enumerate(record_ids):

    for lead in train_leads:
        sig, mask, ann_list, fs = process_record(ecg, rec_id, lead)
        if sig is not None:
            train_signals.append(sig)
            train_masks.append(mask)
            if idx < 3:
                print(f"Train Record {rec_id}, Lead {lead}")
                plot_record(sig, mask, ann_list, fs)

    for lead in val_leads:
        sig, mask, ann_list, fs = process_record(ecg, rec_id, lead)
        if sig is not None:
            val_signals.append(sig)
            val_masks.append(mask)
            if idx < 3:
                print(f"Validation Record {rec_id}, Lead {lead}")
                plot_record(sig, mask, ann_list, fs)

    for lead in test_leads:
        sig, mask, ann_list, fs = process_record(ecg, rec_id, lead)
        if sig is not None:
            test_signals.append(sig)
            test_masks.append(mask)
            if idx < 3:
                print(f"Test Record {rec_id}, Lead {lead}")
                plot_record(sig, mask, ann_list, fs)

np.save("train_signals.npy", np.array(train_signals, dtype=object), allow_pickle=True)
np.save("train_masks.npy",   np.array(train_masks, dtype=object), allow_pickle=True)
np.save("val_signals.npy",   np.array(val_signals, dtype=object), allow_pickle=True)
np.save("val_masks.npy",     np.array(val_masks, dtype=object), allow_pickle=True)
np.save("test_signals.npy",  np.array(test_signals, dtype=object), allow_pickle=True)
np.save("test_masks.npy",    np.array(test_masks, dtype=object), allow_pickle=True)
