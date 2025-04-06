import wfdb
import numpy as np
import os

class ECG:
    def __init__(self, base_path):
        self.base_path = base_path
        self.leads = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]

    def load_record(self, record_id):
        rec = wfdb.rdrecord(f"{self.base_path}/{record_id}")
        return rec.p_signal, rec.fs

    def load_annotations(self, record_id, lead, fs):
        try:
            ann = wfdb.rdann(f"{self.base_path}/{record_id}", extension=lead.lower())
        except Exception as e:
            print(f"Error loading record {record_id}: {e}")
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
                end = min(length - 1, ann["offset"])
                mask[start:end + 1] = 1
        return mask

def process_record(ecg, record_id, lead):
    try:
        signal, fs = ecg.load_record(record_id)
        if lead.lower() not in ecg.leads:
            raise ValueError(f"Lead {lead} not found in record {record_id}.")
        idx = ecg.leads.index(lead.lower())
        raw_signal = signal[:, idx]
        anns = ecg.load_annotations(record_id, lead, fs)
        mask = ecg.mask_create(len(raw_signal), anns)
        return raw_signal, mask, anns, fs
    except Exception as e:
        print(f"Error processing record {record_id} with lead {lead}: {e}")
        return None, None, None, None

def save_data(data_dict, folder, file_name):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, file_name)
    np.save(file_path, data_dict)
    print(f"Saved data to {file_path}")

def main():
    base_path = "datasets_physionet/data"
    ecg = ECG(base_path)
    
 
    record_ids = list(range(1, 200))
 
    path = {
        "train": {"lead": "ii", "folder": "masks/train", "signals_folder": "signals/train"},
        "validation": {"lead": "iii", "folder": "masks/validation", "signals_folder": "signals/validation"},
        "test": {"lead": "i", "folder": "masks/test", "signals_folder": "signals/test"}
    }
    
    for subset_name, config in path.items():
        print(f"\nProcessing subset: {subset_name} with lead {config['lead']}")
        masks = {}
        signals = {}
        for rec_id in record_ids:
            raw_signal, mask, anns, fs = process_record(ecg, rec_id, config["lead"])
            if raw_signal is not None:
                signals[rec_id] = raw_signal
                masks[rec_id] = mask
        save_data(masks, config["folder"], f"masks_{subset_name}.npy")
        save_data(signals, config["signals_folder"], f"signals_{subset_name}.npy")

if __name__ == "__main__":
    main()

