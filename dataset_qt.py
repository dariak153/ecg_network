import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_record(rec_id, base_path):
    try:
        rec_path = os.path.join(base_path, rec_id)
        rec = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, 'q1c')
        
        start_idx, end_idx = ann.sample[0], ann.sample[-1]
        sig = rec.p_signal[start_idx:end_idx+1]
        num_samples = sig.shape[0]
        print(f"Record {rec_id}: {num_samples} samples")
        
        qrs_mask = np.zeros(num_samples, dtype=np.uint8)
        for i in range(len(ann.sample)):
            if ann.symbol[i] == "N":
                peak = ann.sample[i] - start_idx
                onset = ann.sample[i-1] - start_idx if i > 0 and ann.symbol[i-1] == "(" else peak
                offset = ann.sample[i+1] - start_idx if i < len(ann.sample)-1 and ann.symbol[i+1] == ")" else peak
                qrs_mask[onset:offset+1] = 1
                
        return sig, qrs_mask, ann, rec.fs
        
    except Exception as e:
        print(f"Error loading record {rec_id}: {e}")
        return None, None, None, None

def prepare_dataset(id_list, base_path):
    signals = {}
    masks = {}
    max_length = 0
    
    for rec_id in id_list:
        sig, mask, _, _ = load_record(rec_id, base_path)
        if sig is not None:
            signals[rec_id] = sig
            masks[rec_id] = mask
            max_length = max(max_length, len(sig))
    
    dataset = {"lead1": {}, "lead2": {}, "mask": {}}
    
    for rec_id in signals:
        sig_len = len(signals[rec_id])
        if sig_len < max_length:
            pad_before = (max_length - sig_len) // 2
            pad_after = max_length - sig_len - pad_before
            
            padded_sig = np.pad(signals[rec_id], 
                               ((pad_before, pad_after), (0, 0)), 
                               mode='constant')
            
            padded_mask = np.pad(masks[rec_id],
                               (pad_before, pad_after),
                               mode='constant')
        else:
            padded_sig = signals[rec_id]
            padded_mask = masks[rec_id]
        
        dataset["lead1"][rec_id] = padded_sig[:, 0].reshape(-1, 1)
        dataset["lead2"][rec_id] = padded_sig[:, 1].reshape(-1, 1)
        dataset["mask"][rec_id] = padded_mask
    
    print(f"Processed {len(dataset['lead1'])} records, each {max_length} samples long")
    return dataset

def save_dataset(dataset, data_type):
    os.makedirs(f"signals/{data_type}", exist_ok=True)
    os.makedirs(f"masks/{data_type}", exist_ok=True)
    
    np.save(f"signals/{data_type}/{data_type}_lead1.npy", dataset["lead1"])
    np.save(f"signals/{data_type}/{data_type}_lead2.npy", dataset["lead2"])
    np.save(f"masks/{data_type}/{data_type}_mask.npy", dataset["mask"])
    
    print(f"Saved {data_type} dataset with {len(dataset['lead1'])} records")

def process_data(base_path, rec_ids):
    train_val, test = train_test_split(rec_ids, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1/(1-0.1), random_state=42)
    
    print(f"Split {len(rec_ids)} records into: {len(train)} train, {len(val)} validation, {len(test)} test")
    
    train_data = prepare_dataset(train, base_path)
    val_data = prepare_dataset(val, base_path)
    test_data = prepare_dataset(test, base_path)
    
    save_dataset(train_data, "train")
    save_dataset(val_data, "val")
    save_dataset(test_data, "test")
    
    datasets = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }
    
    print("\nTrain records:", list(train_data["lead1"].keys()))
    print("Validation records:", list(val_data["lead1"].keys()))
    print("Test records:", list(test_data["lead1"].keys()))

    print("\nTrain directory:", os.path.abspath("signals/train"))
    print("Validation directory:", os.path.abspath("signals/val"))
    print("Test directory:", os.path.abspath("signals/test"))
    return datasets

def plot_ecg(rec_id, base_path):
    sig, mask, ann, fs = load_record(rec_id, base_path)
    if sig is None:
        print(f"Record {rec_id} not found")
        return
    
    start_idx = ann.sample[0]
    t = np.arange(len(sig)) / fs
    
    ann_samples = ann.sample - start_idx
    ann_samples = ann_samples[(ann_samples >= 0) & (ann_samples < len(sig))]
    ann_time = ann_samples / fs
    
    ann_symbols = np.array(ann.symbol)[(ann.sample >= start_idx) & (ann.sample < start_idx + len(sig))]
    
    colors = {"p": "blue", "N": "red", "t": "green"}
    ann_colors = [colors.get(s, "orange") for s in ann_symbols]
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    for i, (ax, lead_name) in enumerate(zip(axs, ["Lead 1", "Lead 2"])):
        ax.plot(t, sig[:, i], label=lead_name, color="blue")
        
        for tt, sym, col in zip(ann_time, ann_symbols, ann_colors):
            ax.axvline(x=tt, color=col, linestyle="--", alpha=0.7)
        
        y_min, y_max = np.min(sig[:, i]), np.max(sig[:, i])
        ax.fill_between(t, y_min, y_max, where=mask.astype(bool), 
                        color="red", alpha=0.3, label="QRS Mask")
        
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{lead_name} - {rec_id}")
        ax.legend(loc="upper right")
    
    axs[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

def main():
    base_path = 'qt/qt-database-1.0.0/'
    
    rec_ids = [
        "sel123", "sel14157", "sel15814", "sel16265", "sel16272", "sel16273", "sel16420",
        "sel16483", "sel16539", "sel16773", "sel16786", "sel16795", "sel17152", "sel17453",
        "sel0104", "sele0110", "sele0411", "sele0509", "sele0609", "sele0704", "sele0612", 
        "sele0603", "sele0604", "sele0606", "sele0607", "sele0409", "sele0210", "sele0405",
        "sele0303", "sele0211", "sele0203", "sele0170", "sele0121", "sele0133", "sele0136",
        "sele0129", "sele0126", "sele0124", "sele0122", "sele0116", "sele0111", "sel0106",
        "sel853", "sel872", "sel883", "sel803", "sel820", "sel821", "sel52", "sel30",
        "sel811", "sel808", "sel51", "sel49", "sel39", "sel48", "sel47", "sel45", "sel40",
        "sel41", "sel42", "sel43", "sel44", "sel302", "sel307", "sel31", "sel32", "sel33",
        "sel34", "sel38", "sel39"
    ]
    
    datasets = process_data(base_path, rec_ids)
    plot_ecg("sel30", base_path)

if __name__ == "__main__":
    main()