import os
import wfdb
import numpy as np
from sklearn.model_selection import train_test_split

def load_rec(rec_id, base_dir):
    try:
        path = os.path.join(base_dir, rec_id)
        rec = wfdb.rdrecord(path)
        ann = wfdb.rdann(path, 'q1c')
        start, end = ann.sample[0], ann.sample[-1]
        if rec.p_signal.shape[1] >= 2:
            signals = [rec.p_signal[start:end+1, 0], rec.p_signal[start:end+1, 1]]
        else:
            signals = [rec.p_signal[start:end+1, 0]]
        mask = np.zeros(end - start + 1, dtype=np.uint8)
        for i in range(len(ann.sample)):
            if ann.symbol[i] == "N":
                peak = ann.sample[i] - start
                on = ann.sample[i-1] - start if i > 0 and ann.symbol[i-1] == "(" else peak
                off = ann.sample[i+1] - start if i < len(ann.sample)-1 and ann.symbol[i+1] == ")" else peak
                mask[on:off+1] = 1
        return signals, mask, rec.fs
    except Exception as e:
        print("Error loading", rec_id, ":", e)
        return None, None, None

def segment_signal(sig, mask, seg_len=5000):
    num_segs = len(sig) // seg_len
    segs = [sig[i*seg_len:(i+1)*seg_len] for i in range(num_segs)]
    seg_masks = [mask[i*seg_len:(i+1)*seg_len] for i in range(num_segs)]
    return segs, seg_masks

def main():
    base_dir = 'qt/qt-database-1.0.0/'
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
    
    all_signals = []
    all_masks = []
    for rec in rec_ids:
        sigs, mask, fs = load_rec(rec, base_dir)
        if sigs is None:
            continue
        for s in sigs:
            all_signals.append(s)
            all_masks.append(mask)
    
    if not all_signals:
        print("No records loaded.")
        return

    print("Loaded", len(all_signals), "signals")
    
    segments = []
    seg_masks = []
    for s, m in zip(all_signals, all_masks):
        segs, masks = segment_signal(s, m, seg_len=5000)
        segments.extend(segs)
        seg_masks.extend(masks)
    
    print("Total segments:", len(segments))
    
    indices = np.arange(len(segments))
    train_val, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
    val_ratio = 0.1 / (1 - 0.1)
    train_idx, val_idx = train_test_split(train_val, test_size=val_ratio, random_state=42)
    
    train_segs = [segments[i] for i in train_idx]
    train_masks = [seg_masks[i] for i in train_idx]
    val_segs = [segments[i] for i in val_idx]
    val_masks = [seg_masks[i] for i in val_idx]
    test_segs = [segments[i] for i in test_idx]
    test_masks = [seg_masks[i] for i in test_idx]
    
    print("Train segments:", len(train_segs))
    print("Validation segments:", len(val_segs))
    print("Test segments:", len(test_segs))
    
    out_dir = "output_qt"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "train_signals.npy"), np.array(train_segs))
    np.save(os.path.join(out_dir, "train_masks.npy"), np.array(train_masks))
    np.save(os.path.join(out_dir, "val_signals.npy"), np.array(val_segs))
    np.save(os.path.join(out_dir, "val_masks.npy"), np.array(val_masks))
    np.save(os.path.join(out_dir, "test_signals.npy"), np.array(test_segs))
    np.save(os.path.join(out_dir, "test_masks.npy"), np.array(test_masks))
    print("Data saved in folder 'output_qt'.")

if __name__ == "__main__":
    main()
