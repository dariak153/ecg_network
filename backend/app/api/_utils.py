
from pathlib import Path
from typing import Tuple
import wfdb
import numpy as np

API_DIR = Path(__file__).resolve().parent                  
BACKEND_ROOT = API_DIR.parent.parent                    
DATA_ROOT = BACKEND_ROOT / "datasets_physionet" / "data"


def load_signal_and_mask(
    record_id: int,
    lead_name: str,
    start_sec: float,
    duration_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    record_stem = str(record_id)
    base_path = DATA_ROOT / record_stem

    header_path = base_path.with_suffix(".hea")
    data_path = base_path.with_suffix(".dat")
    if not header_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(
            f"Missing WFDB files for record {record_id}: "
            f"{header_path} or {data_path} not found."
        )

    record = wfdb.rdrecord(str(base_path))
    lead_names = [name.lower() for name in record.sig_name]
    lead_key = lead_name.lower()
    if lead_key not in lead_names:
        raise ValueError(f"Unknown lead '{lead_name}'. Available: {record.sig_name}")

    lead_idx = lead_names.index(lead_key)
    full_signal = record.p_signal[:, lead_idx]

    try:
        annotations = wfdb.rdann(str(base_path), extension=lead_key)
        mask_full = np.zeros(full_signal.shape[0], dtype=np.uint8)

        for sample_idx, symbol in zip(annotations.sample, annotations.symbol):
            if symbol == "N":
                mask_full[sample_idx] = 1
    except Exception:
        mask_full = np.zeros(full_signal.shape[0], dtype=np.uint8)

    fs = int(record.fs)
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive.")
    if start_sec < 0:
        raise ValueError("start_sec must be non-negative.")

    start_sample = int(start_sec * fs)
    end_sample = start_sample + int(duration_sec * fs)

    start_sample = max(0, start_sample)
    end_sample = min(len(full_signal), end_sample)
    if start_sample >= end_sample:
        raise ValueError("Selected window is empty.")

    signal_segment = full_signal[start_sample:end_sample]
    mask_segment = mask_full[start_sample:end_sample]

    return signal_segment, mask_segment
