
from pathlib import Path
from typing import Tuple

import io
import os
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wfdb
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from starlette.responses import StreamingResponse

router = APIRouter(prefix="/api/segment", tags=["Segmentation"])

DATA_ROOT = Path(__file__).resolve().parents[2] / "datasets_physionet" / "data"


class Stats(BaseModel):
    average_qrs: float
    min_qrs: float
    max_qrs: float
    heart_rate: float


def _build_qrs_mask(prefix: str, lead_key: str, signal_len: int) -> np.ndarray:
    try:
        annotations = wfdb.rdann(prefix, extension=lead_key)
    except Exception:
        return np.zeros(signal_len, dtype=bool)

    mask = np.zeros(signal_len, dtype=bool)
    symbols = annotations.symbol
    samples = annotations.sample

    for i, sym in enumerate(symbols):
        if sym != "N":
            continue
        start = samples[i - 1] if i > 0 and symbols[i - 1] == "(" else samples[i]
        end = samples[i + 1] if i < len(symbols) - 1 and symbols[i + 1] == ")" else samples[i]
        start = max(0, start)
        end = min(signal_len - 1, end)
        if start <= end:
            mask[start : end + 1] = True

    return mask


def load_fragment(
    prefix: str,
    lead: str,
    start_sec: float,
    duration_sec: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    record = wfdb.rdrecord(prefix)
    lead_names = [n.lower() for n in record.sig_name]
    lead_key = lead.lower()
    if lead_key not in lead_names:
        raise HTTPException(status_code=404, detail=f"Lead '{lead}' not found in record")

    lead_idx = lead_names.index(lead_key)
    full_signal = record.p_signal[:, lead_idx]
    fs = float(record.fs)

    start = int(start_sec * fs)
    end = start + int(duration_sec * fs)
    start = max(0, start)
    end = min(len(full_signal), end)

    if start >= end:
        raise HTTPException(status_code=400, detail="Selected window is empty")

    mask_full = _build_qrs_mask(prefix, lead_key, len(full_signal))
    return full_signal[start:end], mask_full[start:end], fs


@router.get("/record/{record_id}/image", response_class=StreamingResponse)
def get_image(
    record_id: int,
    lead: str = Query(..., min_length=1),
    start_sec: float = Query(0.0, ge=0.0),
    duration_sec: float = Query(8.0, gt=0.0, le=20.0),
):
    prefix = str(DATA_ROOT / str(record_id))
    if not Path(prefix + ".hea").is_file():
        raise HTTPException(status_code=404, detail=f"Missing record {record_id}")

    try:
        signal, mask, fs = load_fragment(prefix, lead, start_sec, duration_sec)

        fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
        t = np.linspace(start_sec, start_sec + duration_sec, len(signal), endpoint=False)
        ax.plot(t, signal, linewidth=1.2)
        ax.fill_between(t, float(signal.min()), float(signal.max()), where=mask, alpha=0.3)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude [mV]")
        ax.set_title(f"Rec {record_id} • {lead}", fontsize=12)
        fig.tight_layout(pad=1.0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@router.get("/record/{record_id}/stats", response_model=Stats)
def get_stats(
    record_id: int,
    lead: str = Query(..., min_length=1),
    start_sec: float = Query(0.0, ge=0.0),
    duration_sec: float = Query(8.0, gt=0.0, le=20.0),
):
    prefix = str(DATA_ROOT / str(record_id))
    if not Path(prefix + ".hea").is_file():
        raise HTTPException(status_code=404, detail=f"Missing record {record_id}")

    try:
        _, mask, fs = load_fragment(prefix, lead, start_sec, duration_sec)

        binary = mask.astype(np.int8)
        diff = np.diff(binary, prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        if len(starts) == 0 or len(ends) == 0:
            return Stats(average_qrs=0.0, min_qrs=0.0, max_qrs=0.0, heart_rate=0.0)

        durations = (ends - starts) / fs
        avg = float(np.mean(durations))
        mn = float(np.min(durations))
        mx = float(np.max(durations))
        hr = len(starts) / duration_sec * 60.0

        return Stats(
            average_qrs=round(avg, 3),
            min_qrs=round(mn, 3),
            max_qrs=round(mx, 3),
            heart_rate=round(hr, 1),
        )

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
