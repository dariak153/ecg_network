import os
import numpy as np
import torch
import wfdb

from .resnet_qrs import ResNetQRS


class QRSDetector:
    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_file_path = model_path or os.path.join(os.path.dirname(__file__), "resnet_model.pt")

        self.model = ResNetQRS().to(self.device)
        self._load_model_weights(model_file_path)
        self.model.eval()

    def _load_model_weights(self, model_file_path: str):
        state_dict = torch.load(model_file_path, map_location=self.device)
        adjusted_state_dict = {
            key.replace("initial.", "init.")
               .replace("layer1.", "l1.")
               .replace("layer2.", "l2.")
               .replace("layer3.", "l3."): value
            for key, value in state_dict.items()
        }
        self.model.load_state_dict(adjusted_state_dict, strict=False)

    def predict_qrs_mask(self, ecg_signal: np.ndarray) -> np.ndarray:
        signal_array = ecg_signal.astype(np.float32).reshape(1, -1, 1)
        input_tensor = torch.from_numpy(signal_array).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        return output.cpu().numpy().squeeze(-1)[0]

    def load_signal_with_mask(
        self,
        record_prefix: str,
        lead_name: str,
        start_sec: float,
        duration_sec: float,
        return_sampling_rate: bool = False
    ):
        sampling_rate = 500.0

        record = wfdb.rdrecord(record_prefix)
        lead_index = [name.lower() for name in record.sig_name].index(lead_name.lower())
        full_signal = record.p_signal[:, lead_index]

        start_sample = int(start_sec * sampling_rate)
        end_sample = start_sample + int(duration_sec * sampling_rate)

        signal_segment = full_signal[start_sample:end_sample]
        qrs_mask_full = self.predict_qrs_mask(full_signal)
        qrs_mask_segment = qrs_mask_full[start_sample:end_sample]

        if return_sampling_rate:
            return signal_segment, qrs_mask_segment, sampling_rate
        return signal_segment, qrs_mask_segment
