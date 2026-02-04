"""
MLSAC Inference Pipeline for Hugging Face.
Expects inputs = {"data": "<base64 raw float32 bytes>"} (seq_len * 8 * 4 bytes).
Returns [{"probability": float}] for compatibility with MLSAC plugin.
"""

import base64
import json
import os
from typing import Any, List

import numpy as np
import torch

try:
    from pipeline import Pipeline  # HF Pipeline base
except ImportError:
    Pipeline = object


class MLSACPipeline(Pipeline if Pipeline != object else object):
    def __init__(self, model=None, **kwargs):
        if Pipeline != object:
            super().__init__(model=model, **kwargs)
        self._model = None
        self._config = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        import torch.nn as nn

        class MLSACModel(nn.Module):
            def __init__(self, seq_len=40, n_features=8, hidden=64):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.fc = nn.Sequential(
                    nn.Linear(64, hidden),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden, 1),
                )

            def forward(self, x):
                x = x.transpose(1, 2)
                x = self.conv(x)
                x = x.squeeze(-1)
                return self.fc(x).squeeze(-1)

        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                self._config = json.load(f)
        else:
            self._config = {"seq_len": 40, "n_features": 8}
        seq_len = self._config.get("seq_len", 40)
        n_features = self._config.get("n_features", 8)
        self._model = MLSACModel(seq_len=seq_len, n_features=n_features).to(self._device)
        bin_path = os.path.join(os.path.dirname(__file__), "pytorch_model.bin")
        if os.path.isfile(bin_path):
            state = torch.load(bin_path, map_location=self._device)
            if isinstance(state, dict) and "state_dict" in state:
                self._model.load_state_dict(state["state_dict"])
            elif isinstance(state, dict):
                self._model.load_state_dict(state)
        self._model.eval()

    def preprocess(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, dict) and "data" in inputs:
            b64 = inputs["data"]
        elif isinstance(inputs, str):
            b64 = inputs
        else:
            raise ValueError("inputs must be {'data': base64_string} or base64 string")
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.float32)
        seq_len = self._config.get("seq_len", 40)
        n_features = self._config.get("n_features", 8)
        arr = arr.reshape(-1, seq_len, n_features)
        return torch.from_numpy(arr).float().to(self._device)

    def _forward(self, model_input: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self._model(model_input)
        return torch.sigmoid(logits)

    def postprocess(self, model_output: torch.Tensor) -> List[dict]:
        prob = model_output.cpu().float().item()
        return [{"probability": prob}]

    def __call__(self, inputs: Any, **kwargs) -> List[dict]:
        if isinstance(inputs, list):
            out = []
            for inp in inputs:
                x = self.preprocess(inp)
                y = self._forward(x)
                out.extend(self.postprocess(y))
            return out
        x = self.preprocess(inputs)
        y = self._forward(x)
        return self.postprocess(y)


def pipeline(task: str = None, model: str = None, **kwargs):
    return MLSACPipeline(model=model, **kwargs)
