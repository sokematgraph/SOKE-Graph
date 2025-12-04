"""
device_manager.py

GPU/CPU device management for PyTorch operations.

This module provides a singleton :class:`DeviceManager` that determines
and caches the optimal compute device (CUDA GPU or CPU) for torch operations.

Classes:
- DeviceManager: Singleton manager for torch device selection
"""

import torch

class DeviceManager:
    _device = None

    @classmethod
    def get_device(cls):
        """Return the global torch device (GPU if available, else CPU)."""
        if cls._device is None:
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[INFO] Using device: {cls._device}")
        return cls._device
