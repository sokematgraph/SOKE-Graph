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
