import torch

class VRamTracker:
    def __init__(self):
        self.vram_usage = {}

    def resetVRamMemoryStat(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def recordPeakVRam(self, label):
        if not torch.cuda.is_available():
            return 0.0

        torch.cuda.synchronize()
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        # print(f"{label}: {peak_vram_gb:.2f} GB / 15.00 GB")

        self.vram_usage[label] = peak_vram_gb
        self.resetVRamMemoryStat()
        return peak_vram_gb