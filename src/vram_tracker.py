import torch
from transformers import Trainer

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

        tempMetrics = {}

        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        allocated_vram = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_vram = torch.cuda.memory_reserved() / (1024 ** 3)

        tempMetrics["peak_vram_gb"] = peak_vram_gb
        tempMetrics["allocated_vram"] = allocated_vram
        tempMetrics["reserved_vram"] = reserved_vram

        if label not in self.vram_usage:
            self.vram_usage[label] = {
                "peak_vram_gb": 0.0,
                "allocated_vram": 0.0,
                "reserved_vram": 0.0
            }

        for key, value in tempMetrics.items():
            self.vram_usage[label][key] = max(
                self.vram_usage[label][key],
                value
            )

        self.vram_usage[label] = tempMetrics

        self.resetVRamMemoryStat()

        return self.vram_usage[label]


class VRAMTrackingTrainer(Trainer):

    def __init__(self, *args, vram_tracker=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.vram_tracker = vram_tracker

        # Intercept Accelerator backward propagation
        self.original_backward = self.accelerator.backward
        self.accelerator.backward = self.tracked_backward

    def compute_loss(self, model, inputs, *args, **kwargs):

        # Only track training forward passes
        if model.training and torch.cuda.is_available():

            self.vram_tracker.resetVRamMemoryStat()

            # Forward propagation
            loss = super().compute_loss(
                model, inputs, *args, **kwargs
            )

            self.vram_tracker.recordPeakVRam(
                f"Forward Pass"
            )

            return loss

        return super().compute_loss(
            model, inputs, *args, **kwargs
        )

    def tracked_backward(self, loss, *args, **kwargs):

        if torch.cuda.is_available():

            self.vram_tracker.resetVRamMemoryStat()

            # Backward propagation
            self.original_backward(loss, *args, **kwargs)

            self.vram_tracker.recordPeakVRam(
                f"Backward Propagation"
            )

        else:
            self.original_backward(loss, *args, **kwargs)