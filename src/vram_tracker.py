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

        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

        self.vram_usage[label] = max(
            self.vram_usage.get(label, 0.0),
            peak_vram_gb
        )

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