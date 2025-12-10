# utils/logger.py
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

class BaseLogger:
    """Abstract base class for loggers."""
    def log_scalar(self, tag, value, step):
        raise NotImplementedError
    def log_hyperparams(self, params):
        raise NotImplementedError
    def close(self):
        pass

class ConsoleLogger(BaseLogger):
    """Logs metrics to the console."""
    def __init__(self, log_dir):
        print("Using Console Logger.")
    def log_scalar(self, tag, value, step):
        print(f"Step: {step} | {tag}: {value:.4f}")
    def log_hyperparams(self, params):
        print("\n--- Hyperparameters ---")
        for key, value in params.items():
            print(f"{key}: {value}")
        print("-----------------------\n")
    def close(self):
        pass

class TensorBoardLogger(BaseLogger):
    """Logs metrics to a TensorBoard file for visualization."""
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
        print(f"To view, run: tensorboard --logdir {os.path.dirname(log_dir)}")
    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
    def log_hyperparams(self, hparams):
        # A simple text-based hyperparameter log is robust and easy
        with open(os.path.join(self.writer.log_dir, "hyperparams.txt"), 'w') as f:
            for key, value in hparams.items():
                f.write(f"{key}: {value}\n")
    def close(self):
        self.writer.close()

def get_logger(logger_type="tensorboard", log_dir="logs/"):
    """Factory function to get the desired logger instance."""
    if logger_type.lower() == "tensorboard":
        return TensorBoardLogger(log_dir)
    elif logger_type.lower() == "console":
        return ConsoleLogger(log_dir)
    else:
        raise ValueError(f"Unknown logger type: {logger_type}")