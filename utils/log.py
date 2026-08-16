import os
import shutil
import logging
import queue
import numpy as np
import torch
import wandb
from sklearn.metrics import precision_recall_curve
from typing import Optional, Dict, Any, Union


class WandbLogger:
    """
    Logger class for Weights & Biases (wandb) integration.
    """
    def __init__(self, project: str, is_used: bool, name: Optional[str] = None, entity: Optional[str] = None):
        self.is_used = is_used
        if self.is_used:
            wandb.init(project=project, name=name, entity=entity)
    
    def watch_model(self, model: torch.nn.Module):
        if self.is_used:
            wandb.watch(model)

    def log_hyperparams(self, params: Dict[str, Any]):
        if self.is_used:
            wandb.config.update(params)

    def log_metrics(self, metrics: Dict[str, float]):
        if self.is_used:
            wandb.log(metrics)

    def log(self, key: str, value: Any, round_idx: int):
        if self.is_used:
            wandb.log({key: value}, step=round_idx)

    def log_str(self, key: str, value: str):
        if self.is_used:
            wandb.log({key: value})

    def save_file(self, path: str):
        if self.is_used and os.path.exists(path):
            wandb.save(path)

    def finish(self):
        if self.is_used:
            wandb.finish()


class CheckpointSaver:
    """
    Handles saving and loading of model checkpoints, tracking the best-performing models.
    """
    def __init__(self, save_dir: str, metric_name: str, maximize_metric: bool = False, log: Optional[logging.Logger] = None):
        self.save_dir = save_dir
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val: Optional[float] = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log or logging.getLogger(__name__)

        self.log.info(f"Checkpoint saver initialized to {'maximize' if maximize_metric else 'minimize'} {metric_name}.")

    def is_best(self, metric_val: Optional[float]) -> bool:
        """Checks if the current metric value is the best seen so far."""
        if metric_val is None:
            return False
        if self.best_val is None:
            return True
        return (self.maximize_metric and metric_val >= self.best_val) or (not self.maximize_metric and metric_val <= self.best_val)

    def save(self, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, metric_val: float):
        """Saves the model checkpoint."""
        ckpt_dict = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        checkpoint_path = os.path.join(self.save_dir, "last.pth.tar")
        torch.save(ckpt_dict, checkpoint_path)
        
        if self.is_best(metric_val):
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, "best.pth.tar")
            shutil.copy(checkpoint_path, best_path)
            self.log.info(f"New best checkpoint saved at epoch {epoch}.")
            print(f"New best checkpoint saved at epoch {epoch}.")


def load_model_checkpoint(checkpoint_file: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Union[torch.nn.Module, tuple]:
    """Loads a model checkpoint."""
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        return model, optimizer
    return model


def get_save_dir(base_dir: str, training: bool, id_max: int = 500) -> str:
    """Creates a unique save directory."""
    subdir = "train" if training else "test"
    for uid in range(1, id_max):
        save_dir = os.path.join(base_dir, subdir, f"{subdir}-{uid:02d}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir
    raise RuntimeError("Maximum save directory limit reached.")


def count_parameters(model: torch.nn.Module) -> int:
    """Counts the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def thresh_max_f1(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int = 1):
    """
    Finds the best threshold for maximizing the F1-score using precision-recall curve analysis.
    """
    y_true = np.expand_dims(y_true, axis=1) if n_classes == 1 else y_true
    y_prob = np.expand_dims(y_prob, axis=1) if n_classes == 1 else y_prob

    best_thresh = []
    for i in range(n_classes):
        precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_prob[:, i])
        with np.errstate(divide='ignore', invalid='ignore'):
            fscore = np.divide(2 * precision * recall, precision + recall)
            fscore[np.isnan(fscore)] = 0

        if fscore.size == 0 or np.all(fscore == 0):
            best_thresh.append(0.5)
        else:
            best_thresh.append(thresholds[np.argmax(fscore)])

    best_thresh = np.array(best_thresh)
    if n_classes == 1:
        return float(best_thresh[0])
    return best_thresh