from dataclasses import dataclass
from typing import Optional

from rich.console import Console
import wandb

console = Console()

@dataclass
class LoggerConfig:
    use_wandb: bool
    project: Optional[str] = None
    run_name: Optional[str] = None

class Logger:
    def __init__(self, cfg: LoggerConfig):
        self.cfg = cfg
        self.wandb_run = None

        if self.cfg.use_wandb:
            self.wandb_run = wandb.init(
                project=self.cfg.project,
                name=self.cfg.run_name,
                config={},
            )

    def log(self, msg: str) -> None:
        console.log(msg)

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        if self.wandb_run is not None:
            wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.wandb_run is not None:
            wandb.finish()
