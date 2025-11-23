from dataclasses import dataclass
from typing import Dict

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

@dataclass
class TrainerConfig:
    device: str
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    gradient_accumulation_steps: int
    log_every_n_steps: int
    save_every_n_steps: int

class Trainer:
    def __init__(
        self,
        cfg: TrainerConfig,
        model: torch.nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        logger,
    ):
        self.cfg = cfg
        self.logger = logger
        self.accelerator = Accelerator()
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        (
            self.model,
            self.train_loader,
            self.eval_loader,
            self.optimizer,
        ) = self.accelerator.prepare(
            self.model,
            self.train_loader,
            self.eval_loader,
            self.optimizer,
        )

    def train(self):
        global_step = 0
        for epoch in range(self.cfg.epochs):
            self.model.train()
            self.logger.log(f"Epoch {epoch + 1}/{self.cfg.epochs}")
            epoch_loss = 0.0

            for step, batch in enumerate(tqdm(self.train_loader, desc="Training")):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs["loss"]
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.reset_grad()

                epoch_loss += loss.item()
                global_step += 1

                if global_step % self.cfg.log_every_n_steps == 0:
                    avg_loss = epoch_loss / (step + 1)
                    self.logger.log_metrics(
                        {"train_loss": avg_loss},
                        step=global_step,
                    )

            eval_metrics = self.evaluate()
            self.logger.log(
                f"Epoch {epoch + 1} done. Train loss={epoch_loss/(step+1):.4f}, Eval accuracy={eval_metrics['accuracy']:.4f}"
            )
            self.logger.log_metrics(
                {"eval_accuracy": eval_metrics["accuracy"]},
                step=global_step,
            )

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=None,
                )
                logits = outputs["logits"]
                preds = logits.argmax(dim=-1)
                labels = batch["labels"]
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy}
