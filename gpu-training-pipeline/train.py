import hydra
from omegaconf import DictConfig, OmegaConf

from src.dataloaders.text_classification import DataConfig, get_dataloaders
from src.models.text_classification_model import (
    TextClassificationConfig,
    TextClassificationModel,
)
from src.trainers.trainer import TrainerConfig, Trainer
from src.utils.logging_utils import Logger, LoggerConfig
from src.utils.seed import set_seed


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Set deterministic behavior
    set_seed(cfg.seed)

    # Logger
    logger_cfg = LoggerConfig(
        use_wandb=cfg.logging.wandb,
        project=cfg.logging.project,
        run_name=cfg.logging.run_name,
    )
    logger = Logger(logger_cfg)

    # Data
    data_cfg = DataConfig(
        name=cfg.data.name,
        text_field=cfg.data.text_field,
        label_field=cfg.data.label_field,
        max_length=cfg.data.max_length,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        subset_ratio=cfg.data.subset_ratio,
    )
    model_name = cfg.model.model_name

    train_loader, eval_loader = get_dataloaders(data_cfg, model_name=model_name)

    # Model
    model_cfg = TextClassificationConfig(
        model_name=model_name,
        num_labels=cfg.model.num_labels,
        freeze_base=cfg.model.freeze_base,
    )
    model = TextClassificationModel(model_cfg)

    # Trainer
    trainer_cfg = TrainerConfig(
        device=cfg.trainer.device,
        epochs=cfg.trainer.epochs,
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        warmup_ratio=cfg.trainer.warmup_ratio,
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        save_every_n_steps=cfg.trainer.save_every_n_steps,
    )

    trainer = Trainer(
        cfg=trainer_cfg,
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        logger=logger,
    )

    trainer.train()
    logger.finish()


if __name__ == "__main__":
    main()
