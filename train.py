import hydra
import pytorch_lightning as pl
import pytorch_lightning.loggers as loggers
import torchvision.transforms as transforms
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from utils import create_dataframe_for_embeddings, create_dataframe_for_latent_projections
from vqvae.foster.vqvae import VectorQuantizedVariationalAutoencoder


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    """
    To test the hydra configuration, override cfg.view_config to True
    You can do this from command line with:
        python train.py view_config=True
    """

    if cfg.debug:
        print(OmegaConf.to_yaml(cfg))
        return

    wandb.login()

    wandb_logger = loggers.WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name if cfg.wandb.name != "None" else None,
        log_model="all",
        save_dir="checkpoints/"
    )

    train_dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
    val_dataset = MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True,
                                  num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, drop_last=True,
                                num_workers=4)

    vqvae = VectorQuantizedVariationalAutoencoder(cfg.model)

    checkpoint_callback = ModelCheckpoint(monitor="validation loss", mode="min")  # save best model

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=cfg.training.num_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(
        model=vqvae,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    vqvae.eval()

    if cfg.wandb.log_embeddings:
        print("logging embeddings...")
        df_embeddings = create_dataframe_for_embeddings(vqvae, cfg.model)
        df_embeddings['Image'] = df_embeddings['Image'].map(lambda x: wandb.Image(x))
        wandb.log({'embeddings': wandb.Table(dataframe=df_embeddings)})

    if cfg.wandb.log_latent_projections:
        print("logging latent projections...")
        df_train_data_latent_projection = create_dataframe_for_latent_projections(vqvae, train_dataloader, cfg.model)
        wandb.log({'train data latent projection': wandb.Table(dataframe=df_train_data_latent_projection)})


if __name__ == "__main__":
    run_experiment()
