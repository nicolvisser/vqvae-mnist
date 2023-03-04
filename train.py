import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vqvae.models.foster import VectorQuantizedVariationalAutoencoder
import pandas as pd


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

    wandb.init(entity=cfg.wandb.entity,
               project=cfg.wandb.project,
               config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
               id=cfg.wandb.name if cfg.wandb.name != "None" else None,
               resume="allow")

    datasets = {
        'train': MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True),
        'validation': MNIST(root='data/', train=False, transform=transforms.ToTensor(), download=True)
    }

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=cfg.training.batch_size, shuffle=True),
        'validation': DataLoader(datasets['validation'], batch_size=cfg.training.batch_size, shuffle=True)
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VectorQuantizedVariationalAutoencoder(**cfg.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    def on_epoch(epoch, model, optimizer, logs):
        wandb.log(logs, step=epoch)

    def on_complete(epoch, model, optimizer, loss_logs):

        table_logs = {}
        if cfg.wandb.log_embeddings:
            print("Logging embeddings...")
            table_logs['embeddings'] = wandb.Table(
                data=create_dataframe_for_embeddings(model, cfg)
            )
        if cfg.wandb.log_validation_set:
            print("Logging latent space projections for validation set...")
            table_logs['validation_set'] = wandb.Table(
                data=create_dataframe_for_data(model, dataloaders['validation'], cfg, device)
            )
        wandb.log(table_logs)

        model_path = 'checkpoints/model.pth'
        torch.save(model.state_dict(), model_path)
        model_artifact = wandb.Artifact('model', type='model')
        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)

        optimizer_path = 'checkpoints/optimizer.pth'
        torch.save(optimizer.state_dict(), optimizer_path)
        optimizer_artifact = wandb.Artifact('optimizer', type='optimizer')
        optimizer_artifact.add_file(optimizer_path)
        wandb.log_artifact(optimizer_artifact)

    model.run_training(
        device,
        optimizer,
        dataloaders,
        **cfg.training,
        epoch_callback=on_epoch,
        complete_callback=on_complete
    )


def create_dataframe_for_embeddings(trained_model, cfg, ):
    columns = ["Embedding Index"] + ["Image"] + [f'z_{i}' for i in range(cfg.model.embedding_dim)]
    df_embeddings = pd.DataFrame(columns=columns)
    for k in range(cfg.model.num_embeddings):
        z = trained_model.vq.embedding[k].unsqueeze(0)
        x_hat = trained_model.decoder(z).reshape(1, 28, 28)
        img = wandb.Image(transforms.ToPILImage()(x_hat))
        z = z.cpu().numpy().squeeze()
        df_embeddings.loc[k] = [k, img, *z]
    return df_embeddings


def create_dataframe_for_data(trained_model, data_loader, cfg, device):
    columns = [f'z_{i}' for i in range(cfg.model.embedding_dim)] + ["Label", "Embedding_Index"]
    df_validation_set = pd.DataFrame(columns=columns)
    row_idx = 0
    for inputs, labels in data_loader:
        trained_model.eval()
        zs = trained_model.encoder(inputs.to(device))
        quantized, embedding_indices, _ = trained_model.vq.encode(zs.unsqueeze(0))

        zs = zs.detach().cpu().numpy().squeeze()
        labels = labels.cpu().numpy().astype(int)
        embedding_indices = embedding_indices.detach().cpu().numpy().squeeze().astype(int)
        for z, label, embedding_index in zip(zs, labels, embedding_indices):
            df_validation_set.loc[row_idx] = [*z, label, embedding_index]
            row_idx = row_idx + 1
    return df_validation_set


if __name__ == "__main__":
    run_experiment()
