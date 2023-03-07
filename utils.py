import pandas as pd
from torchvision import transforms as transforms
from tqdm import tqdm


def create_dataframe_for_embeddings(model, cfg, ):
    columns = [f'z_{i}' for i in range(cfg.vq.embedding_dim)] + ['Image']
    df_embeddings = pd.DataFrame(columns=columns)
    for k in range(cfg.vq.num_embeddings):
        z = model.vq.embedding[k].unsqueeze(0)
        x_hat = model.decoder(z).reshape(1, 28, 28)
        img = transforms.ToPILImage()(x_hat)
        z = z.cpu().numpy().squeeze()
        df_embeddings.loc[k] = [*z, img]
    return df_embeddings


def create_dataframe_for_latent_projections(model, data_loader, cfg):
    columns = [f'z_{i}' for i in range(cfg.vq.embedding_dim)] + ["Label", "Embedding Index"]
    df_latent_projection = pd.DataFrame(columns=columns)
    row_idx = 0
    for inputs, labels in tqdm(data_loader):
        zs = model.encoder(inputs)
        quantized, embedding_indices, _ = model.vq.encode(zs.unsqueeze(0))
        zs = zs.detach().cpu().numpy().squeeze()
        labels = labels.cpu().numpy().astype(int)
        embedding_indices = embedding_indices.detach().cpu().numpy().squeeze().astype(int)
        for z, label, embedding_index in zip(zs, labels, embedding_indices):
            df_latent_projection.loc[row_idx] = [*z, label, embedding_index]
            row_idx = row_idx + 1
    return df_latent_projection
