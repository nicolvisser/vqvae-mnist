import torch
from torch import nn
from torchinfo import summary
from vqvae.vq import VectorQuantizer


class Encoder(nn.Module):
    def __init__(self, z_dim, leaky_relu_negative_slope=0.3, dropout_p=0.25):
        super().__init__()

        self.conv2d_0 = nn.Conv2d(1, 32, 3, 1, 1)
        self.batch_norm_0 = nn.BatchNorm2d(32)
        self.conv2d_1 = nn.Conv2d(32, 64, 3, 2, 1)
        self.batch_norm_1 = nn.BatchNorm2d(64)
        self.conv2d_2 = nn.Conv2d(64, 64, 3, 2, 1)
        self.batch_norm_2 = nn.BatchNorm2d(64)
        self.conv2d_3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.batch_norm_3 = nn.BatchNorm2d(64)
        self.linear = nn.Linear(in_features=64 * 7 * 7, out_features=z_dim)
        self.leakyReLU = nn.LeakyReLU(leaky_relu_negative_slope)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.leakyReLU(self.dropout(self.batch_norm_0(self.conv2d_0(x))))
        x = self.leakyReLU(self.dropout(self.batch_norm_1(self.conv2d_1(x))))
        x = self.leakyReLU(self.dropout(self.batch_norm_2(self.conv2d_2(x))))
        x = self.leakyReLU(self.dropout(self.batch_norm_3(self.conv2d_3(x))))
        x = self.linear(x.view(-1, 64 * 7 * 7))
        return x

    def summary(self, input_size=None, input_data=None):
        return summary(self, input_size=input_size, input_data=input_data)


class Decoder(nn.Module):
    def __init__(self, z_dim, leaky_relu_negative_slope=0.3, dropout_p=0.25):
        super().__init__()

        self.linear = nn.Linear(in_features=z_dim, out_features=64 * 7 * 7)
        self.batchNorm_0 = nn.BatchNorm1d(64 * 7 * 7)

        self.convTranspose2d_0 = nn.ConvTranspose2d(64, 64, 3, 1, 1, 0)
        self.batchNorm_1 = nn.BatchNorm2d(64)

        self.convTranspose2d_1 = nn.ConvTranspose2d(64, 64, 3, 2, 1, 1)
        self.batchNorm_2 = nn.BatchNorm2d(64)

        self.convTranspose2d_2 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.batchNorm_3 = nn.BatchNorm2d(32)

        self.convTranspose2d_3 = nn.ConvTranspose2d(32, 1, 3, 1, 1, 0)

        self.dropout = nn.Dropout(dropout_p)
        self.leakyReLU = nn.LeakyReLU(leaky_relu_negative_slope)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leakyReLU(self.dropout(self.batchNorm_0(self.linear(x))))
        x = x.view(-1, 64, 7, 7)
        x = self.leakyReLU(self.dropout(self.batchNorm_1(self.convTranspose2d_0(x))))
        x = self.leakyReLU(self.dropout(self.batchNorm_2(self.convTranspose2d_1(x))))
        x = self.leakyReLU(self.dropout(self.batchNorm_3(self.convTranspose2d_2(x))))
        x = self.sigmoid(self.convTranspose2d_3(x))

        return x

    def summary(self, input_size=None, input_data=None):
        return summary(self, input_size=input_size, input_data=input_data)


class Autoencoder(nn.Module):
    def __init__(self, z_dim, leaky_relu_negative_slope=0.3, dropout_p=0.25):
        super().__init__()
        self.encoder = Encoder(z_dim=z_dim,
                               leaky_relu_negative_slope=leaky_relu_negative_slope,
                               dropout_p=dropout_p)
        self.decoder = Decoder(z_dim=z_dim,
                               leaky_relu_negative_slope=leaky_relu_negative_slope,
                               dropout_p=dropout_p)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def summary(self, input_size=None, input_data=None):
        return summary(self, input_size=input_size, input_data=input_data)


class VariationalEncoder(nn.Module):
    def __init__(self, z_dim, leaky_relu_negative_slope=0.3, dropout_p=0.25):
        super().__init__()
        self.z_dim = z_dim
        self.mu_encoder = Encoder(z_dim=z_dim,
                                  leaky_relu_negative_slope=leaky_relu_negative_slope,
                                  dropout_p=dropout_p)
        self.log_var_encoder = Encoder(z_dim=z_dim,
                                       leaky_relu_negative_slope=leaky_relu_negative_slope,
                                       dropout_p=dropout_p)

    def forward(self, x):
        batch_size = x.shape[0] if x.dim() == 4 else 1
        mu = self.mu_encoder(x)
        log_var = self.log_var_encoder(x)
        z = mu + torch.exp(log_var / 2) * torch.randn_like(mu).to(x.device)
        kl_loss = -0.5 * torch.sum(1 + log_var - torch.square(mu) - torch.exp(log_var)) / batch_size
        return z, kl_loss

    def summary(self, input_size=None, input_data=None):
        return summary(self, input_size=input_size, input_data=input_data)


class VariationalAutoencoder(nn.Module):
    def __init__(self, z_dim, leaky_relu_negative_slope=0.3, dropout_p=0.25):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = VariationalEncoder(z_dim=z_dim,
                                          leaky_relu_negative_slope=leaky_relu_negative_slope,
                                          dropout_p=dropout_p)
        self.decoder = Decoder(z_dim=z_dim,
                               leaky_relu_negative_slope=leaky_relu_negative_slope,
                               dropout_p=dropout_p)

    def forward(self, x):
        z, kl_loss = self.encoder(x)
        x_hat = self.decoder(z)
        recon_loss = torch.nn.functional.mse_loss(x, x_hat)
        return x, recon_loss, kl_loss

    def summary(self, input_size: object = None, input_data: object = None) -> object:
        return summary(self, input_size=input_size, input_data=input_data)


class VectorQuantizedVariationalAutoencoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, leaky_relu_negative_slope=0.3,
                 dropout_p=0.25):
        super().__init__()

        self.encoder = Encoder(z_dim=embedding_dim,
                               leaky_relu_negative_slope=leaky_relu_negative_slope,
                               dropout_p=dropout_p)

        self.vq = VectorQuantizer(num_embeddings=num_embeddings,
                                  embedding_dim=embedding_dim,
                                  commitment_cost=commitment_cost,
                                  dim=-1,
                                  decay=decay)

        self.decoder = Decoder(z_dim=embedding_dim,
                               leaky_relu_negative_slope=leaky_relu_negative_slope,
                               dropout_p=dropout_p)

    def forward(self, x):
        x = self.encoder(x)
        x, vq_loss, perplexity = self.vq(x)
        x = self.decoder(x)
        return x, vq_loss, perplexity

    def encode(self, x):
        z_e = self.encoder(x)
        _, encodings_indices, _ = self.vq.encode(z_e)
        return encodings_indices.squeeze(-1)

    def summary(self, input_size: object = None, input_data: object = None) -> object:
        return summary(self, input_size=input_size, input_data=input_data)
