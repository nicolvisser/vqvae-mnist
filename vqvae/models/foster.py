import torch
from torch import nn
from torchinfo import summary
from vqvae.vq import VectorQuantizer

from typing import Callable
from tqdm import tqdm


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


class VectorQuantizedVariationalAutoencoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, leaky_relu_negative_slope,
                 dropout_p):
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

    def run_training(self, device, optimizer, dataloaders, batch_size, learning_rate, num_epochs,
                     epoch_callback: Callable[[dict], None] = None,
                     complete_callback: Callable[[nn.Module], None] = None):

        assert "train" in dataloaders.keys()
        assert "validation" in dataloaders.keys()

        self.to(device)

        criterion = torch.nn.MSELoss()
        for epoch in tqdm(range(num_epochs)):
            logs = {}
            for phase in ['train', 'validation']:
                self.train() if phase == 'train' else self.eval()
                running_vq_loss = 0.0
                running_recon_loss = 0.0
                running_loss = 0.0
                running_perplexity = 0.0
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    outputs, vq_loss, perplexity = self.forward(inputs)
                    perplexity = perplexity
                    recon_loss = criterion(inputs, outputs)
                    loss = recon_loss + vq_loss
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    running_recon_loss += recon_loss.detach() * batch_size
                    running_vq_loss += vq_loss.detach() * batch_size
                    running_loss += loss.detach() * batch_size
                    running_perplexity += perplexity.detach() * batch_size
                epoch_recon_loss = running_recon_loss / len(dataloaders[phase].dataset)
                epoch_vq_loss = running_vq_loss / len(dataloaders[phase].dataset)
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_perplexity = running_perplexity / len(dataloaders[phase].dataset)
                logs[f"{phase} reconstruction loss"] = epoch_recon_loss.item()
                logs[f"{phase} vq loss"] = epoch_vq_loss.item()
                logs[f"{phase} perplexity"] = epoch_perplexity.item() / self.vq.num_embeddings
                logs[f"{phase} loss"] = epoch_loss.item()
            if epoch_callback is not None:
                epoch_callback(epoch, self, optimizer, logs)
        if complete_callback is not None:
            complete_callback(epoch, self, optimizer, logs)
