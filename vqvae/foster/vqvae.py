import pytorch_lightning as pl
import torch
import torch.nn.functional as f

from vqvae import VectorQuantizer
from vqvae.foster.decoder import Decoder
from vqvae.foster.encoder import Encoder


class VectorQuantizedVariationalAutoencoder(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.encoder = Encoder(**cfg.encoder)
        self.vq = VectorQuantizer(**cfg.vq)
        self.decoder = Decoder(**cfg.decoder)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        z_e = self.encoder(x)
        z_q, vq_loss, perplexity = self.vq(z_e)
        x_hat = self.decoder(z_q)
        recon_loss = f.mse_loss(x_hat, x)
        loss = recon_loss + vq_loss
        self.log('train loss', loss)
        self.log('train reconstruction loss', recon_loss)
        self.log('train vq loss', vq_loss)
        self.log('train perplexity', perplexity / self.cfg.vq.num_embeddings)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        z_e = self.encoder(x)
        z_q, vq_loss, perplexity = self.vq(z_e)
        x_hat = self.decoder(z_q)
        recon_loss = f.mse_loss(x_hat, x)
        loss = recon_loss + vq_loss
        self.log('validation loss', loss)
        self.log('validation reconstruction loss', recon_loss)
        self.log('validation vq loss', vq_loss)
        self.log('validation perplexity', perplexity / self.cfg.vq.num_embeddings)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optimizer.learning_rate)
        return optimizer
