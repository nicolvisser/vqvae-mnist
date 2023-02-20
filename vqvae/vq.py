import torch
from torch import nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, dim=-1, decay=None, epsilon=1e-5):
        super().__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost
        self._dim = dim

        init_bound = 1 / self._num_embeddings
        embedding = torch.Tensor(num_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)

        self._use_ema = (decay is not None) and (decay != 0.0)
        if self._use_ema:
            self._decay = decay
            self._epsilon = epsilon
            self.register_buffer("_ema_count", torch.zeros(num_embeddings))
            self.register_buffer("_ema_weight", self.embedding.clone())

    def encode(self, x):
        """Encodes an input tensor to its quantized and discrete versions

        Args:
            x (torch.Tensor): an input tensor with embedding of length `embedding_dim` in axis position `dim`

        Returns:
            quantized (torch.Tensor): an output tensor with the same shape as the input tensor. the outputs are
            snapped to the values of the closes embedding vector in the codebook.
            encodings_indices (torch.Tensor): an output tensor with the encoded indices in the embedding dimension
            encodings_one_hot (torch.Tensor): an output tensor with the encoded embedding vector in one-hot format
            in the embedding dimension

        Example:
            if
                dim = -4
                embedding_dim = E
                num_embeddings = N
            and
                x.shape = (B,E,C,H,W)

            then
                quantized.shape = (B,E,C,H,W)
                encodings_indices.shape = (B,1,C,H,W)
                encodings_one_hot.shape = (B,N,C,H,W)
        """

        assert x.shape[self._dim] == self._embedding_dim, f"Embedding dimension of input, {x.shape[self._dim]}, does " \
                                                          f"not match embedding_dim, {self._embedding_dim}"
        x = torch.moveaxis(x, self._dim, -1).contiguous()
        x_flat = x.detach().view(-1, self._embedding_dim)
        quantized_flat, encodings_indices_flat, encodings_one_hot_flat = self._encode(x_flat)
        quantized = torch.moveaxis(quantized_flat.view_as(x), -1, self._dim).contiguous()
        encodings_indices = torch.moveaxis(encodings_indices_flat.view([*x.shape[:-1], 1]), -1, self._dim).contiguous()
        encodings_one_hot = torch.moveaxis(encodings_one_hot_flat.view([*x.shape[:-1], self._num_embeddings]), -1,
                                           self._dim).contiguous()
        return quantized, encodings_indices, encodings_one_hot

    def _encode(self, x_flat):
        """Encodes a flat input tensor to its quantized and discrete versions

        Args:
            x_flat (torch.Tensor): an input tensor with embedding dimension of length `embedding_dim` in the last axis
            and all the other dimensions flattened into the first

        Returns:
            quantized (torch.Tensor): an output tensor with the same shape as the input tensor. the outputs are
            snapped to the values of the closes embedding vector in the codebook.
            encodings_indices (torch.Tensor): an output tensor with the encoded indices in the last dimension
            encodings_one_hot (torch.Tensor): an output tensor with the encoded embedding vector in one-hot format
            in the last dimension

        """
        distances = torch.addmm(torch.sum(x_flat ** 2, dim=1, keepdim=True) +
                                torch.sum(self.embedding ** 2, dim=1),
                                x_flat,
                                self.embedding.t(),
                                beta=1.0,
                                alpha=-2.0)
        encodings_indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(encodings_indices, self.embedding)
        encodings_one_hot = F.one_hot(encodings_indices, self._num_embeddings).float()
        return quantized, encodings_indices, encodings_one_hot

    def forward(self, x):
        """Forward pass of the VQ layer

        Args:
            x (torch.Tensor): an input tensor with embedding of length `embedding_dim` in axis position `dim`

        """

        x = torch.moveaxis(x, self._dim, -1).contiguous()  # Move embedding axis to the end and flatten other dimensions
        x_flat = x.detach().view(-1, self._embedding_dim)
        quantized_flat, _, encodings_one_hot_flat = self._encode(x_flat)
        quantized = quantized_flat.view_as(x)
        e_latent_loss = F.mse_loss(quantized.detach(), x)

        if not self._use_ema:
            q_latent_loss = F.mse_loss(quantized, x.detach())
            loss = q_latent_loss + self._commitment_cost * e_latent_loss

        if self._use_ema:
            # Update embeddings using exponential moving averages
            if self.training:
                self._ema_count = self._decay * self._ema_count + (1 - self._decay) * torch.sum(encodings_one_hot_flat,
                                                                                                dim=0)
                n = torch.sum(self._ema_count)
                # Laplace smoothing
                self._ema_count = (self._ema_count + self._epsilon) / (n + self._embedding_dim * self._epsilon) * n
                dw = torch.matmul(encodings_one_hot_flat.t(), x_flat)
                self._ema_weight = self._decay * self._ema_weight + (1 - self._decay) * dw
                self._embedding = self._ema_weight / self._ema_count.unsqueeze(-1)
            loss = self._commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(encodings_one_hot_flat, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Move embedding axis back to its initial position
        quantized = torch.moveaxis(quantized, -1, self._dim).contiguous()
        return quantized, loss, perplexity
