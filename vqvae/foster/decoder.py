from torch import nn
from torchinfo import summary


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
