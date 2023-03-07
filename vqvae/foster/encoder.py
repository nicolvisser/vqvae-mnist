from torch import nn
from torchinfo import summary


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
