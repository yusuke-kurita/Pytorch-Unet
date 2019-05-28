from torch import cat
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.en_conv1 = nn.Conv2d(2 ** 0, 2 ** 4, 5, 2, 1)
        self.en_conv2 = nn.Conv2d(2 ** 4, 2 ** 5, 5, 2, 1)
        self.en_conv3 = nn.Conv2d(2 ** 5, 2 ** 6, 5, 2, 1)
        self.en_conv4 = nn.Conv2d(2 ** 6, 2 ** 7, 5, 2, 1)
        self.en_conv5 = nn.Conv2d(2 ** 7, 2 ** 8, 5, 2, 1)
        self.en_conv6 = nn.Conv2d(2 ** 8, 2 ** 9, 5, 2, 1)
        self.de_conv6 = nn.ConvTranspose2d(2 ** 9, 2 ** 8, 5, 2, 1)
        self.de_conv5 = nn.ConvTranspose2d(2 ** 9, 2 ** 7, 5, 2, 1)
        self.de_conv4 = nn.ConvTranspose2d(2 ** 8, 2 ** 6, 5, 2, 1)
        self.de_conv3 = nn.ConvTranspose2d(2 ** 7, 2 ** 5, 5, 2, 1)
        self.de_conv2 = nn.ConvTranspose2d(2 ** 6, 2 ** 4, 5, 2, 1)
        self.de_conv1 = nn.ConvTranspose2d(2 ** 5, 2 ** 0, 5, 2, 1)
        self.en_norm1 = nn.BatchNorm2d(2 ** 4)
        self.en_norm2 = nn.BatchNorm2d(2 ** 5)
        self.en_norm3 = nn.BatchNorm2d(2 ** 6)
        self.en_norm4 = nn.BatchNorm2d(2 ** 7)
        self.en_norm5 = nn.BatchNorm2d(2 ** 8)
        self.norm = nn.BatchNorm2d(2 ** 9)
        self.de_norm5 = nn.BatchNorm2d(2 ** 8)
        self.de_norm4 = nn.BatchNorm2d(2 ** 7)
        self.de_norm3 = nn.BatchNorm2d(2 ** 6)
        self.de_norm2 = nn.BatchNorm2d(2 ** 5)
        self.de_norm1 = nn.BatchNorm2d(2 ** 4)

    def forward(self, input):
        en1 = F.leaky_relu(self.en_norm1(self.en_conv1(input)), 0.2)
        en2 = F.leaky_relu(self.en_norm2(self.en_conv2(en1)), 0.2)
        en3 = F.leaky_relu(self.en_norm3(self.en_conv3(en2)), 0.2)
        en4 = F.leaky_relu(self.en_norm4(self.en_conv4(en3)), 0.2)
        en5 = F.leaky_relu(self.en_norm5(self.en_conv5(en4)), 0.2)
        h = F.leaky_relu(self.norm(self.en_conv6(en5)), 0.2)
        de5 = F.relu(F.dropout2d(self.de_norm5(
            self.de_conv6(h, output_size=en5.size()))))
        de4 = F.relu(F.dropout2d(self.de_norm4(
            self.de_conv5(cat((de5, en5), 1), output_size=en4.size()))))
        de3 = F.relu(F.dropout2d(self.de_norm3(
            self.de_conv4(cat((de4, en4), 1), output_size=en3.size()))))
        de2 = F.relu(F.dropout2d(self.de_norm2(
            self.de_conv3(cat((de3, en3), 1), output_size=en2.size()))))
        de1 = F.relu(self.de_norm1(
            self.de_conv2(cat((de2, en2), 1), output_size=en1.size())))
        output = F.sigmoid(self.de_conv1(
            cat((de1, en1), 1), output_size=input.size()))

        return output
