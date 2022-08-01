'''
Models for dcgan, both generator and discriminator
'''
from attr import asdict
import torch.nn as nn
from . import N_FEAT_LATENT, N_FEAT_MAP_G, N_FEAT_MAP_D, CHANNELS

# function that defines initial weights
def weights_init(m: nn.Module) -> None:
    m_name = m.__class__.__name__
    if m_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif m_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class G(nn.Module):
    def __init__(self, ngpu: int):
        self.ngpu = ngpu
        super(G, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(
                        N_FEAT_LATENT,
                        N_FEAT_MAP_G*8,
                        4,
                        1,
                        0,
                        bias=False
                ),
                nn.BatchNorm2d(N_FEAT_MAP_G*8),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(
                        N_FEAT_MAP_G*8,
                        N_FEAT_MAP_G*4,
                        4,
                        2,
                        1,
                        bias=False
                ),
                nn.BatchNorm2d(N_FEAT_MAP_G*4),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(
                        N_FEAT_MAP_G*4,
                        N_FEAT_MAP_G*2,
                        4,
                        2,
                        1
                ),
                nn.BatchNorm2d(N_FEAT_MAP_G*2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                        N_FEAT_MAP_G*2,
                        N_FEAT_MAP_G,
                        4,
                        2,
                        1
                ),
                nn.BatchNorm2d(N_FEAT_MAP_G*2),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(
                        N_FEAT_MAP_G,
                        CHANNELS,
                        4,
                        2,
                        1,
                ),
                nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class D(nn.Module):
    def __init__(self, ngpu: int):
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(
                CHANNELS,
                N_FEAT_MAP_D,
                4,
                2,
                1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                N_FEAT_MAP_D,
                N_FEAT_MAP_D*2,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(N_FEAT_MAP_D*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                    N_FEAT_MAP_D*2,
                    N_FEAT_MAP_D*4,
                    4,
                    2,
                    1,
                    bias=False
            ),
            nn.BatchNorm2d(N_FEAT_MAP_D*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                    N_FEAT_MAP_D*4,
                    N_FEAT_MAP_D*8,
                    4,
                    2,
                    1,
                    bias=False
            ),
            nn.BatchNorm2d(N_FEAT_MAP_D*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                    N_FEAT_MAP_D*8,
                    1,
                    4,
                    1,
                    0,
                    bias=False
            ),
            nn.Sigmoid()
        )
    def forward(self, input):
        self.main(input)

class Inception:
    pass

class AlexNet:
    def __init__(self, ngpu: int):
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(CHANNELS, N_FEAT_MAP_D, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),
            
            nn.Conv2d(N_FEAT_MAP_D*2, N_FEAT_MAP_D*4, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2),

            nn.Conv2d(N_FEAT_MAP_D*4, N_FEAT_MAP_D*8, 4, 2, 1),
        )