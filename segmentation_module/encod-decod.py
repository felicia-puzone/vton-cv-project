import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transforms = transforms.Compose([transforms.ToTensor()])

# TODO: define training and testing datasets
# train_dataset = datasets.MNIST('/tmp/data', train=True, download=True,
#                               transform=transforms)
# test_dataset = datasets.MNIST('./data', train=False, download=True,
#                              transform=transforms)
BATCH_SIZE = 64  # number of data points in each batch
N_EPOCHS = 10  # times to run the model on complete data
INPUT_DIM = 28 * 28  # size of each input
HIDDEN_DIM = 256  # hidden dimension
LATENT_DIM = 20  # latent vector dimension
lr = 1e-3  # learning rate


# TODO: define training and testing batches
# train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)

class Encoder(nn.Module):


    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int):
        '''
      Args:
          input_dim: A integer indicating the size of input
            (in case of MNIST 28 * 28).
          hidden_dim: A integer indicating the size of hidden dimension.
          z_dim: A integer indicating the latent dimension.
      '''
        super(Encoder, self).__init__()

        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * z_dim),
        )

    def forward(self, x: torch.Tensor):
        # x is of shape [batch_size, input_dim]

        hidden = self.encoder(x)
        z_mu, z_logvar = hidden[:, :self.z_dim], hidden[:, self.z_dim:]

        return z_mu, z_logvar


class Decoder(nn.Module):


    def __init__(self, z_dim: int, hidden_dim: int, output_dim: int):
        '''
      Args:
          z_dim: A integer indicating the latent size.
          hidden_dim: A integer indicating the size of hidden dimension.
          output_dim: A integer indicating the output dimension
            (in case of MNIST it is 28 * 28)
      '''
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(x)


class Segmentation_Module(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.'''

    def __init__(self, enc: Encoder, dec: Decoder):
        super(Segmentation_Module, self).__init__()

        self.enc = enc
        self.dec = dec

    def forward(self, x: torch.Tensor):
        pass
