# Models that need to be trained
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, input_len, bit_len):
        """Initiate a new Generator model
        input_len: Arbitrary length of input variables that will determine the output's value
        bit_len: Length of bits the output number will have
        """
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_len, bit_len)
        self.activation = nn.Sigmoid()


    def forward(self, x):
        return self.activation(self.fc1(x))


class Discriminator(nn.Module):
    def __init__(self, bit_len):
        """Initiate a new Discriminator model
        bit_len: Length of bits within the number that the model will classify
        """
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(bit_len, 1)
        self.activation = nn.Sigmoid()


    def forward(self, x):
        return self.activation(self.fc1(x))
