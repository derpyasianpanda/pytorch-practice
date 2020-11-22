from models import *
from math import log2
from random import randint
from torch import optim, nn, tensor
from utility import generate_evens


def train(max, epochs=10, batch_size=1000, input_len=10):
    bit_len = int(log2(max))

    generator = Generator(input_len, bit_len)
    discriminator = Discriminator(bit_len)

    generator_optim = optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optim = optim.Adam(discriminator.parameters(), lr=0.001)

    loss = nn.BCELoss()

    for _ in range(epochs):
        generator_optim.zero_grad()
        noise = [randint(0, 100) for _ in range(input_len)]
        generated_data = generator(noise)

        test_labels, test_data = generate_evens(max, batch_size)
        test_labels, test_data = tensor(test_labels), tensor(test_data)
        print(test_data, test_labels)


train(1000)