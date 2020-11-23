import torch
import numpy as np
from models import *
from time import time
from torch import optim, nn, tensor
from utility import generate_evens, bits_to_int


SAVE_MODEL = False
FILEPATH = "./simple-gan/models"
DISPLAY_TRAINING = False

BIT_LEN = 8 # Maximum number will be (2 ^ bit_len) - 1
INPUT_LEN = BIT_LEN
TEST_AMOUNT = 20
MAXIMUM_INPUT_VARIATION = 3


def train(
    generator, discriminator, bit_len, input_len,
    epochs=500, batch_size=15, display_training=False, display_training_freq=100
):
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

    loss = nn.BCELoss()

    for epoch in range(epochs):
        generator_optimizer.zero_grad()
        # Q: Why did I need to convert to floats for input???
        # A: Seems to be b/c the neural network has weights that are floats and in the end
        # going through it will make it become a float. Might as well start at a float then?
        noise = torch.randint(0, MAXIMUM_INPUT_VARIATION, (batch_size, input_len)).float()
        generated_data = generator(noise)

        test_labels, test_data = generate_evens(bit_len, batch_size)
        test_labels, test_data = tensor(test_labels).float(), tensor(test_data).float()

        discriminate_generated = discriminator(generated_data)
        generator_loss = loss(discriminate_generated, test_labels.view(batch_size, 1))
        generator_loss.backward()
        generator_optimizer.step()

        discriminator_optimizer.zero_grad()
        test_discriminator_results = discriminator(test_data)
        test_discriminator_loss = loss(test_discriminator_results, test_labels.view(batch_size, 1))

        discriminate_generated = discriminator(generated_data.detach())
        generator_discriminator_loss = loss(discriminate_generated, torch.zeros(batch_size, 1))

        # Take average loss against generator and against test data
        discriminator_loss = (test_discriminator_loss + generator_discriminator_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()

        if epoch % display_training_freq == 0 and display_training:
            print(f"Epoch #{epoch}: " +
                str([bits_to_int(bits) for bits in np.rint(generated_data.detach().numpy())]))


generator = Generator(INPUT_LEN, BIT_LEN)
discriminator = Discriminator(BIT_LEN)

train(generator, discriminator, BIT_LEN, INPUT_LEN, display_training=DISPLAY_TRAINING)

generator_output = generator(
    torch.randint(0, MAXIMUM_INPUT_VARIATION, (TEST_AMOUNT, INPUT_LEN)).float())
print("TESTING GENERATOR",
    [bits_to_int(bits) for bits in np.rint(generator_output.detach().numpy())])

if SAVE_MODEL:
    torch.save(generator, f"{FILEPATH}/generator-{BIT_LEN}x{INPUT_LEN}-{int(time())}.pth")
    torch.save(discriminator, f"{FILEPATH}/discriminator-{BIT_LEN}x{INPUT_LEN}-{int(time())}.pth")