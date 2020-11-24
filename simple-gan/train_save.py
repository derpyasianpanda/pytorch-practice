import torch
import numpy as np
from models import *
from time import time
from torch import optim, nn, tensor
from utility import generate_evens, bits_to_int


SAVE_MODEL = False
FILEPATH = "./simple-gan/models"
DISPLAY_TRAINING = False

BIT_LEN = 8 # Maximum amount of bits to create a number. Max number will be (2 ^ bit_len) - 1
INPUT_LEN = BIT_LEN # How many random inputs the generator will use to generate a number
MAXIMUM_INPUT_VARIATION = 3 # How many different input states that can be used for the generator
TEST_AMOUNT = 20


def train(
    generator, discriminator, bit_len, input_len,
    epochs=500, batch_size=15, display_training=False, display_training_freq=100
):
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

    loss = nn.BCELoss()

    for epoch in range(epochs):
        # Generating test data for each epoch (training step)
        test_labels, test_data = generate_evens(bit_len, batch_size)
        # The labels are put under a view b/c the loss function requires the extra batch_size x 1 size
        test_labels, test_data = \
            tensor(test_labels).float().view(batch_size, 1), tensor(test_data).float()

        # Begin training the generator
        generator_optimizer.zero_grad()
        # Q: Why did I need to convert to floats for input???
        # A: Seems to be b/c the neural network has weights that are floats and in the end
        # going through it will make it become a float. Might as well start at a float then?
        input_noise = torch.randint(0, MAXIMUM_INPUT_VARIATION, (batch_size, input_len)).float()
        generated_data = generator(input_noise)

        # Use discriminator to determine if the generator is generating properly
        discriminate_generated = discriminator(generated_data)

        # Calculate generator loss and update the model
        generator_loss = loss(discriminate_generated, test_labels)
        generator_loss.backward()
        generator_optimizer.step()

        # Begin training the discriminator
        discriminator_optimizer.zero_grad()

        # Train the discriminator with the test data and calculate the loss
        test_discriminator_results = discriminator(test_data)
        test_discriminator_loss = loss(test_discriminator_results, test_labels)

        # Train the discriminator with the generated data and the loss as if the generated output should have all been wrong
        discriminate_generated = discriminator(generated_data.detach())
        generator_discriminator_loss = loss(discriminate_generated, torch.zeros(batch_size, 1))

        # Take average loss against generator and against test data and update the model
        discriminator_loss = (test_discriminator_loss + generator_discriminator_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()

        if epoch % display_training_freq == 0 and display_training:
            # We need to convert everything back to ints with rint b/c the model operates w/ floats
            print(f"Epoch #{epoch}: " +
                str([bits_to_int(bits) for bits in np.rint(generated_data.detach().numpy())]))


# Initialize new models
generator = Generator(INPUT_LEN, BIT_LEN)
discriminator = Discriminator(BIT_LEN)

# Perform training
train(generator, discriminator, BIT_LEN, INPUT_LEN, display_training=DISPLAY_TRAINING)

# Test the newly trained generator
generator_output = generator(
    torch.randint(0, MAXIMUM_INPUT_VARIATION, (TEST_AMOUNT, INPUT_LEN)).float())
print("TESTING GENERATOR",
    [bits_to_int(bits) for bits in np.rint(generator_output.detach().numpy())])

# Save the models with unique identifiers based on model and time
if SAVE_MODEL:
    torch.save(generator, f"{FILEPATH}/generator-{BIT_LEN}x{INPUT_LEN}-{int(time())}.pth")
    torch.save(discriminator, f"{FILEPATH}/discriminator-{BIT_LEN}x{INPUT_LEN}-{int(time())}.pth")