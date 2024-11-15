import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import seisbench.data as sbd
from scipy import signal
import warnings
import math
import torch.nn.init as init
import os
import LangsethData

warnings.filterwarnings('ignore')

class ParallelConv1dTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, final_layer=False):
        super(ParallelConv1dTransposeBlock, self).__init__()
        self.conv_transpose3 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose5 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv_transpose7 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.bn = nn.BatchNorm1d(out_channels * 3)
        self.elu = nn.ELU()
        self.final_layer = final_layer

    def forward(self, x):
        out3 = self.conv_transpose3(x)
        out5 = self.conv_transpose5(x)
        out7 = self.conv_transpose7(x)
        out = torch.cat([out3, out5, out7], dim=1)  # Concatenate along the channel dimension
        if not self.final_layer:
            out = self.bn(out)
            out = self.elu(out)
        return out

class ParallelConvGenerator(nn.Module):
    def __init__(self, latent_dim, num_layers, initial_filters, output_channels, seq_length=512):
        super(ParallelConvGenerator, self).__init__()
        filter_multiplier = 2  # Hardwired

        # Linear layer to expand latent space to feature map size
        self.fc_initial = nn.Linear(latent_dim, initial_filters * (filter_multiplier ** (num_layers - 1)) * 3 * seq_length // (2 ** num_layers))

        layers = []
        for i in range(num_layers):
            in_channels = initial_filters * (filter_multiplier ** (num_layers - i - 1)) * 3
            out_channels = initial_filters * (filter_multiplier ** (num_layers - i - 2)) if i != num_layers - 1 else output_channels
            layers.append(ParallelConv1dTransposeBlock(in_channels, out_channels, final_layer=(i == num_layers - 1)))
        self.layers = nn.ModuleList(layers)
        self.seq_length = seq_length

    def forward(self, z):
        # Expand latent space to feature map size
        hh = self.fc_initial(z).view(z.size(0), -1, self.seq_length // (2 ** len(self.layers)))  # Reshape to match expected input shape of deconv layers

        for layer in self.layers:
            hh = layer(hh)

        return torch.tanh(hh.mean(dim=1, keepdim=True))  # Output shape is (batch, 1, seq_length)

class ParallelConv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, final_layer=False):
        super(ParallelConv1dBlock, self).__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm1d(out_channels * 3)
        self.elu = nn.ELU()
        self.final_layer = final_layer

    def forward(self, x):
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)
        out = torch.cat([out3, out5, out7], dim=1)  # Concatenate along the channel dimension
        if not self.final_layer:
            out = self.bn(out)
            out = self.elu(out)
        return out

class ParallelConvDiscriminator(nn.Module):
    def __init__(self, input_channels, num_layers, initial_filters, seq_length=512):
        super(ParallelConvDiscriminator, self).__init__()
        filter_multiplier = 2  # Hardwired

        layers = []
        for i in range(num_layers):

            in_channels = input_channels if i == 0 else initial_filters * (filter_multiplier ** (i - 1)) * 3
            out_channels = initial_filters * (filter_multiplier ** i)
            layers.append(ParallelConv1dBlock(in_channels, out_channels, final_layer=(i == num_layers - 1)))
        self.layers = nn.ModuleList(layers)

        # Final linear layer to map the features to a single output value (between 0 and 1)
        self.fc_final = nn.Linear(initial_filters * (filter_multiplier ** (num_layers - 1)) * 3 * (seq_length // (2 ** num_layers)), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward through convolutional layers
        for layer in self.layers:
            x = layer(x)

        # Flatten the output for the final classification
        x = x.view(x.size(0), -1)

        # Pass through the final linear layer and apply sigmoid to get a probability score
        x = self.fc_final(x)
        return self.sigmoid(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to plot and save generated samples
def plot_generated_samples(generator, fixed_noise, epoch, num_samples=5):
    generator.eval()
    with torch.no_grad():
        generated_samples = generator(fixed_noise[:num_samples]).cpu().numpy()

    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i + 1)
        plt.plot(generated_samples[i][0])  # Plot the generated sequence
        plt.title(f"Sample {i + 1}")
        plt.axis('off')
    plt.suptitle(f"Generated Samples at Epoch {epoch + 1}")
    plt.tight_layout()
    plt.savefig(f'GAN_samples_epoch_{epoch + 1}.png')
    plt.close()

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    np.random.seed(3) #same seed as used during training for an in-house check
    torch.manual_seed(3)

    batch_size = 32
    seq_length = 1024 # window length
    print_interval = 2  # Print every x batches

    print('Constructing the dataset')
    dataloader_train, dataloader_valid = LangsethData.create_dataloaders('../LangsethDataset_v2/', seq_length, batch_size, mode='1d', num_workers=4)

    # Hyperparameters
    input_channels = 1  # 2-component seismograms
    num_layers = 2  # Number of layers in U-Net
    learning_rate = 1e-3
    num_epochs = 5000
    num_samples = 100  # Example number of samples
    initial_filters = 8  # Number of filters in the first layer
    filter_multiplier = 2  # Multiplier for the number of filters in each subsequent layer

    # Set up parameters for the GAN
    noise_dim = 25
    output_channels = 1
    hidden_dim = 256

    # Initialize GRU-based generator and GRU-based discriminator
    generator = ParallelConvGenerator(noise_dim, num_layers, initial_filters, output_channels, seq_length)
    #generator = GRUGenerator(noise_dim, hidden_dim, num_layers, output_channels, seq_length)

    discriminator = ParallelConvDiscriminator(input_channels, num_layers+1, initial_filters, seq_length)
    #discriminator = GRUDiscriminator(output_channels, hidden_dim, 5)

    print("Generator initialized with " + str(count_parameters(generator)) + " trainable parameters")
    print("Discriminator initialized with " + str(count_parameters(discriminator)) + " trainable parameters")

    # Optimizers
    lr = 0.0002
    beta1 = 0.5
    generator_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # Loss function
    adversarial_loss = nn.BCELoss()

    # Training the GAN
    num_epochs = 100
    batch_size = 64
    real_label = 1.
    fake_label = 0.

    # Define schedulers to halve the learning rate every 10 epochs
    generator_scheduler = optim.lr_scheduler.StepLR(generator_optimizer, step_size=10, gamma=0.66)
    discriminator_scheduler = optim.lr_scheduler.StepLR(discriminator_optimizer, step_size=10, gamma=0.66)

    # Directory to save checkpoints
    checkpoint_dir = './checkpoints_run2'
    os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer, g_scheduler, d_scheduler):
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'g_scheduler_state_dict': g_scheduler.state_dict(),
            'd_scheduler_state_dict': d_scheduler.state_dict()
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}_run2.pth'))
        print(f"Checkpoint saved at epoch {epoch+1}")

    # Fixed noise for consistent sample generation
    fixed_noise = torch.randn(batch_size, noise_dim)

    for epoch in range(num_epochs):
        generator.train()
        total_batches = len(dataloader_train)
        print_interval = total_batches // 10  # 10% progress tracking

        for batch_idx, batch in enumerate(dataloader_train):
            real_data = batch['signal']

            # some basic processing is needed for this application...
            real_data = real_data[:, 0, :].unsqueeze(1)

            # Normalize each channel of real data
            for i in range(real_data.shape[0]):
                for j in range(real_data.shape[1]):
                    real_data[i, j].div_(torch.abs(real_data[i, j]).max())  # In-place normalization

            # Train Discriminator
            discriminator_optimizer.zero_grad()

            # Real data
            batch_size = real_data.size(0)
            labels = torch.full((batch_size, 1), real_label, dtype=torch.float32)
            outputs = discriminator(real_data)
            d_loss_real = adversarial_loss(outputs, labels)
            d_loss_real.backward()

            # Fake data
            noise = torch.randn(batch_size, noise_dim)
            fake_data = generator(noise)
            labels.fill_(fake_label)
            outputs = discriminator(fake_data.detach())
            d_loss_fake = adversarial_loss(outputs, labels)
            d_loss_fake.backward()

            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            discriminator_optimizer.step()

            # Train Generator
            generator_optimizer.zero_grad()
            labels.fill_(real_label)  # Real labels for generator
            outputs = discriminator(fake_data)
            g_loss = adversarial_loss(outputs, labels)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            generator_optimizer.step()

            # Print progress at 10% intervals
            if (batch_idx + 1) % print_interval == 0:
                progress = 100 * (batch_idx + 1) / total_batches
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{total_batches}], "
                    f"Progress: {progress:.1f}%, D Loss: {d_loss_real.item() + d_loss_fake.item():.4f}, "
                    f"G Loss: {g_loss.item():.4f}")

        # Update learning rate schedules
        generator_scheduler.step()
        discriminator_scheduler.step()

        # Save checkpoint after each epoch
        save_checkpoint(epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, generator_scheduler, discriminator_scheduler)

        # Plot generated samples every X epochs
        #if (epoch + 1) % 2 == 0:  # Change 2 to the desired interval
        plot_generated_samples(generator, fixed_noise, epoch)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train D Loss: {d_loss_real.item() + d_loss_fake.item():.4f}, '
            f'G Loss: {g_loss.item():.4f}')
        #print(f"Learning rate for generator: {generator_scheduler.get_last_lr()[0]}")
        #print(f"Learning rate for discriminator: {discriminator_scheduler.get_last_lr()[0]}")
