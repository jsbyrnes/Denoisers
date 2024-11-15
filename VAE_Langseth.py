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
import LangsethData

from WaveDecompNet import autoencoder_1D_models_torch as wdn

warnings.filterwarnings('ignore')

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(Conv1dBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ELU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))

class Conv1dTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, final_layer=False):
        super(Conv1dTransposeBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.relu = nn.ELU()
        self.bn = nn.BatchNorm1d(out_channels)
        self.final_layer = final_layer

    def forward(self, x):
        if self.final_layer:
            return self.conv_transpose(x)  # No ReLU and BatchNorm for the final layer
        else:
            return self.bn(self.relu(self.conv_transpose(x)))

class Encoder(nn.Module):
    def __init__(self, input_channels, num_layers, initial_filters, filter_multiplier):
        super(Encoder, self).__init__()
        layers = []
        self.num_layers = num_layers
        self.initial_filters = initial_filters
        self.filter_multiplier = filter_multiplier
        for i in range(num_layers):
            in_channels = input_channels if i == 0 else initial_filters * (filter_multiplier ** (i - 1))
            out_channels = initial_filters * (filter_multiplier ** i)
            layers.append(Conv1dBlock(in_channels, out_channels))
        self.layers = nn.ModuleList(layers)
        self.final_channels = initial_filters * (filter_multiplier ** (num_layers - 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, num_layers, initial_filters, filter_multiplier, output_channels):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.initial_filters = initial_filters
        self.filter_multiplier = filter_multiplier
        layers = []
        for i in range(num_layers):
            in_channels = initial_filters * (filter_multiplier ** (num_layers - i - 1))
            out_channels = initial_filters * (filter_multiplier ** (num_layers - i - 2)) if i != num_layers - 1 else output_channels
            layers.append(Conv1dTransposeBlock(in_channels, out_channels, final_layer=(i == num_layers - 1)))
        self.layers = nn.ModuleList(layers)

    def forward(self, z):
        h = z
        for layer in self.layers:
            h = layer(h)
        return h

class VAE(nn.Module):
    def __init__(self, input_channels, latent_dim, num_layers, initial_filters, filter_multiplier, input_length):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, num_layers, initial_filters, filter_multiplier)
        self.decoder = Decoder(num_layers, initial_filters, filter_multiplier, input_channels)

        # Calculate the size of the final output of the encoder
        self.encoder_output_size = self.final_conv_output_size(input_length, num_layers)

        # Convolutional layers to compute mu and logvar
        self.conv_mu = nn.Conv1d(self.encoder.final_channels, latent_dim, kernel_size=1, stride=1, padding=0)
        self.conv_logvar = nn.Conv1d(self.encoder.final_channels, latent_dim, kernel_size=1, stride=1, padding=0)

        # Convolutional layer to project latent space back to feature map size
        self.conv_initial = nn.Conv1d(latent_dim, self.encoder.final_channels, kernel_size=1, stride=1, padding=0)

    def final_conv_output_size(self, input_length, num_layers):
        # Calculate the size after convolutional layers given stride=2
        return input_length // (2 ** num_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        hh = self.encoder(x)

        # Compute mu and logvar
        mu = self.conv_mu(hh)
        logvar = self.conv_logvar(hh)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Project back to feature map size for the decoder
        z = self.conv_initial(z)

        # Decode
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def sample(self, num_samples):
        # Sample from the latent space
        z = torch.randn(num_samples, self.decoder.layers[0].in_channels, self.encoder_output_size).to(next(self.parameters()).device)
        samples = self.decoder(z)
        return samples

# Function to plot and save examples
def plot_examples(epoch, original, reconstructed, num_examples=5, filename='generated_examples_epoch_{epoch}.png'):
    fig, axs = plt.subplots(num_examples, 2, figsize=(15, 10))
    for i in range(num_examples):
        # Plot original channel 1
        axs[i, 0].plot(original[i, 0].cpu().numpy(), label='Original Channel 1')
        axs[i, 0].plot(reconstructed[i, 0].cpu().detach().numpy(), label='Reconstructed Channel 1')
        axs[i, 0].legend()
        # Plot original channel 2
        #axs[i, 1].plot(original[i, 1].cpu().numpy(), label='Original Channel 2')
        #axs[i, 1].plot(reconstructed[i, 1].cpu().detach().numpy(), label='Reconstructed Channel 2')
        #axs[i, 1].legend()
    plt.tight_layout()
    plt.savefig(filename.format(epoch=epoch))
    plt.close(fig)

# Function to generate new samples from the trained VAE
def generate_samples(model, num_samples, latent_dim):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients
        # Sample from the standard normal distribution
        z = torch.randn(num_samples, latent_dim)
        # Generate samples using the decoder
        generated_samples = model.decoder(z)
    return generated_samples

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to save model checkpoints
def save_checkpoint(model, optimizer, epoch, loss, path="vae_checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

if __name__ == "__main__":

    np.random.seed(3) #same seed as used during training for an in-house check
    torch.manual_seed(3)

    batch_size = 32
    nlen = 1024 # window length
    print_interval = 1  # Print every x batches

    print('Constructing the dataset')
    dataloader_train, dataloader_valid = LangsethData.create_dataloaders('../LangsethDataset_v2/', nlen, batch_size, mode='1d', num_workers=4)

    # Hyperparameters
    input_channels = 1  # 2-component seismograms
    latent_dim = 24
    num_layers = 5  # Number of layers in U-Net
    learning_rate = 1e-3
    num_epochs = 5000
    num_samples = 100  # Example number of samples
    initial_filters = 16  # Number of filters in the first layer
    filter_multiplier = 2  # Multiplier for the number of filters in each subsequent layer

    # Model, optimizer, and training
    model = VAE(input_channels, latent_dim, num_layers, initial_filters, filter_multiplier, nlen)
    #model = VAE()
    print("Model initialized with " + str(count_parameters(model)) + " trainable parameters")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=10, verbose=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_bce = 0
        total_kld = 0

        # Training loop
        for batch in dataloader_train:
            optimizer.zero_grad()
            real_batch = batch['signal']

            # some basic processing is needed for this application...
            real_batch = real_batch[:, 0, :].unsqueeze(1)

            # Normalize each channel of real data
            for i in range(real_batch.shape[0]):
                for j in range(real_batch.shape[1]):
                    real_batch[i, j].div_(torch.abs(real_batch[i, j]).max())  # In-place normalization
                
            # Forward pass through the VAE
            recon_batch, mu, logvar = model(real_batch)

            # Compute losses
            BCE = F.mse_loss(recon_batch, real_batch, reduction='sum')  # Reconstruction loss
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL Divergence

            loss = BCE + KLD

            # Backpropagation
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
            optimizer.step()

            # Accumulate the loss for logging
            total_loss += loss.item()
            total_bce += BCE.item()
            total_kld += KLD.item()

        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {total_loss:.4f} (BCE: {total_bce:.4f}, KLD: {total_kld:.4f})')

        # Validation loop
        model.eval()
        val_loss = 0
        val_bce = 0
        val_kld = 0
        with torch.no_grad():  # Disable gradient calculation for validation
            for batch in dataloader_valid:
                valid_batch = batch['signal']

                # some basic processing is needed for this application...
                valid_batch = valid_batch[:, 0, :].unsqueeze(1)

                # Normalize each channel of real data
                for i in range(valid_batch.shape[0]):
                    for j in range(valid_batch.shape[1]):
                        valid_batch[i, j].div_(torch.abs(valid_batch[i, j]).max())  # In-place normalization

                # Forward pass through the VAE
                recon_batch, mu, logvar = model(valid_batch)
                
                # Compute losses
                BCE = F.mse_loss(recon_batch, valid_batch, reduction='sum')  # Reconstruction loss
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL Divergence

                loss = BCE + KLD

                # Accumulate the loss for logging
                val_loss += loss.item()
                val_bce += BCE.item()
                val_kld += KLD.item()

        print(f'Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_loss:.4f} (BCE: {val_bce:.4f}, KLD: {val_kld:.4f})')

        # Step the learning rate scheduler
        scheduler.step(total_loss)

        # Save the model checkpoint
        if epoch % print_interval == 0:
            #avg_loss = total_loss / len(dataloader_valid)
            #avg_bce = total_bce / len(dataloader_valid)
            #avg_kld = total_kld / len(dataloader_valid)
            #print(f'Epoch {epoch}, Total Loss: {avg_loss}, Reconstruction Loss: {avg_bce}, KL Divergence: {avg_kld}')

            # Plot and save examples
            for batch in dataloader_valid:
                valid_batch = batch['signal']

                valid_batch = valid_batch[:, 0, :].unsqueeze(1)

                # Normalize each channel of real data
                for i in range(valid_batch.shape[0]):
                    for j in range(valid_batch.shape[1]):
                        valid_batch[i, j].div_(torch.abs(valid_batch[i, j]).max())  # In-place normalization

                recon_batch, _, _ = model(valid_batch)
                plot_examples(epoch, valid_batch[:5], recon_batch[:5], num_examples=5, filename='generated_examples_epoch_{epoch}.png')
                break

            if scheduler.get_last_lr()[0] < 1e-6:
                print('Learning rate reduce below minimum')
                break

    #save_checkpoint(model, optimizer, epoch, total_loss, path=f"vae_checkpoint_epoch_{epoch}.pth")
    torch.save(model.state_dict(), 'VAE_v3.pt')

    print("Training complete")


