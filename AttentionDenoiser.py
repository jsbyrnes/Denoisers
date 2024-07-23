#drafted by chatgpt 4

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pywt
import matplotlib.pyplot as plt
import seisbench.data as sbd
from scipy import signal

# GPU setup
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Positional Encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Define custom transformer model with positional encoding
class CustomTransformer(nn.Module):
    def __init__(self, input_dim, num_layers=6, num_heads=8, hidden_dim=512, dropout=0.1, max_len=5000):
        super(CustomTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(input_dim, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, input_tensor):
        input_tensor = self.pos_encoder(input_tensor)
        output = self.transformer_encoder(input_tensor)
        output = self.linear(output)
        return output

class MultiLayerAttentionModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, dropout=0.1, activation_fn=nn.ELU()):
        super(MultiLayerAttentionModel, self).__init__()
        self.pos_encoder = PositionalEncoding(input_dim, max_len=5000)
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.activation_fn = activation_fn
        self.norm_layers = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, input_tensor):
        input_tensor = self.pos_encoder(input_tensor)
        input_tensor = input_tensor.permute(1, 0, 2)  # Change shape to (L, N, E)

        for attn, norm in zip(self.layers, self.norm_layers):
            attn_output, _ = attn(input_tensor, input_tensor, input_tensor)
            attn_output = self.activation_fn(attn_output)  # Apply activation function after attention
            input_tensor = attn_output + input_tensor  # Add & Norm step
            input_tensor = norm(input_tensor)

        input_tensor = input_tensor.permute(1, 0, 2)  # Change shape back to (N, L, E)
        output = self.linear(input_tensor)
        return output

class UNet(nn.Module):
    def __init__(self, nf, nlen, drop=0, ncomp=1, fac=1):
        super(UNet, self).__init__()
        self.ncomp = ncomp
        in_channels = 2 if ncomp == 1 else 6

        # Define the encoder part
        self.enc1 = self.conv_block(in_channels, 8*fac, drop)
        self.enc2 = self.conv_block(8*fac, 16*fac, drop, stride=2)
        self.enc3 = self.conv_block(16*fac, 32*fac, drop, stride=2)
        self.enc4 = self.conv_block(32*fac, 64*fac, drop, stride=2)
        self.enc5 = self.conv_block(64*fac, 128*fac, drop, stride=2)

        # Define the decoder part
        self.dec4 = self.conv_block(128*fac + 64*fac, 64*fac, drop)
        self.dec3 = self.conv_block(64*fac + 32*fac, 32*fac, drop)
        self.dec2 = self.conv_block(32*fac + 16*fac, 16*fac, drop)
        self.dec1 = self.conv_block(16*fac + 8*fac, 8*fac, drop)

        # Final output layer
        self.final = nn.Conv2d(8*fac, 1 if ncomp == 1 else 3, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        # Decoder with skip connections
        dec4 = self.upconv(enc5, enc4, 64)
        dec3 = self.upconv(dec4, enc3, 32)
        dec2 = self.upconv(dec3, enc2, 16)
        dec1 = self.upconv(dec2, enc1, 8)

        # Final convolution
        out = self.final(dec1)
        return out

    def conv_block(self, in_channels, out_channels, drop, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(drop),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(drop)
        )

    def upconv(self, x, skip, out_channels):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x.size(1), out_channels, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to perform full DWT on a single seismogram
def full_dwt_transform(seismogram, wavelet='db4'):
    max_level = pywt.dwt_max_level(len(seismogram), wavelet)
    coeffs = pywt.wavedec(seismogram, wavelet, level=max_level)
    return coeffs

# Function to prepare input tensor from wavelet coefficients
def prepare_input_tensor(seismogram_batch, wavelet='db4', num_components=3):
    input_tensors = []
    coeffs_slices = []
    for seismogram in seismogram_batch:
        concatenated_coeffs = []
        component_slices = []
        for component in range(num_components):
            coeffs = full_dwt_transform(seismogram[:, component], wavelet)
            slices = [len(c) for c in coeffs]
            component_slices.append(slices)
            concatenated_coeffs.append(np.concatenate(coeffs))
            # print("Wavelet coefficients shape:", [c.shape for c in coeffs])  # Debugging line
        concatenated_coeffs = np.hstack(concatenated_coeffs)
        input_tensor = torch.tensor(concatenated_coeffs, dtype=torch.float32).unsqueeze(0)
        input_tensors.append(input_tensor)
        coeffs_slices.append(component_slices)
    input_tensor = torch.cat(input_tensors, dim=0)
    return input_tensor, coeffs_slices

# Corrected function to reorganize the output tensor into wavelet coefficients format
def reorganize_coeffs(output_tensor, coeffs_slices, num_components=3):
    output_tensor = output_tensor.detach().numpy()  # Ensure numpy array
    reconstructed_seismograms = []

    for i, slices in enumerate(coeffs_slices):
        start = 0
        components = []
        for component_slices in slices:
            coeffs = []
            for slice_len in component_slices:
                end = start + slice_len
                coeff = output_tensor[i, 0, start:end]  # Handle batch dimension correctly
                coeffs.append(coeff)  # No need to squeeze as it's 1D
                start = end  # Update start index
            components.append(coeffs)
        reconstructed_seismograms.append(components)

    return reconstructed_seismograms

# Function to reconstruct seismogram from wavelet coefficients
def idwt_transform(coeffs, wavelet='db4'):
    try:
        coeffs = [np.asarray(c) for c in coeffs]  # Ensure correct shape
        reconstructed = pywt.waverec(coeffs, wavelet)
        return reconstructed
    except Exception as e:
        print("Error during inverse DWT:", e)
        return None

# Monte Carlo dropout denoising function
def monte_carlo_denoising(model, test_seismogram, wavelet='db4', num_samples=10, num_components=3):
    model.train()  # Ensure the dropout layers are active
    predictions = []

    if len(test_seismogram.shape) == 2:
        #only one. Add in the missing first dimension
        test_seismogram = np.expand_dims(test_seismogram, 0)

    test_input_tensor, test_coeffs_slices = prepare_input_tensor(test_seismogram, wavelet, num_components)
    if test_input_tensor.dim() == 2:
        test_input_tensor = test_input_tensor.unsqueeze(0)  # Ensure 3D tensor for transformer
    test_input_tensor = test_input_tensor.permute(1, 0, 2)

    for _ in range(num_samples):
        with torch.no_grad():
            denoised_output_tensor = model(test_input_tensor.to(device))

        reorganized_coeffs = reorganize_coeffs(denoised_output_tensor, test_coeffs_slices, num_components)

        denoised_seismogram = []
        for component_coeffs in reorganized_coeffs[0]:
            denoised_component = idwt_transform(component_coeffs, wavelet)
            if denoised_component is not None:
                denoised_seismogram.append(denoised_component)
            else:
                print("Error: Unable to reconstruct component.")

        if denoised_seismogram:
            denoised_seismogram = np.stack(denoised_seismogram, axis=-1)
            predictions.append(denoised_seismogram)

    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions, axis=0)
    return mean_prediction

def data_generator(batch_size, data, seq_length, sr, test=False):
    #data is a data set from the seisbench plateform. Pretty handy.
    #data format should be (batch, length, component). Raw seismograms

    #will need to adjust the components here. Hardwired to three.

    md = data.metadata
    ntraces,_ = md.shape
    nd  = 2000 #dont need to load everything
    sos = signal.butter(2, 2.5, 'hp', fs=sr, output='sos')

    data_to_noise = 35393

    #subset it once, or you get a "bus error" idk why
    #skip set of 500 each for later validation

    if not test:

        data_inds  = np.random.randint(data_to_noise - 5000, size=(nd,)) #this samples the data in OBST2024 dataset
        noise_inds = np.random.randint(low=data_to_noise + 5000, high=ntraces, size=(nd,)) #this samples the noise in OBST2024 dataset

    else: #these are always the same

        data_inds  = np.arange(data_to_noise-5000, data_to_noise)#np.random.randint(low=data_to_noise - 500, high=data_to_noise, size=(nd,)) #this samples the data in OBST2024 dataset
        noise_inds = np.arange(data_to_noise, ntraces)#np.random.randint(low=data_to_noise, high=data_to_noise + 500, size=(nd,)) #this samples the noise in OBST2024 dataset

    datavec  = data.get_waveforms(data_inds)[:,0:3,:]
    itpvec   = np.round(md.trace_p_arrival_sample[data_inds].to_numpy())*(sr/100)#they are valid for 100 Hz but OBST2024 does't scale
    noisevec = data.get_waveforms(noise_inds)[:,0:3,:]

    #reformat this to get random data from the OBST2024 dataset

    batch_inputs  = np.zeros((batch_size,seq_length, 3))
    batch_outputs = np.zeros((batch_size,seq_length, 3))

    for ii in range(batch_size):
        # window data, noise, and data+noise timeseries based on shift

        #The original data format was to have all three components in one vector, which is fucking bizare

        #make subsig1, subsig2, subsig3
        sig_valid = False
        while not sig_valid:

            #if not test:
            sig_ind = np.random.randint(nd)
            amp = np.random.randn()*2.5
            #else:
            #    sig_ind = ii
            #    amp = 1#2*ii/batch_size

            #sigwfs = data.get_waveforms(sig_ind)[0:3,:]
            sigwfs = datavec[sig_ind, :, :]

            if not np.array([ v.any() for v in sigwfs ]).all():
                continue

            itp = itpvec[sig_ind]

            if np.isnan(itp):
                continue

            sigwfs = sigwfs[:, np.where(np.sum(sigwfs, axis=0))[0]]

            if not test:
                start_d    = int(np.random.uniform(np.max((0, itp - seq_length)), 3000 - seq_length - 10))
            else:
                start_d = np.min( (int(itp - 128), 3000 - seq_length - 10))

            subsig1 = signal.resample(sigwfs[0,:], 3000)
            subsig2 = signal.resample(sigwfs[1,:], 3000)
            subsig3 = signal.resample(sigwfs[2,:], 3000)

            subsig1 = signal.sosfilt(sos, subsig1)
            subsig2 = signal.sosfilt(sos, subsig2)
            subsig3 = signal.sosfilt(sos, subsig3)

            subsig1 = subsig1[start_d:start_d+seq_length]
            subsig2 = subsig2[start_d:start_d+seq_length]
            subsig3 = subsig3[start_d:start_d+seq_length]

            sig_amp = np.max(np.abs(np.concatenate((subsig1,subsig2,subsig3))))
            subsig1 = subsig1*(amp/sig_amp)
            subsig2 = subsig2*(amp/sig_amp)
            subsig3 = subsig3*(amp/sig_amp)

            sig_valid = True

        noise_valid = False
        while not noise_valid:

            #if not test:
            noise_ind = np.random.randint(nd)
            amp = np.random.uniform(-2,2)

            #else:
            #    noise_ind = ii
            #    amp = 2*ii/batch_size + 0.05#gets bigger and bigger

            noisewfs = noisevec[noise_ind, :, :]

            if not np.array([ v.any() for v in noisewfs ]).all():
                continue

            noisewfs = noisewfs[:, np.where(np.sum(noisewfs, axis=0))[0]]

            subnoise1 = signal.resample(noisewfs[0,:], 3000)
            subnoise2 = signal.resample(noisewfs[1,:], 3000)
            subnoise3 = signal.resample(noisewfs[2,:], 3000)

            subnoise1 = signal.sosfilt(sos, subnoise1)
            subnoise2 = signal.sosfilt(sos, subnoise2)
            subnoise3 = signal.sosfilt(sos, subnoise3)

            if not test:
                start = int(np.random.uniform(0, 3000 - seq_length - 10))#some buffer if the pick sucks
            else:
                start = 1000

            subnoise1 = subnoise1[start:start+seq_length]
            subnoise2 = subnoise2[start:start+seq_length]
            subnoise3 = subnoise3[start:start+seq_length]

            sig_amp = np.max(np.abs(np.concatenate((subnoise1,subnoise2,subnoise3))))

            subnoise1 = subsig1 + subnoise1*(amp/sig_amp)
            subnoise2 = subsig2 + subnoise2*(amp/sig_amp)
            subnoise3 = subsig3 + subnoise3*(amp/sig_amp)
            noise_valid = True

        sig_std = np.std(np.concatenate((subnoise1,subnoise2,subnoise3)))

        subsig1 = subsig1/sig_std
        subsig2 = subsig2/sig_std
        subsig3 = subsig3/sig_std
        subnoise1 = subnoise1/sig_std
        subnoise2 = subnoise2/sig_std
        subnoise3 = subnoise3/sig_std

        #noisy data
        batch_inputs[ii, :, 0] = subnoise1
        batch_inputs[ii, :, 1] = subnoise2
        batch_inputs[ii, :, 2] = subnoise3

        #clean signals
        batch_outputs[ii, :, 0] = subsig1
        batch_outputs[ii, :, 1] = subsig2
        batch_outputs[ii, :, 2] = subsig3

    return batch_inputs, batch_outputs

def stft_3comp_data_generator(model, batch_size, data_vec, noise_vec, itpvec, sr, nperseg, noverlap, norm_input, eps=1e-9, nlen=128, valid=False, post=False):
    #data is a data set from the seisbench plateform. Pretty handy.
    #don't need meta data for the noise

    nd, _, _ = data_vec.shape
    nn, _, _ = noise_vec.shape

    sos = signal.butter(2, 2.5, 'hp', fs=sr, output='sos')
    data_to_noise = 35393

    nf = int(np.ceil(nperseg/2))

    #if not valid:

    #    data_inds  = np.random.randint(data_to_noise - 500, size=(nd,)) #this samples the data in OBST2024 dataset
    #    noise_inds = np.random.randint(low=data_to_noise + 500, high=ntraces, size=(nd,)) #this samples the noise in OBST2024 dataset

    #else:

    #    data_inds  = np.random.randint(low=data_to_noise - 500, high=data_to_noise, size=(nd,)) #this samples the data in OBST2024 dataset
    #    noise_inds = np.random.randint(low=data_to_noise, high=data_to_noise + 500, size=(nd,)) #this samples the noise in OBST2024 dataset

    #datavec  = data.get_waveforms(data_inds)[:,0:3,:]
    #itpvec   = np.round(md.trace_p_arrival_sample[data_inds].to_numpy())*(sr/100)#they are valid for 100 Hz but OBST2024 does't scale, you have to do it
    #noisevec = data.get_waveforms(noise_inds)[:,0:3,:]

    while True:
        # grab batch

        #reformat this to get random data from the OBS dataset, and use the p waves picks to get noise as the stuff before it

        #what are the outputs I need to produce
        #batch_inputs - generated by data_2_input on the seperate stft
        #batch_outputs - processed ontop of the inputs
        #subsigs - concatenated data traces
        #subnoises - concatenated noise traces
        #raw_stft - same as batch_inputs just different normalization

        #I've replaced Amanda's fixed 16 with the size of the frequency vector

        batch_inputs = np.zeros((batch_size,nf,nlen,6))
        raw_stft     = np.zeros((batch_size,nf,nlen,6))

        if model=='v1':
            batch_outputs=np.zeros((batch_size,nf,nlen,3))
        elif model=='v2':
            batch_outputs=np.zeros((batch_size,nf,nlen,6))
        elif model=='v3':
            batch_outputs=np.zeros((batch_size,nf,nlen,9))

        subsigs=np.zeros((batch_size,3*nlen))
        subnoises=np.zeros((batch_size,3*nlen))

        for ii in range(batch_size):
            # window data, noise, and data+noise timeseries based on shift

            #The original data format was to have all three components in one vector, which is fucking bizare

            #make subsig1, subsig2, subsig3
            sig_valid = False
            while not sig_valid:

                sig_ind = np.random.randint(nd)

                if not valid:

                    sig_ind  = np.random.randint(0, nd - 500)
                    amp   = np.random.randn()*2.5

                else:

                    sig_ind  = nd - 500 + ii#np.random.randint(low=500, high=nd) #this samples the data in OBST2024 dataset
                    amp = 1

                sigwfs = data_vec[sig_ind, :, :]

                if not np.array([ v.any() for v in sigwfs ]).all():
                    breakpoint()
                    continue

                itp = itpvec[sig_ind]

                if np.isnan(itp):
                    breakpoint()
                    continue

                #trim the data, seisbench pads it to make everything equal length
                #that's fine for picking, terrible for denoising.

                sigwfs = sigwfs[:, np.where(np.sum(sigwfs, axis=0))[0]]
                if not valid:
                    start    = int(np.random.uniform(np.max((0, itp - nlen)), 3000 - nlen - 10))
                else:
                    start = np.min( (int(itp - 128), 3000 - nlen - 10))

                subsig1 = signal.resample(sigwfs[0,:], 3000)
                subsig2 = signal.resample(sigwfs[1,:], 3000)
                subsig3 = signal.resample(sigwfs[2,:], 3000)

                subsig1 = signal.sosfilt(sos, subsig1)
                subsig2 = signal.sosfilt(sos, subsig2)
                subsig3 = signal.sosfilt(sos, subsig3)

                subsig1 = subsig1[start:start+nlen]
                subsig2 = subsig2[start:start+nlen]
                subsig3 = subsig3[start:start+nlen]

                sig_amp = np.max(np.abs(np.concatenate((subsig1,subsig2,subsig3))))
                subsig1 = subsig1*(amp/sig_amp)
                subsig2 = subsig2*(amp/sig_amp)
                subsig3 = subsig3*(amp/sig_amp)

                sig_valid = True

            noise_valid = False
            while not noise_valid:

                if not valid:

                    noise_ind = np.random.randint(nn - 500) #this samples the noise in OBST2024 dataset
                    amp = np.random.uniform(-2,2)

                else:

                    noise_ind  = np.random.randint(low=500, high=nn) #this samples the data in OBST2024 dataset
                    amp = 2*ii/batch_size + 0.05#get's noiser and noiser

                noisewfs = noise_vec[noise_ind, :, :]

                if not np.array([ v.any() for v in noisewfs ]).all():
                    breakpoint()
                    continue


                #trim the data, seisbench pads it to make everything equal length
                #that's fine for picking, terrible for denoising.
                noisewfs = noisewfs[:, np.where(np.sum(noisewfs, axis=0))[0]]

                subnoise1 = signal.resample(noisewfs[0,:], 3000)
                subnoise2 = signal.resample(noisewfs[1,:], 3000)
                subnoise3 = signal.resample(noisewfs[2,:], 3000)

                subnoise1 = signal.sosfilt(sos, subnoise1)
                subnoise2 = signal.sosfilt(sos, subnoise2)
                subnoise3 = signal.sosfilt(sos, subnoise3)

                if not valid:
                    start = int(np.random.uniform(0, 3000 - nlen - 10))#some buffer if the pick sucks
                else:
                    start = 1000

                subnoise1 = subnoise1[start:start+nlen]
                subnoise2 = subnoise2[start:start+nlen]
                subnoise3 = subnoise3[start:start+nlen]

                sig_amp = np.max(np.abs(np.concatenate((subnoise1,subnoise2,subnoise3))))
                subnoise1 = subnoise1*(amp/sig_amp)
                subnoise2 = subnoise2*(amp/sig_amp)
                subnoise3 = subnoise3*(amp/sig_amp)

                noise_valid = True

            subsigs[ii,:]=np.concatenate((subsig1,subsig2,subsig3))
            subnoises[ii,:]=np.concatenate((subnoise1,subnoise2,subnoise3))
            #subsigs[ii,:]=np.concatenate((np.random.rand(subsig1.size),np.random.rand(subsig1.size),np.random.rand(subsig1.size)))
            #subnoises[ii,:]=np.concatenate((np.random.rand(subsig1.size),np.random.rand(subsig1.size),np.random.rand(subsig1.size)))

            _, _, stftsig1 = signal.stft(subsig1, fs=sr, nperseg=nperseg, noverlap=noverlap)
            _, _, stftnoise1 = signal.stft(subnoise1, fs=sr, nperseg=nperseg, noverlap=noverlap)
            stftinput1 = stftsig1+stftnoise1
            _, _, stftsig2 = signal.stft(subsig2, fs=sr, nperseg=nperseg, noverlap=noverlap)
            _, _, stftnoise2 = signal.stft(subnoise2, fs=sr, nperseg=nperseg, noverlap=noverlap)
            stftinput2 = stftsig2+stftnoise2
            _, _, stftsig3 = signal.stft(subsig3, fs=sr, nperseg=nperseg, noverlap=noverlap)
            _, _, stftnoise3 = signal.stft(subnoise3, fs=sr, nperseg=nperseg, noverlap=noverlap)
            stftinput3 = stftsig3+stftnoise3

            # batch inputs are the real and imaginary parts of the stft of signal+noise
            raw_stft[ii,:,:,:]     = data_2_input(stftinput1,stftinput2,stftinput3,norm_input=False)
            batch_inputs[ii,:,:,:] = data_2_input(stftinput1,stftinput2,stftinput3,norm_input)

            if model=='v1':
                # batch outputs are real valued signal masks
                with np.errstate(divide='ignore'):
                    rat1=np.nan_to_num(np.abs(stftnoise1)/np.abs(stftsig1),posinf=1e20)
                batch_outputs[ii,:,:,0]=1/(1+rat1) # signal mask
                # batch outputs are
                with np.errstate(divide='ignore'):
                    rat2=np.nan_to_num(np.abs(stftnoise2)/np.abs(stftsig2),posinf=1e20)
                batch_outputs[ii,:,:,1]=1/(1+rat2) # signal mask
                # batch outputs are
                with np.errstate(divide='ignore'):
                    rat3=np.nan_to_num(np.abs(stftnoise3)/np.abs(stftsig3),posinf=1e20)
                batch_outputs[ii,:,:,2]=1/(1+rat3) # signal mask
            elif model=='v2':
                # batch outputs are
                batch_outputs[ii,:,:,0]=np.real(stftsig1)
                batch_outputs[ii,:,:,1]=np.imag(stftsig1)
                batch_outputs[ii,:,:,2]=np.real(stftsig2)
                batch_outputs[ii,:,:,3]=np.imag(stftsig2)
                batch_outputs[ii,:,:,4]=np.real(stftsig3)
                batch_outputs[ii,:,:,5]=np.imag(stftsig3)
                if norm_input:
                    batch_outputs[ii,:,:,0]=batch_outputs[ii,:,:,0]/np.max(np.abs(stftinput1))
                    batch_outputs[ii,:,:,1]=batch_outputs[ii,:,:,1]/np.max(np.abs(stftinput1))
                    batch_outputs[ii,:,:,2]=batch_outputs[ii,:,:,2]/np.max(np.abs(stftinput2))
                    batch_outputs[ii,:,:,3]=batch_outputs[ii,:,:,3]/np.max(np.abs(stftinput2))
                    batch_outputs[ii,:,:,4]=batch_outputs[ii,:,:,4]/np.max(np.abs(stftinput3))
                    batch_outputs[ii,:,:,5]=batch_outputs[ii,:,:,5]/np.max(np.abs(stftinput3))
            elif model=='v3':
                # batch outputs are
                batch_outputs[ii,:,:,0]=np.log(np.abs(stftsig1/stftinput1)+eps)
                batch_outputs[ii,:,:,1]=np.cos(np.angle(stftsig1/stftinput1))
                batch_outputs[ii,:,:,2]=np.sin(np.angle(stftsig1/stftinput1))
                batch_outputs[ii,:,:,3]=np.log(np.abs(stftsig2/stftinput2)+eps)
                batch_outputs[ii,:,:,4]=np.cos(np.angle(stftsig2/stftinput2))
                batch_outputs[ii,:,:,5]=np.sin(np.angle(stftsig2/stftinput2))
                batch_outputs[ii,:,:,6]=np.log(np.abs(stftsig3/stftinput3)+eps)
                batch_outputs[ii,:,:,7]=np.cos(np.angle(stftsig3/stftinput3))
                batch_outputs[ii,:,:,8]=np.sin(np.angle(stftsig3/stftinput3))
            # make all angles positive

        if post:
            yield(batch_inputs,batch_outputs,subsigs,subnoises,raw_stft)
        else:
            yield(batch_inputs,batch_outputs)

# Hyperparameters

model_type = 'MLA' #MLA, Transformer, or UNet

num_layers = 1
num_heads = 1
dropout = 0.1
batch_size = 256
learning_rate = 0.001
num_epochs = 250  # Default to 3 epochs
num_components = 3
hidden_dim = 1024*num_components #transformer only
wavelet = 'db1'
sr = 50
fac=4 # model size
eps=1e-9 # epsilon
drop=0.2 # model drop rate
nlen=1024# window length
nperseg=31 # param for STFT
noverlap=30 # param for STFT
norm_input=False # leave this alone
print_interval = 10  # Print every 5 batches

np.random.seed(150)
torch.manual_seed(0)

data = sbd.OBST2024(sample_rate=sr) #sampling rate doesn't work

# Amount of samples to use per epoch
num_samples = batch_size*200
seq_length = 1024

# Determine input dimension from wavelet transform and make validation data
if not model_type == 'UNet':

    seismogram_data, clean_seismogram_data = data_generator(batch_size, data, seq_length, sr, True)
    example_coeffs = full_dwt_transform(seismogram_data[0, :, 0], wavelet)
    input_dim = len(np.concatenate(example_coeffs)) * num_components

    #prepare as a validation set
    input_tensor_valid, coeffs_slices_valid = prepare_input_tensor(seismogram_data, wavelet, num_components)
    if input_tensor_valid.dim() == 2:
        input_tensor_valid = input_tensor_valid.unsqueeze(0)  # Ensure 3D tensor for transformer
    input_tensor_valid = input_tensor_valid.permute(1, 0, 2)  # Transformer expects (seq_len, batch, input_dim)
    target_tensor_valid, _ = prepare_input_tensor(clean_seismogram_data, wavelet, num_components)
    if target_tensor_valid.dim() == 2:
        target_tensor_valid = target_tensor_valid.unsqueeze(0)  # Ensure 3D tensor for transformer
    target_tensor_valid = target_tensor_valid.permute(1, 0, 2)

# Define custom transformer model
if model_type == 'MLA':
    model = MultiLayerAttentionModel(input_dim, num_layers, num_heads, dropout).to(device)
    print("Multi-layer attention network initialized with " + str(count_parameters(model)) + " trainable parameters")

if model_type == 'Transformer':
    model = CustomTransformer(input_dim, num_layers, num_heads, hidden_dim, dropout).to(device)
    print("Transformer initialized with " + str(count_parameters(model)) + " trainable parameters")

if model_type == 'UNet':
    model = UNet(int(np.ceil(nperseg/2)), nlen, drop=dropout, ncomp=3,fac=fac).to(device)
    print("UNet initialized with " + str(count_parameters(model)) + " trainable parameters")

    #35393 is the index between data and noise in OBST2024

    total, _ = data.metadata.shape

    data_vec  = data.get_waveforms(np.arange(0, 35393))[:,0:3,:]
    itpvec    = np.round(data.metadata.trace_p_arrival_sample[:35393].to_numpy())*(sr/100)#they are valid for 100 Hz but OBST2024 does't scale when you load it, you have to do it
    noise_vec = data.get_waveforms(np.arange(total - 35393, total))[:,0:3,:]

    my_test_data = stft_3comp_data_generator('v1',50,data_vec, noise_vec, itpvec,sr,nperseg,noverlap,norm_input,eps,nlen,valid=True, post=False)
    input_tensor_valid,target_tensor_valid = next(my_test_data)


# Define loss function and optimizer
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, threshold=0.00001)
#scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

# Training loop with shuffling and padding
for epoch in range(num_epochs):
    running_loss = 0.0
    batch_count = 0  # Counter for batches

    #This step is slow. But hardly the bottle-neck compared with training. Do more batches if too slow, data is heavily augmented.

    if not model_type =='UNet':
        seismogram_data, clean_seismogram_data = data_generator(num_samples, data, seq_length, sr)
    else: #different for UNet
        my_test_data = obs_tools.stft_3comp_data_generator('v1',50,data_vec, noise_vec, itpvec,sr,nperseg,noverlap,norm_input,eps,nlen,valid=True, post=False)
        seismogram_data,clean_seismogram_data = next(my_test_data)

    model.train()

    for i in range(0, len(seismogram_data), batch_size):

        if not model_type =='UNet':

            batch_data = seismogram_data[i:i+batch_size, :, :]
            batch_clean_data = clean_seismogram_data[i:i+batch_size, :, :]

            input_tensor, coeffs_slices = prepare_input_tensor(batch_data, wavelet, num_components)
            if input_tensor.dim() == 2:
                input_tensor = input_tensor.unsqueeze(0)  # Ensure 3D tensor for transformer
            input_tensor = input_tensor.permute(1, 0, 2).to(device)
            target_tensor, _ = prepare_input_tensor(batch_clean_data, wavelet, num_components)
            if target_tensor.dim() == 2:
                target_tensor = target_tensor.unsqueeze(0)  # Ensure 3D tensor for transformer
            target_tensor = target_tensor.permute(1, 0, 2).to(device)

        else:

            breakpoint()

        optimizer.zero_grad()

        outputs = model(input_tensor)

        loss = criterion(outputs, target_tensor)

        loss.backward()
        optimizer.step()

        running_loss = (running_loss*batch_count + loss.item())/(batch_count + 1)

        batch_count += 1  # Increment batch counter

        if batch_count % print_interval == 0:  # Print every `print_interval` batches
            print(f'Epoch {epoch + 1}, Batch {batch_count}, Loss: {running_loss:.5f}')
            #running_loss = 0.0

    #validation loss
    model.eval()
    outputs = model(input_tensor_valid.to(device))
    val_loss = criterion(outputs, target_tensor_valid.to(device))
    scheduler.step(val_loss)
    print(f'--> Validation loss: {val_loss.item():.5f}; learning rate {scheduler.get_last_lr()[0]}')

print('Finished Training')

torch.save(model, 'model_test.pt')

seismogram_data, clean_seismogram_data = data_generator(10, data, seq_length, sr, True)

# Run Monte Carlo Dropout to denoise the test seismogram
num_samples = 100
denoised_seismogram = monte_carlo_denoising(model, seismogram_data, num_samples=num_samples, num_components=num_components)

plt.figure()
plt.subplot(311)
plt.plot(clean_seismogram_data[0, :, 0])
plt.plot(clean_seismogram_data[0, :, 1])
plt.plot(clean_seismogram_data[0, :, 2])
plt.subplot(312)
plt.plot(seismogram_data[0, :, 0])
plt.plot(seismogram_data[0, :, 1])
plt.plot(seismogram_data[0, :, 2])
plt.subplot(313)
plt.plot(denoised_seismogram[:, 0])
plt.plot(denoised_seismogram[:, 1])
plt.plot(denoised_seismogram[:, 2])

plt.show()
