#drafted by chatgpt 4

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TopKPooling
from torch_geometric.data import Data, DataLoader
import pywt
import numpy as np
import matplotlib.pyplot as plt
import seisbench.data as sbd
from scipy import signal
from collections import OrderedDict

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNetv1(nn.Module):
    def __init__(self, drop=0, ncomp=1, fac=1):
        super(UNetv1, self).__init__()
        
        if ncomp == 1:
            in_channels = 2
        elif ncomp == 3:
            in_channels = 6

        self.conv1 = nn.Conv2d(in_channels, 8 * fac, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8 * fac)
        
        self.conv1b = nn.Conv2d(8 * fac, 8 * fac, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(8 * fac)
        self.drop1b = nn.Dropout(drop)
        
        self.conv2 = nn.Conv2d(8 * fac, 8 * fac, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(8 * fac)
        
        self.conv2b = nn.Conv2d(8 * fac, 16 * fac, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(16 * fac)
        self.drop2b = nn.Dropout(drop)
        
        self.conv3 = nn.Conv2d(16 * fac, 16 * fac, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(16 * fac)
        
        self.conv3b = nn.Conv2d(16 * fac, 32 * fac, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(32 * fac)
        self.drop3b = nn.Dropout(drop)
        
        self.conv4 = nn.Conv2d(32 * fac, 32 * fac, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32 * fac)
        
        self.conv4b = nn.Conv2d(32 * fac, 64 * fac, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm2d(64 * fac)
        self.drop4b = nn.Dropout(drop)
        
        self.conv5 = nn.Conv2d(64 * fac, 64 * fac, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(64 * fac)
        
        self.conv5b = nn.Conv2d(64 * fac, 128 * fac, kernel_size=3, padding=1)
        self.bn5b = nn.BatchNorm2d(128 * fac)
        self.drop5b = nn.Dropout(drop)

        self.upconv4 = nn.ConvTranspose2d(128 * fac, 64 * fac, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4u = nn.BatchNorm2d(64 * fac)

        self.conv4u = nn.Conv2d(128 * fac, 64 * fac, kernel_size=3, padding=1)
        self.bn4u2 = nn.BatchNorm2d(64 * fac)

        self.upconv3 = nn.ConvTranspose2d(64 * fac, 32 * fac, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3u = nn.BatchNorm2d(32 * fac)

        self.conv3u = nn.Conv2d(64 * fac, 32 * fac, kernel_size=3, padding=1)
        self.bn3u2 = nn.BatchNorm2d(32 * fac)

        self.upconv2 = nn.ConvTranspose2d(32 * fac, 16 * fac, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2u = nn.BatchNorm2d(16 * fac)

        self.conv2u = nn.Conv2d(32 * fac, 16 * fac, kernel_size=3, padding=1)
        self.bn2u2 = nn.BatchNorm2d(16 * fac)

        self.upconv1 = nn.ConvTranspose2d(16 * fac, 8 * fac, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1u = nn.BatchNorm2d(8 * fac)

        self.conv1u = nn.Conv2d(16 * fac, 8 * fac, kernel_size=3, padding=1)
        self.bn1u2 = nn.BatchNorm2d(8 * fac)

        if ncomp == 1:
            self.final = nn.Conv2d(8 * fac, 1, kernel_size=1)
        else:
            self.final = nn.Conv2d(8 * fac, 3, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc1 = F.elu(self.bn1(self.conv1(x)))
        enc1b = self.drop1b(F.elu(self.bn1b(self.conv1b(enc1))))
        
        enc2 = F.elu(self.bn2(self.conv2(enc1b)))
        enc2b = self.drop2b(F.elu(self.bn2b(self.conv2b(enc2))))
        
        enc3 = F.elu(self.bn3(self.conv3(enc2b)))
        enc3b = self.drop3b(F.elu(self.bn3b(self.conv3b(enc3))))
        
        enc4 = F.elu(self.bn4(self.conv4(enc3b)))
        enc4b = self.drop4b(F.elu(self.bn4b(self.conv4b(enc4))))
        
        enc5 = F.elu(self.bn5(self.conv5(enc4b)))
        enc5b = self.drop5b(F.elu(self.bn5b(self.conv5b(enc5))))

        dec4 = F.elu(self.bn4u(self.upconv4(enc5b)))
        dec4 = torch.cat([dec4, enc4b], dim=1)
        dec4 = F.elu(self.bn4u2(self.conv4u(dec4)))
        
        dec3 = F.elu(self.bn3u(self.upconv3(dec4)))
        dec3 = torch.cat([dec3, enc3b], dim=1)
        dec3 = F.elu(self.bn3u2(self.conv3u(dec3)))
        
        dec2 = F.elu(self.bn2u(self.upconv2(dec3)))
        dec2 = torch.cat([dec2, enc2b], dim=1)
        dec2 = F.elu(self.bn2u2(self.conv2u(dec2)))
        
        dec1 = F.elu(self.bn1u(self.upconv1(dec2)))
        dec1 = torch.cat([dec1, enc1b], dim=1)
        dec1 = F.elu(self.bn1u2(self.conv1u(dec1)))

        out = self.final(dec1)
        out = self.sigmoid(out)
        return out

# GCN-UNet model
class GCNUNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNUNet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.5)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.5)
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.unpool1 = TopKPooling(hidden_channels, ratio=2.0)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.unpool2 = TopKPooling(hidden_channels, ratio=2.0)
        self.conv6 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch):
        # Encoder
        x1 = F.relu(self.conv1(x, edge_index))
        x1, edge_index1, _, batch1, _ = self.pool1(x1, edge_index, batch=batch)
        x2 = F.relu(self.conv2(x1, edge_index1))
        x2, edge_index2, _, batch2, _ = self.pool2(x2, edge_index1, batch=batch1)
        
        # Bottleneck
        x3 = F.relu(self.conv3(x2, edge_index2))
        
        # Decoder
        x4 = self.unpool1(x3, edge_index2, batch2)
        x4 = F.relu(self.conv4(x4, edge_index2))
        x5 = self.unpool2(x4, edge_index1, batch1)
        x5 = F.relu(self.conv5(x5, edge_index1))
        
        x_out = self.conv6(x5, edge_index)
        
        return x_out

# GAT-UNet model
class GATUNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GATUNet, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.pool1 = TopKPooling(hidden_channels * heads, ratio=0.5)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.pool2 = TopKPooling(hidden_channels * heads, ratio=0.5)
        
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        
        self.conv4 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.unpool1 = TopKPooling(hidden_channels * heads, ratio=2.0)
        self.conv5 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        self.unpool2 = TopKPooling(hidden_channels * heads, ratio=2.0)
        self.conv6 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
    
    def forward(self, x, edge_index, batch):
        # Encoder
        x1 = F.elu(self.conv1(x, edge_index))
        x1, edge_index1, _, batch1, _ = self.pool1(x1, edge_index, batch=batch)
        x2 = F.elu(self.conv2(x1, edge_index1))
        x2, edge_index2, _, batch2, _ = self.pool2(x2, edge_index1, batch=batch1)
        
        # Bottleneck
        x3 = F.elu(self.conv3(x2, edge_index2))
        
        # Decoder
        x4 = self.unpool1(x3, edge_index2, batch2)
        x4 = F.elu(self.conv4(x4, edge_index2))
        x5 = self.unpool2(x4, edge_index1, batch1)
        x5 = F.elu(self.conv5(x5, edge_index1))
        
        x_out = self.conv6(x5, edge_index)
        
        return x_out

def dwt_transform(seismogram, wavelet='db1'):
    coeffs = [pywt.wavedec(channel, wavelet) for channel in seismogram.T]
    return np.concatenate([np.concatenate(c) for c in coeffs])

def create_graph(seismogram):
    node_features = dwt_transform(seismogram)
    num_nodes = len(node_features)
    x = torch.tensor(node_features, dtype=torch.float).unsqueeze(1)  # Unsqueeze to add feature dimension
    
    edge_index = []
    
    # Time-adjacent connections
    for j in range(num_nodes - 1):
        edge_index.append([j, j + 1])
        edge_index.append([j + 1, j])
        
        # Connections between wavelet levels (daughter wavelets)
        if (j + 1) % (num_nodes // 3) == 0 and (j + 1) < num_nodes - 1:
            edge_index.append([j, j + (num_nodes // 3)])
            edge_index.append([j + (num_nodes // 3), j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def data_2_input(stftinput1,stftinput2,stftinput3,norm_input):
    if norm_input:
        stftinput1=stftinput1/np.max(np.abs(stftinput1))
        stftinput2=stftinput2/np.max(np.abs(stftinput2))
        stftinput3=stftinput3/np.max(np.abs(stftinput3))
    tmp=np.zeros((1,stftinput1.shape[0],stftinput1.shape[1],6))
    tmp[0,:,:,0]=np.real(stftinput1)
    tmp[0,:,:,1]=np.imag(stftinput1)
    tmp[0,:,:,2]=np.real(stftinput2)
    tmp[0,:,:,3]=np.imag(stftinput2)
    tmp[0,:,:,4]=np.real(stftinput3)
    tmp[0,:,:,5]=np.imag(stftinput3)
    return tmp

def data_generator(batch_size, datavec, noisevec, itpvec, seq_length, sr, test=False):
    #data is a data set from the seisbench plateform. Pretty handy.
    #data format should be (batch, length, component). Raw seismograms

    #will need to adjust the components here. Hardwired to three.

    nd, _, _ = data_vec.shape
    nn, _, _ = noise_vec.shape
    sos = signal.butter(2, 2.5, 'hp', fs=sr, output='sos')

    #subset it once, or you get a "bus error" idk why
    #skip set of 500 each for later validation

    if not test:

        data_inds  = np.random.randint(nd - 1000, size=(batch_size,)) #this samples the data in OBST2024 dataset
        noise_inds = np.random.randint(nn - 1000, size=(batch_size,)) #this samples the noise in OBST2024 dataset

    else: #these are always the same

        data_inds  = np.arange(nd - 1000, nd - 1000 + batch_size)#np.random.randint(low=data_to_noise - 500, high=data_to_noise, size=(nd,)) #this samples the data in OBST2024 dataset
        noise_inds = np.arange(nn - 1000, nn - 1000 + batch_size)#np.random.randint(low=data_to_noise, high=data_to_noise + 500, size=(nd,)) #this samples the noise in OBST2024 dataset

    #reformat this to get random data from the OBST2024 dataset

    batch_sig  = np.zeros((batch_size,seq_length, 3))
    batch_noise = np.zeros((batch_size,seq_length, 3))

    for ii in range(batch_size):
        # window data, noise, and data+noise timeseries based on shift

        #The original data format was to have all three components in one vector, which is fucking bizare

        #make subsig1, subsig2, subsig3
        sig_valid = False
        while not sig_valid:

            amp = np.random.uniform(0,2)*(-1)**np.random.randint(low=1, high=10)#sometimes it should be very small
            sigwfs = datavec[data_inds[ii], :, :]

            if not np.array([ v.any() for v in sigwfs ]).all():
                continue

            itp = itpvec[data_inds[ii]]

            if np.isnan(itp):
                continue

            sigwfs = sigwfs[:, np.where(np.sum(sigwfs, axis=0))[0]]
            start_d = int(np.min( (np.max( (int(itp - seq_length/2), seq_length/2)), 3000 - seq_length - 64)))

            if not test:
                start_d = start_d + int(np.random.randint(low=-seq_length/2, high=seq_length/2))

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
            subsig1 = subsig1*(sig_amp/amp)
            subsig2 = subsig2*(sig_amp/amp)
            subsig3 = subsig3*(sig_amp/amp)

            sig_valid = True

        noise_valid = False
        while not noise_valid:

            amp = np.random.uniform(0.0,2)*(-1)**np.random.randint(low=1, high=10)

            noisewfs = noisevec[noise_inds[ii], :, :]

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

            subnoise1 = subnoise1*(sig_amp/amp)
            subnoise2 = subnoise2*(sig_amp/amp)
            subnoise3 = subnoise3*(sig_amp/amp)
            noise_valid = True

        #noise data
        batch_noise[ii, :, 0] = subnoise1
        batch_noise[ii, :, 1] = subnoise2
        batch_noise[ii, :, 2] = subnoise3

        #clean signals
        batch_sig[ii, :, 0] = subsig1
        batch_sig[ii, :, 1] = subsig2
        batch_sig[ii, :, 2] = subsig3

    return batch_sig, batch_noise

def stft_3comp_data_generator(model, batch_size, data_vec, noise_vec, sr, nperseg, noverlap, norm_input, eps=1e-9, nlen=128, valid=False, post=False):
    #data is a data set from the seisbench plateform. Pretty handy.
    #don't need meta data for the noise

    nd, _, _ = data_vec.shape
    nn, _, _ = noise_vec.shape

    nf = int(np.ceil(nperseg/2))

    while True:
        # grab batch

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

            #make subsig1, subsig2, subsig3
            sig_ind = np.random.randint(nd)
            sigwfs = data_vec[sig_ind, :, :]
            subsig1 = sigwfs[:, 0]
            subsig2 = sigwfs[:, 1]
            subsig3 = sigwfs[:, 2]

            noise_ind = np.random.randint(nn) #this samples the noise in OBST2024 dataset
            noisewfs = noise_vec[noise_ind, :, :]
            subnoise1 = noisewfs[:, 0]
            subnoise2 = noisewfs[:, 1]
            subnoise3 = noisewfs[:, 2]

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

model_type = 'UNet'

batch_size = 32
learning_rate = 0.0001
num_epochs = 100  # Default to 3 epochs
num_components = 3
hidden_dim = 1024*num_components #transformer only
wavelet = 'db1'
sr = 50
fac=1# model size
eps=1e-9 # epsilon
drop=0.1 # model drop rate
nlen=128# window length
nperseg=31 # param for STFT
noverlap=30 # param for STFT
norm_input=True # leave this alone
print_interval = 25  # Print every 5 batches
cmap='PuRd'

np.random.seed(200)
torch.manual_seed(200)

#for later plotting
f, t, _ = signal.stft(np.zeros(nlen),fs=sr,nperseg=31, noverlap=30)

data = sbd.OBST2024(sample_rate=sr) #sampling rate doesn't work

# Amount of samples to use per epoch
num_samples = batch_size*250
model = UNetv1(drop=drop, ncomp=num_components, fac=fac).to(device) #pytorch version of amanda's tensorflow UNet
print("CNN-UNet initialized with " + str(count_parameters(model)) + " trainable parameters")

#35393 is the index between data and noise in OBST2024

total, _ = data.metadata.shape

data_vec  = data.get_waveforms(np.arange(0, 35393))[:,0:3,:]
itpvec    = np.round(data.metadata.trace_p_arrival_sample[:35393].to_numpy())*(sr/100)#they are valid for 100 Hz but OBST2024 does't scale when you load it, you have to do it
noise_vec = data.get_waveforms(np.arange(total - 35393, total))[:,0:3,:]

print('---> Making the validation dataset')
seismogram_data_valid, noise_data_valid = data_generator(1000, data_vec, noise_vec, itpvec,  nlen, sr)

my_test_data = stft_3comp_data_generator('v1',1000,seismogram_data_valid,noise_data_valid,sr,nperseg,noverlap,norm_input,eps,nlen)#,valid=True, post=False)
input_data_valid,target_data_valid = next(my_test_data)

input_tensor_valid = torch.from_numpy(input_data_valid).float().to(device)
target_tensor_valid = torch.from_numpy(target_data_valid).float().to(device)

# Check if the data needs reordering (assuming data is in channels-last format)
if input_tensor_valid.shape[-1] < input_tensor_valid.shape[-2]:  # A simple check that may apply if channels < width
    input_tensor_valid = input_tensor_valid.permute(0, 3, 1, 2)  # Reorder dimensions to channels-first
    target_tensor_valid = target_tensor_valid.permute(0, 3, 1, 2)  # Reorder dimensions to channels-first

# Define loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.25, patience=2, eps=1e-14, threshold=0.0001)

# Training loop with shuffling
for epoch in range(num_epochs):
    running_loss = 0.0
    batch_count = 0  # Counter for batches

    #This step is slow. But hardly the bottle-neck compared with training. Do more batches if too slow, data is heavily augmented.
    print('-----> Epochs# ' + str(epoch))
    seismogram_data, noise_data = data_generator(num_samples, data_vec, noise_vec, itpvec, nlen, sr)

    my_test_data = stft_3comp_data_generator('v1',num_samples,seismogram_data, noise_data,sr,nperseg,noverlap,norm_input,eps,nlen)
    input_data,target_data = next(my_test_data)
    
    input_tensor  = torch.from_numpy(input_data).float().to(device)
    target_tensor = torch.from_numpy(target_data).float().to(device)
    # Check if the data needs reordering (assuming data is in channels-last format)

    if input_tensor.shape[-1] < input_tensor.shape[-2]:  # A simple check that may apply if channels < width
        input_tensor = input_tensor.permute(0, 3, 1, 2)  # Reorder dimensions to channels-first
        target_tensor = target_tensor.permute(0, 3, 1, 2)  # Reorder dimensions to channels-first

    model.train()

    for i in range(0, num_samples, batch_size):

        optimizer.zero_grad()

        outputs = model(input_tensor[i:i+batch_size, :, :, :])
        loss    = criterion(outputs, target_tensor[i:i+batch_size, :, :, :])

        loss.backward()
        optimizer.step()

        running_loss = (running_loss*batch_count + loss.item())/(batch_count + 1)

        batch_count += 1  # Increment batch counter

        if batch_count % print_interval == 0:  # Print every `print_interval` batches
            print(f'Epoch {epoch + 1}, Batch {batch_count}, Loss: {running_loss:.5f}')
            #running_loss = 0.0

    #validation loss
    model.eval()

    running_val_loss = 0.0
    batch_count = 0  # Counter for batches

    for i in range(0, 1000, batch_size):

        optimizer.zero_grad()

        outputs = model(input_tensor_valid[i:i+batch_size, :, :, :])
        loss    = criterion(outputs, target_tensor_valid[i:i+batch_size, :, :, :])

        loss.backward()
        optimizer.step()

        running_val_loss = (running_val_loss*batch_count + loss.item())/(batch_count + 1)

        batch_count += 1  # Increment batch counter
    
    mask = model(input_tensor_valid[[0], :, :, :]).detach().numpy()
    mask = mask[0,0,:,:]
    mask_true = target_data_valid[0,:,:,0]
    trace = input_data_valid[0,:,:,0] + input_data_valid[0,:,:,1]*1j

    _, trace_in = signal.istft(trace,fs=sr,nperseg=31, noverlap=30)
    _, trace_pred = signal.istft(trace*mask,fs=sr,nperseg=31, noverlap=30)
    _, trace_true = signal.istft(trace*mask_true,fs=sr,nperseg=31, noverlap=30)

    amp = np.max(np.abs(trace_in))

    plt.figure()
    plt.subplot(211)
    plt.plot(t, trace_in)
    plt.ylim([ -amp*1.2, amp*1.2 ])
    plt.grid()
    plt.subplot(212)
    plt.plot(t, trace_true, 'k')
    plt.plot(t, trace_pred, 'r', linewidth=0.5)
    plt.ylim([ -amp*1.2, amp*1.2 ])
    plt.grid()
    plt.savefig('Validation_Epoch' + str(epoch) + '.png')
    plt.close()

    #and plot the masks because something weird is going on
    plt.figure()
    plt.subplot(211)
    plt.pcolormesh(t, f, mask_true, cmap=cmap, vmin=0, vmax=1, shading='auto')
    plt.subplot(212)
    plt.pcolormesh(t, f, mask, cmap=cmap, vmin=0, vmax=1, shading='auto')
    plt.savefig('Validation_Epoch' + str(epoch) + 'Mask.png')
    plt.close()

    scheduler.step(running_val_loss)
    print(f'--> Validation loss: {running_val_loss:.5f}; learning rate {scheduler.get_last_lr()[0]}')

    #save current model
    torch.save(model.state_dict(), 'model_test-CNN_UNet_fac' + str(fac) +'.pt')

print('Finished Training')

""" 
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
                    continue

                itp = itpvec[sig_ind]

                if np.isnan(itp):
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

                else:

                    noise_ind  = np.random.randint(low=nn - 500+5, high=nn) #this samples the data in OBST2024 dataset
                    #amp = 2*ii/batch_size + 0.5#get's noiser and noiser

                amp = np.random.uniform(-2,2)

                noisewfs = noise_vec[noise_ind, :, :]

                if not np.array([ v.any() for v in noisewfs ]).all():
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
 
 
            
class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, fac=1, dropout_rate=0.5, activation_fn=nn.ELU(inplace=True)):
        super(UNet, self).__init__()

        # Initialize features based on the scaling factor
        features = 4*fac
        self.encoder1 = UNet._block(in_channels, features, name="enc1", activation_fn=activation_fn, dropout_rate=dropout_rate)
        self.encoder2 = UNet._block(features, features * 2, name="enc2", stride=2, activation_fn=activation_fn, dropout_rate=dropout_rate)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3", stride=2, activation_fn=activation_fn, dropout_rate=dropout_rate)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4", stride=2, activation_fn=activation_fn, dropout_rate=dropout_rate)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck", stride=2, activation_fn=activation_fn, dropout_rate=dropout_rate)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4", activation_fn=activation_fn, dropout_rate=dropout_rate)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3", activation_fn=activation_fn, dropout_rate=dropout_rate)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2", activation_fn=activation_fn, dropout_rate=dropout_rate)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1", activation_fn=activation_fn, dropout_rate=dropout_rate)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name, stride=1, activation_fn=nn.ReLU(inplace=True), dropout_rate=0.0):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                            stride=stride
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", activation_fn),
                    (name + "dropout1", nn.Dropout(p=dropout_rate)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", activation_fn),
                    (name + "dropout2", nn.Dropout(p=dropout_rate)),
                ]
            )
        )

 
 """