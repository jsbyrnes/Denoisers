#####Model Zoo

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
#import pywt
import numpy as np

# Positional Encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
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
    def __init__(self, input_dim, num_layers=6, num_heads=8, hidden_dim=512, dropout=0.1, max_len=128):
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

class UNet_3CT(nn.Module):
    def __init__(self, drop=0.1, ncomp=1, fac=1):
        super(UNet_3CT, self).__init__()
        
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

        self.transformer = CustomTransformer(128 * fac, num_layers=3, num_heads=2, hidden_dim=128 * fac, dropout=drop)

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

        #bottleneck
        v = enc5b.squeeze()

        #correction for nbatch = 1
        if len(v.shape)==2:
            v = v.unsqueeze(0)

        transformed = self.transformer(torch.permute(v, [0, 2, 1 ]))
        transformed = torch.permute(transformed, [0, 2, 1]).unsqueeze(2)

        dec4 = F.elu(self.bn4u(self.upconv4(transformed)))
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

class UNet_xC(nn.Module):
    def __init__(self, drop=0.1, ncomp=1, fac=1):
        super(UNet_4CT, self).__init__()
        
        if ncomp == 1:
            in_channels = 2
        elif ncomp == 3:
            in_channels = 6
        elif ncomp == 4:
            in_channels = 8

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

        self.final = nn.Conv2d(8 * fac, ncomp, kernel_size=1)

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

        #bottleneck
        v = enc5b.squeeze()

        #correction for nbatch = 1
        if len(v.shape)==2:
            v = v.unsqueeze(0)

        transformed = self.transformer(torch.permute(v, [0, 2, 1 ]))
        transformed = torch.permute(transformed, [0, 2, 1]).unsqueeze(2)

        dec4 = F.elu(self.bn4u(self.upconv4(transformed)))
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

class UNet_4CT(nn.Module):
    def __init__(self, drop=0.1, ncomp=1, fac=1):
        super(UNet_4CT, self).__init__()
        
        if ncomp == 1:
            in_channels = 2
        elif ncomp == 3:
            in_channels = 6
        elif ncomp == 4:
            in_channels = 8

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

        self.transformer = CustomTransformer(128 * fac, num_layers=3, num_heads=2, hidden_dim=128 * fac, dropout=drop)

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

        self.final = nn.Conv2d(8 * fac, ncomp, kernel_size=1)

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

        #bottleneck
        v = enc5b.squeeze()

        #correction for nbatch = 1
        if len(v.shape)==2:
            v = v.unsqueeze(0)

        transformed = self.transformer(torch.permute(v, [0, 2, 1 ]))
        transformed = torch.permute(transformed, [0, 2, 1]).unsqueeze(2)

        dec4 = F.elu(self.bn4u(self.upconv4(transformed)))
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