import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


SEGMENT_SAMPLES = 15600
N_MELS = 64
YAMNET_OUT_DIM = 1024
PROJ_DIM = 256
LSTM_HID = 128
LSTM_DIRECTIONS = 2
LSTM_OUT_DIM = LSTM_HID*LSTM_DIRECTIONS #256 dim
NUM_CLASSES = 1

class YamnetBackbone(nn.Module):
    def __init__(self,n_mels=N_MELS,out_dim=YAMNET_OUT_DIM):
        super().__init__()
        def block(c_in, c_out):
            return nn.Sequential(
                    nn.Conv2d(c_in, c_out, 3, padding=1),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(),
                    nn.Conv2d(c_out, c_out, 3, padding=1),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU())
    
        self.features = nn.Sequential(
            # ---- Block 1 (2 layers)
            block(1, 32),
            nn.MaxPool2d(2,2),
    
            # ---- Block 2 (2 layers)
            block(32, 64),
            nn.MaxPool2d(2,2),
    
            # ---- Block 3 (2 layers)
            block(64, 128),
            nn.MaxPool2d(2,2),
    
            # ---- Block 4 (2 layers)
            block(128, 256),
            nn.MaxPool2d(2,2),
    
            # ---- Block 5 (2 layers)
            block(256, 512),
            nn.AdaptiveAvgPool2d((1,1))
        )
    
        self.fc = nn.Linear(512, out_dim)

    def forward(self, logmel):
        x = logmel.unsqueeze(1)       # (B,1,n_mels,frames)
        x = self.features(x)          # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)     # (B, 512)
        x = self.fc(x)                # (B, out_dim)
        return x

class YamnetProjection(nn.Module): ##DENSE LAYER
    def __init__(self,in_dim=YAMNET_OUT_DIM,proj_dim=PROJ_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim,proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim)
        )
    def forward(self,eyam):
        return self.proj(eyam) #(B, PROJ_DIM)


class WaveEncoder(nn.Module):
    def __init__(self,input_dim=1,lstm_hidden=LSTM_HID,num_layers=2,bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=lstm_hidden,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional,
                            dropout=0.1)
    def forward(self,wav):
        #waveform in tensor float type (B,T)
        B, T = wav.shape
        x = wav.unsqueeze(-1) #(B,T,-1)
        out,_ = self.lstm(x) #Output: (B,T,hidden*directions)
        return out #(B,T,LSTM_OUT_DIM)


class ConditionalAttention(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
    def forward(self,Eyam,h_seq):
        # Eyam: (B,D)
        # h_seq: (B,T,D)
        # dot production dot(Eyam,h_t): (B,T)
        # elementwise multiply then sum over D
        # Eyam -> (B,1,D)
    
        Ey = Eyam.unsqueeze(1) #(B,1,D)
        dot = torch.sum(Ey*h_seq,dim=-1) #(B,T)
        scores = torch.tanh(dot)
        weights = F.softmax(scores,dim=-1)
        context = (weights.unsqueeze(-1)*h_seq).sum(dim=1) #(B,D)
        return context,weights #context: (B,D) || weights: (B,T)

class ClassifierHead(nn.Module):
    def __init__(self,in_dim=PROJ_DIM*2,num_classes=NUM_CLASSES,hidden=256,dropout=0.3):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes)
            )

    def forward(self,x):
        logits = self.model(x)
        return logits

class JointModel(nn.Module):
    def __init__(self,lstm_hidden=LSTM_HID,
               yamnet_out_dim=YAMNET_OUT_DIM,
               proj_dim=PROJ_DIM,
               n_mels=N_MELS,
               num_classes=NUM_CLASSES):
        super().__init__()
        self.wave_encoder = WaveEncoder(input_dim=1,
                                    lstm_hidden=lstm_hidden,
                                    num_layers=2,
                                    bidirectional=True)
        self.yamnet_backbone = YamnetBackbone(n_mels=n_mels,out_dim=yamnet_out_dim)
        self.yamnet_projection = YamnetProjection(in_dim=yamnet_out_dim,proj_dim=proj_dim)

        self.h_proj = nn.Sequential(
                    nn.Linear(lstm_hidden*2,proj_dim),
                    nn.ReLU(),
                    nn.LayerNorm(proj_dim)
                  )

        self.attention = ConditionalAttention(yamnet_out_dim,lstm_hidden)
        self.classifier = ClassifierHead(in_dim=proj_dim*2,hidden=256,num_classes=num_classes)

    def forward(self,wav,logmel):
      
        h_seq = self.wave_encoder(wav)
        h_seq_proj = self.h_proj(h_seq)
    
        eyam = self.yamnet_backbone(logmel)
        eyam_proj = self.yamnet_projection(eyam)
    
        context,weights = self.attention(eyam_proj,h_seq_proj)
    
        joint = torch.cat([context,eyam_proj],dim=1)
    
        logits = self.classifier(joint)
    
        return logits