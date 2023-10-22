import torch
import torch.nn as nn

# CLEEG model
class DANet(nn.Module):
    def __init__(self,n_chan=8,fc_in = 8,latsize = 64):
        super(DANet,self).__init__()
        self.n_chan = n_chan
        self.latsize = latsize
        self.conv1 = nn.Conv2d(1,fc_in,(n_chan,1),padding=0) # padding='valid'
        self.Bn1 = nn.BatchNorm2d(1)
        # self.In1 = nn.InstanceNorm2d(1,affine = True)
        self.FC = nn.Sequential(
            nn.Linear(fc_in,latsize),
            nn.Tanh(),
            nn.Linear(latsize,n_chan)
        )
    def forward(self,x):
        # encoder
        x = self.conv1(x)
        x = torch.permute(x,(0,2,1,3))
        x = self.Bn1(x)
        latent = torch.permute(x,(0,1,3,2))
        latent = self.FC(latent)
        out = torch.permute(latent,(0,1,3,2))
        return out

class DANet_temporoal(nn.Module):
    def __init__(self,n_chan=8,fc_in = 8,latsize = 64):
        super(DANet_temporoal,self).__init__()
        self.n_chan = n_chan
        self.latsize = latsize
        self.conv1 = nn.Conv2d(1,fc_in,(n_chan,1),padding=0) # padding='valid'
        self.conv2 = nn.Conv2d(1,1,(1,250),padding="same") # 1 second =  sfreq * 1
        self.Bn1 = nn.BatchNorm2d(1)
        self.Bn2 = nn.BatchNorm2d(1)
        self.FC = nn.Sequential(
            nn.Linear(fc_in,latsize),
            nn.Tanh(),
            nn.Linear(latsize,n_chan)
        )
    def forward(self,x):
        x = self.conv1(x)
        x = torch.permute(x,(0,2,1,3))
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)
        latent = torch.permute(x,(0,1,3,2))
        latent = self.FC(latent)
        out = torch.permute(latent,(0,1,3,2))
        return out

class DANet_linear(nn.Module):
    def __init__(self,n_chan=8,fc_in = 8,latsize = 64):
        super(DANet_linear,self).__init__()
        self.n_chan = n_chan
        self.latsize = latsize
        self.conv1 = nn.Conv2d(1,fc_in,(n_chan,1),padding=0) # padding='valid'
        self.Bn1 = nn.BatchNorm2d(1)
        self.FC = nn.Sequential(
            nn.Linear(fc_in,latsize),
            nn.Linear(latsize,n_chan)
        )
    def forward(self,x):
        x = self.conv1(x)
        x = torch.permute(x,(0,2,1,3))
        x = self.Bn1(x)
        latent = torch.permute(x,(0,1,3,2))
        latent = self.FC(latent)
        out = torch.permute(latent,(0,1,3,2))
        return out