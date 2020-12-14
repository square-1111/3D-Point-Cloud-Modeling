from ae_model import DecoderWithFC
import torch
from torch import nn

class latentGenerator(nn.Module):
    """Creating Generator"""
    def __init__(self, *args):

        super(latentGenerator, self).__init__()
        
        params = args[0]

        self.generator = DecoderWithFC(params)
        
    
    def forward(self, X):
        out_signal = self.generator(X)
        return torch.reshape(out_signal, (-1, 2048, 3))

class latentDiscriminator(nn.Module):
    """Creating Discriminator"""
    def __init__(self, *args):

        super(latentDiscriminator, self).__init__()

        params = args[0]
        
        self.flatten = nn.Flatten()

        self.fc = DecoderWithFC(params)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, X):
        X_flat = self.flatten(X).float()
        d_logit = self.fc(X_flat)
        d_prob = self.sigmoid(d_logit)
        return d_prob 