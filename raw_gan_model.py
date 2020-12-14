from ae_model import DecoderWithFC
import torch
from torch import nn

class rawGenerator(nn.Module):
    """Creating Generator"""
    def __init__(self, *args):

        super(rawGenerator, self).__init__()

        params = args[0]

        self.generator = DecoderWithFC(params)
    
    def forward(self, X):
        out_signal = self.generator(X)
        relu = nn.ReLU()
        return torch.reshape(relu(out_signal), (-1, 2048, 3))

class rawDiscriminator(nn.Module):

    def __init__(self,*args):
        """
        Architecture of Discriminator for Raw GAN network

        Args:
        *args : dictionary of parameters
        """

        super(rawDiscriminator, self).__init__()

        param = args[0]
        n_filters = param['n_filters']
        verbose = param['verbose']
        filter_sizes = param['filter_sizes']
        stride = param['stride']
        padding = param['padding']
        padding_mode = param['padding_mode']
        b_norm = param['b_norm']
        non_linearity = param['non_linearity']
        pool = param['pool']
        pool_sizes = param['pool_sizes']
        dropout_prob = param['dropout_prob']
        linear_layer = param['linear_layer']
        num_linear = len(linear_layer)


        self.closing = param['closing']
        
            
        if verbose:
            print("Building Discriminator")
        
        n_layers = len(n_filters)
        filter_sizes = filter_sizes * n_layers
        strides = stride * n_layers

        self.model = nn.ModuleList()

        for i in range(n_layers-1):
        
            self.model.append(
                nn.Conv1d(in_channels=n_filters[i], out_channels=n_filters[i+1],
                            kernel_size=filter_sizes[i], stride=strides[i],
                            padding=padding, padding_mode=padding_mode).cuda())
            
            if b_norm:
                self.model.append(
                    nn.BatchNorm1d(num_features=n_filters[i+1]).cuda())
            
            if non_linearity is not None:
                self.model.append(non_linearity(0.2).cuda())
                # layer = (layer)
            
            if pool is not None and pool_sizes is not None:
                if pool_sizes[i] is not None:
                    self.model.append(
                        pool(kernel_size=pool_sizes[i]).cuda())
            
            if dropout_prob is not None and dropout_prob > 0:
                self.model.append(nn.Dropout(1 - dropout_prob).cuda())
            
            self.model.append(nn.Flatten().cuda())

        for i in range(num_linear - 1):
            self.model.append(nn.Linear(in_features=linear_layer[i],
                                        out_features=linear_layer[i+1]).cuda())
            
            if non_linearity is not None:
                self.model.append(non_linearity(0.2).cuda())
        
        self.model.append(nn.Sigmoid())
        

    def forward(self, X):
        for layer in self.model:
            X = layer(X)
        
        if self.closing is not None:
            X = closing(X).cuda()

        return X