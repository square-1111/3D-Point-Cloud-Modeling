from torch import nn
import torch

class Encoder(nn.Module):

    def __init__(self,*args):
        """
        An Encoder (recognition network), which mapes inputs onto a latent space

        Args:
        *args : dictionary of parameters
        """

        super(Encoder, self).__init__()

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


        self.symmetry = param['symmetry']
        self.closing = param['closing']
        
            
        if verbose:
            print("Building Encoder")
        
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
            self.model.append(non_linearity().cuda())
            # layer = (layer)
        
        if pool is not None and pool_sizes is not None:
            if pool_sizes[i] is not None:
                self.model.append(
                    pool(kernel_size=pool_sizes[i]).cuda())
        
        if dropout_prob is not None and dropout_prob > 0:
            self.model.append(nn.Dropout(1 - dropout_prob).cuda())
        

    def forward(self, X):

        for layer in self.model:
                X = layer(X)

        if self.symmetry is not None:
            X = self.symmetry(X, axis=-1).values
        
        if self.closing is not None:
            X = closing(X).cuda()

        return X

class DecoderWithFC(nn.Module):
    
    def __init__(self, *args):
        """
        An Decoder (recognition network), which mapes inputs onto a latent space
        """

        super(DecoderWithFC, self).__init__()

        param = args[0]
        verbose = param['verbose']
        layer_sizes = param['layer_sizes'] 
        n_layers = len(layer_sizes)
        dropout_prob = param['dropout_prob']
        b_norm = param['b_norm']
        non_linearity = param['non_linearity']
        b_norm_finish = param['b_norm_finish']
        

        if verbose:
            print("Building Decoder")

        self.model = nn.ModuleList()
        for i in range(n_layers-1):
            
            self.model.append(
                nn.Linear(in_features=layer_sizes[i], 
                            out_features=layer_sizes[i+1]).cuda())
            
            if b_norm:
                self.model.append(
                    nn.BatchNorm1d(num_features=n_filters[i+1]).cuda())
            
            if non_linearity is not None:
                self.model.append(non_linearity().cuda())
            
            if dropout_prob is not None and dropout_prob > 0:
                self.model.append(nn.Dropout(1 - dropout_prob).cuda())

        self.model.append(
            nn.Linear(in_features=layer_sizes[-1], 
                        out_features=layer_sizes[-1]).cuda())

        if b_norm_finish: 
            self.model.append(
                    nn.BatchNorm1d(num_features=n_filters[-1]).cuda())
        
    def forward(self, X):
        for layer in self.model:
            X = layer(X)
        return X

class AutoEncoder(nn.Module):
    """Combining the Encoder and Decoder Architecture"""
    def __init__(self, *args):

        super(AutoEncoder, self).__init__()

        encoder_args = args[0]
        decoder_args = args[1]

        self.encoder = Encoder(encoder_args)
        self.decoder = DecoderWithFC(decoder_args)
    
    def forward(self, X):
        latent = self.encoder(X)
        output = self.decoder(latent)
        return torch.reshape(output, (-1, 2048, 3))
