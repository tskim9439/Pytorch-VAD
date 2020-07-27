#%%
import torch
import torch.nn as nn
# %%
class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim TxNxH into (TxN)xH, and applies to Module
        Allows handling of variable sequence lengths and minibatch size

        Args:
            module : Module to apply input to
        """
        super(SequenceWise, self).__init__()
        self.module = module
    
    def forward(self,x):
        t, n = x.size(0), x.size(1)
        x = x.view(t*n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x
    
    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(self.input_size)) if batch_norm else None

        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional, bias=False)

        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum().view(x.size(0), x.size(1), -1)
        return x

#%%
class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False):
        super(BRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                    bidirectional=bidirectional, bias=False)
        
        self.num_directions = 2 if bidirectional else 1
    
    def flatten_parameters(self):
        self.rnn.flatten_parameters()
    
    def forward(self, x):
        x, _ = self.rnn(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1) # (TxNxH * 2) -> (TxNxH) by sum
        return x

#%%
class rnn_base_autoencoder(nn.Module):
    def __init__(self, I, H, L=4, rnn_type=nn.LSTM):
        super(rnn_base_autoencoder, self).__init__()
        self.I = I
        self.H = H
        self.L = L
        self.rnn_type = rnn_type

        self.first_linear = nn.Conv1d(self.I, self.H, kernel_size=1, stride=1, padding=0)
        self.rnn_layers = nn.ModuleList([BRNN(input_size=H, hidden_size=H, 
                                rnn_type=rnn_type, bidirectional=True) for _ in range(self.L)])
        self.last_linear = nn.Conv1d(self.H, self.I, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.first_linear(x)
        x = x.transpose(1,2).transpose(0, 1) #Transpose NxHxT -> TxNxH
        for i in range(len(self.rnn_layers)):
            x = self.rnn_layers[i](x) + x
        
        x = x.transpose(0,1).transpose(1, 2) #Transepose TxNxH -> NxHxT
        x = self.last_linear(x)
        return x
#%%

class rnn_base_classifier(nn.Module):
    def __init__(self, I, H, L=3, rnn_type=nn.LSTM):
        super(rnn_base_classifier, self).__init__()
        self.I = I
        self.H = H
        self.L = L
        self.rnn_type = rnn_type

        self.first_linear = nn.Conv1d(self.I, self.H, kernel_size=1, stride=1, padding=0)
        self.rnn_layers = nn.ModuleList([BRNN(input_size=H, hidden_size=H, 
                                rnn_type=rnn_type, bidirectional=True) for _ in range(self.L)])
        self.last_linear = nn.Conv1d(self.H, 1, kernel_size=1, stride=1, padding=0)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        torch.autograd.set_detect_anomaly(True)
        x = x.view(x.shape[0], x.shape[-2], x.shape[-1])
        x = self.first_linear(x)
        x = x.transpose(1,2).transpose(0, 1) #Transpose NxHxT -> TxNxH
        for i in range(len(self.rnn_layers)):
            x = self.rnn_layers[i](x) + x
        
        x = x.transpose(0,1).transpose(1, 2) #Transepose TxNxH -> NxHxT
        x = self.last_linear(x)
        x = self.activation(x)
        x = x.squeeze(dim=1)
        return x

# %%
