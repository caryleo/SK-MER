import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        ##############
        # pe = pe.unsqueeze(0) # 我的实现这里不需要batch
        ##############
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
            x: (batch_size, seq_len, d_model)
        """
        # if print_dims:
        #     print("{0}: x: type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        # x = x + self.pe[:, :x.size(1)]
        ###################
        x = x + self.pe[:x.size(0), :x.size(1)]
        #################3#
        return self.dropout(x)