import torch.nn as nn
import torchvision.models as models
import torch
from gtrxl_torch import GTrXL as GTrXL_head

class GTrXL(nn.Module):
    def __init__(self, config : dict, **kwargs):
        super().__init__()

        self.vision_module = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32)
        )

        WIDTH, HEIGHT = 64, 64
        OUTPUTS = 6
        linear_input_size = int(WIDTH * HEIGHT * 32 / 4)
        self.batch_norm_size = 32
        self.output_height = 32

        hidden_size = kwargs['gtrxl_hidden_size']
        num_heads = kwargs['num_heads']
        transformer_num_layers = kwargs['trans_num_layers']
        gru_num_layers = kwargs['gru_num_layers']
        embedding_size = kwargs['embedding_size']
        device = kwargs['device']

        self.lin_embed = nn.Linear(linear_input_size, embedding_size)

        # using https://github.com/alantess/gtrxl-torch/tree/main implementation of gtrxl
        self.head = GTrXL_head(d_model = embedding_size,
                               nheads = num_heads,
                               transformer_layers = transformer_num_layers,
                               hidden_dims = hidden_size,
                               n_layers = gru_num_layers,
                               batch_first = True)
        self.lin_out = nn.Linear(embedding_size, OUTPUTS)

        self.hidden_size = hidden_size
        self.device = device
        
        if 'pretrained_weights' in kwargs:
            self.load_state_dict(kwargs['pretrained_weights'])

    def reset_memory(self):
        """ resets hidden and cell state to initial parameters, should be called at the start of each episode"""
        return

    def forward(self, x):
        #center the color values around 0 for the cnn
        x = x - 0.5
        
        # CNN
        B, T, C, H, W = x.shape
        x_reshaped = x.reshape(B*T, C, H, W)
        x_reshaped = self.vision_module(x_reshaped)
        x_flat = torch.flatten(x_reshaped, start_dim=1)
        x_flat = x_flat.reshape(B,T,-1)

        x_embed = self.lin_embed(x_flat)

        x = self.head(x_embed)
        x = self.lin_out(x)
        return x[:,-1,:] # return only last q-values in batches

__all__ = ["GTrXL"]
