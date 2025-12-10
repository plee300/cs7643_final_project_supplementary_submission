import torch.nn as nn
import torchvision.models as models
import torch

class DRQN(nn.Module):
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
        
        self.flatten = nn.Flatten()

        WIDTH, HEIGHT = 64, 64
        OUTPUTS = 6
        linear_input_size = int(WIDTH * HEIGHT * 32 / 4 + 3)  # +3 for the target color input

        lstm_hidden_size = kwargs['lstm_hidden_size']
        lstm_num_layers = kwargs['lstm_num_layers']
        dropout = kwargs['dropout']
        device = kwargs['device']

        self.head = nn.LSTM(linear_input_size, lstm_hidden_size, lstm_num_layers, batch_first = True, dropout = dropout, device=device)
        self.lin_out = nn.Linear(lstm_hidden_size, OUTPUTS)

        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.device = device

        # self.prev_hidden = torch.zeros(self.lstm_num_layers, self.lstm_hidden_size, device=device)
        # self.prev_state = torch.zeros(self.lstm_num_layers, self.lstm_hidden_size, device=device)
        # self.soft = nn.Softmax()
        
        if 'pretrained_weights' in kwargs:
            self.load_state_dict(kwargs['pretrained_weights'])

    def reset_memory(self):
        """ resets hidden and cell state to initial parameters, should be called at the start of each episode"""
        return

    def forward(self, x):
        # extract the 0,0 pixel from each image in the batch as the target color
        # we'll feed this directly to the head to help the network know what to look for
        prev_hidden = torch.zeros(self.lstm_num_layers, self.lstm_hidden_size, device=self.device).detach()
        prev_state = torch.zeros(self.lstm_num_layers, self.lstm_hidden_size, device=self.device).detach()
        for t in range(x.shape[1]):
            x_t = x[:, t, :, :, :]

            target_color = x_t[:, :, 0, 0]
            
            #center the color values around 0 for the cnn
            x_t = x_t - 0.5
            
            x_t = self.vision_module(x_t)
            x_t = self.flatten(x_t)
            x_t = torch.cat((x_t, target_color), dim=1)

            x_t, (h, c) = self.head(x_t, (prev_hidden, prev_state))
            prev_hidden = h.detach()
            prev_state = c.detach()
        x = self.lin_out(x_t)
        return x

__all__ = ["DRQN"]
