import torch.nn as nn
import torchvision.models as models
import torch

class DRQN(nn.Module):
    def __init__(self, config : dict, **kwargs):
        super().__init__()

        self.vision_module = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            test_input = torch.zeros((1, 3, 64, 64))
            test_output = self.vision_module(test_input)

        lstm_hidden_size = kwargs['lstm_hidden_size']
        lstm_num_layers = kwargs['lstm_num_layers']
        dropout = kwargs['dropout']
        device = kwargs['device']

        self.head = nn.LSTM(test_output.shape[1], lstm_hidden_size, lstm_num_layers, batch_first = True, dropout = dropout, device=device)
        self.lin_out = nn.Linear(lstm_hidden_size, 6)

        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.device = device

        self.prev_hidden = torch.zeros(self.lstm_num_layers, self.lstm_hidden_size, device=device)
        self.prev_state = torch.zeros(self.lstm_num_layers, self.lstm_hidden_size, device=device)
        
        if 'pretrained_weights' in kwargs:
            try:
                self.load_state_dict(kwargs['pretrained_weights'], strict=False)
            except RuntimeError as e:
                # Only load vision
                pretrained_dict = kwargs['pretrained_weights']
                model_dict = self.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.startswith('vision_module')}
                model_dict.update(pretrained_dict)
                self.load_state_dict(model_dict, strict=False)

    def reset_memory(self):
        """ resets hidden and cell state to initial parameters, should be called at the start of each episode"""
        self.prev_hidden = torch.zeros(self.lstm_num_layers, self.lstm_hidden_size, device=self.device)
        self.prev_state = torch.zeros(self.lstm_num_layers, self.lstm_hidden_size, device=self.device)

    def forward(self, x):
        # extract the 0,0 pixel from each image in the batch as the target color
        # we'll feed this directly to the head to help the network know what to look for
        x = x - 0.5
        
        x = self.vision_module(x)

        x, (h, c) = self.head(x, (self.prev_hidden, self.prev_state))
        self.prev_hidden = h.detach()
        self.prev_state = c.detach()
        x = self.lin_out(x)
        return x

__all__ = ["DRQN"]
