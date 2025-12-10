import torch.nn as nn
import torch

class VisualDQN(nn.Module):
    def __init__(self, config : dict, **kwargs):
        super().__init__()

        self.vision_module = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        test_input = torch.zeros((1, 3, 64, 64))
        test_output = self.vision_module(test_input)

        self.head = nn.Sequential(
            nn.Linear(test_output.shape[1]+3, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
        if 'pretrained_weights' in kwargs:
            self.load_state_dict(kwargs['pretrained_weights'], strict=False)

    def reset_memory(self):
        '''placeholder method for function call in test_env.py'''
        return

    def forward(self, x):
        # extract the 0,0 pixel from each image in the batch as the target color
        # we'll feed this directly to the head to help the network know what to look for
        target_color = x[:, :, 0, 0]
        
        #center the color values around 0 for the cnn
        x = x - 0.5
        
        x = self.vision_module(x)
        x = torch.cat((x, target_color), dim=1)
        x = self.head(x)
        return x

__all__ = ["VisualDQN"]
