# agents/ctqn.py
import torch
import torch.nn as nn

class CTQN(nn.Module):
    def __init__(self, config: dict, **kwargs):
        super().__init__()

        # vision encoder
        self.vision_module = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Flatten()
        )

        test_input = torch.zeros((1, 3, 64, 64))
        vis_out = self.vision_module(test_input).shape[1]

        embed_dim = config["transformer_hidden_size"]

        self.input_linear = nn.Linear(vis_out + 3, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=config["transformer_num_heads"],
            dropout=config["transformer_dropout"],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["transformer_num_layers"]
        )

        self.output_head = nn.Linear(embed_dim, 6)

    def reset_memory(self):
        return

    def forward(self, x):
        target_color = x[:, :, 0, 0]
        x = x - 0.5
        x = self.vision_module(x)
        x = torch.cat((x, target_color), dim=1)
        x = self.input_linear(x).unsqueeze(1)  # Add sequence dim

        x = self.transformer(x)
        x = self.output_head(x[:, -1])
        return x

__all__ = ["CTQN"]
