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
        self.mem_length = config.get("mem_length", 1)

        if 'pretrained_weights' in kwargs:
            self.load_state_dict(kwargs['pretrained_weights'])

    def reset_memory(self):
        return

    def forward(self, x):
        # x: (batch, seq_len, 3, 64, 64)
        B, T, C, H, W = x.shape

        # Flatten memory into batch
        x = x.reshape(B * T, C, H, W)

        # Extract target color (3 dims)
        target_color = x[:, :, 0, 0]  # shape (B*T, 3)

        # Normalize
        x = x - 0.5
        x = self.vision_module(x)  # (B*T, vis_out)

        # Add target color
        x = torch.cat((x, target_color), dim=1)

        # Embed
        x = self.input_linear(x)  # (B*T, embed_dim)

        # Reshape back into sequences
        x = x.reshape(B, T, -1)  # (B, T, embed_dim)

        # Transformer
        x = self.transformer(x)  # (B, T, embed_dim)

        # Q-values for last token
        x = x[:, -1]  # (B, embed_dim)
        x = self.output_head(x)  # (B, 6)

        return x

__all__ = ["CTQN"]
