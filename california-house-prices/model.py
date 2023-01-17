import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, init_weights=True):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(658, 384),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(64, 1)
        )

        if init_weights:
            self.initialize_weights()

    def forward(self, x):
        return self.mlp(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
