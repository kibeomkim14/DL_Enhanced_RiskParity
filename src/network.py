import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.layer = nn.Sequential(
                            nn.Linear(46,72),
                            nn.Dropout(p=0.5),
                            nn.Linear(72,48),
                            nn.Dropout(p=0.5),
                            nn.Linear(48,32),
                            nn.Dropout(p=0.5),
                            nn.Linear(32,8)
                        )

    def weight_init(self):
        for layer in self.layer:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        output = self.layer(x)
        return output