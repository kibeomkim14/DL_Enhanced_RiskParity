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


class PortfolioLayer(nn.Module):
    def __init__(self, asset_num:int=8, error_adjusted:bool=True):
        super(self,).__init__()
        self.asset_num = asset_num
        if error_adjusted:
            self.input_dim = asset_num * 3
        else: 
            self.input_dim = asset_num * 2

        self.layer = nn.Sequential(
                            nn.Linear(self.input_dim,self.asset_num+1), # account for the cash position
                            nn.Softmax(dim=0)
                        )
    
    def forward(self, input):
        output = self.layer(input)
        return output