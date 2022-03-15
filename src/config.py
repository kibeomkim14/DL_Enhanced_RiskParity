import torch

NUM_EPOCHS = 200
PARAM_PATH = '/Users/mac/Desktop/PycharmProjects/TAADL/models/'
DATA_PATH  = '/Users/mac/Desktop/PycharmProjects/TAADL/input/'


# define feature vector, which is speical to each time series
# first 5: stock, gov bond, corp bond, real estate, commodity
# last 3 : us, europe, other
e1 = torch.Tensor([1,0,0,0,0,0,1,0]) # IEV, europe
e2 = torch.Tensor([1,0,0,0,0,0,0,1]) # EEM, em. market
e3 = torch.Tensor([0,0,1,0,0,1,0,0]) # AGG, us corp bonds
e4 = torch.Tensor([0,1,0,0,0,1,0,0]) # IEF, us gov bonds
e5 = torch.Tensor([0,0,0,1,0,1,0,0]) # IYR, us real estates
e6 = torch.Tensor([0,0,0,0,1,0,0,0]) # IAU, gold
e7 = torch.Tensor([1,0,0,0,0,1,0,0]) # ITOT, us equities

e_dict = {0: e1, 1: e2, 2: e3, 3: e4, 4: e5, 5: e6, 6: e7}