
import torch

criterion = torch.nn.MSELoss()
def occupancy_loss(esti_values, values):
    return criterion(esti_values, values)

def selfrec_loss(esti_shape, shape):
    return criterion(esti_shape, shape)

