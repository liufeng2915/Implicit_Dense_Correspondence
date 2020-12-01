
import torch

occ_criterion = torch.nn.MSELoss()
def occupancy_loss(esti_values, values):
    return occ_criterion(esti_values, values)
