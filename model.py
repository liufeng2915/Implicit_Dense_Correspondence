import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import os

class Encoder(nn.Module):
    def __init__(self, channel=3, z_dim=256):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)       
        self.fc2 = nn.Linear(512,  z_dim)
        self.relu = nn.ReLU(inplace=True)
                
    def forward(self, x):
        # x:  B*num_points*3

        x = x.transpose(2,1).contiguous()
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        x = F.max_pool1d(x, x.shape[2])
        x = x.view(-1, 1024)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class ImplicitFun(nn.Module):
    def __init__(self, z_dim=256, num_branches=12):
        super(ImplicitFun, self).__init__()
        input_dim = z_dim+3
        #
        self.sd_fc1 = nn.Linear(input_dim, 3072)
        self.sd_fc2 = nn.Linear(3072, 384)
        self.sd_fc3 = nn.Linear(384, num_branches)

    def forward(self, z, points):

        num_pts = points.shape[1]
        z = z.unsqueeze(1).repeat(1, num_pts, 1)
        pointz = torch.cat((points, z), dim=2)

        x1 = F.leaky_relu(self.sd_fc1(pointz), 0.02)
        x2 = F.leaky_relu(self.sd_fc2(x1), 0.02)
        x3 = torch.sigmoid(self.sd_fc3(x2))
        x4 = F.max_pool1d(x3, x3.shape[2])

        return x3, x4


class InverseImplicitFun(nn.Module):
    def __init__(self, z_dim=256, num_branches=12):
        super(InverseImplicitFun, self).__init__()

        self.fc1 = nn.Linear(z_dim+num_branches, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512+z_dim+num_branches, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, 3)


    def forward(self, z, values):

        z = z.unsqueeze(1).repeat(1, values.shape[1], 1)
        valuez = torch.cat((z, values), dim=2)

        x1 = self.fc1(valuez)
        x1 = F.leaky_relu(x1, negative_slope=0.02, inplace=True)
        x2 = self.fc2(x1)
        x2 = F.leaky_relu(x2, negative_slope=0.02, inplace=True)
        x3 = self.fc3(x2)
        x3 = F.leaky_relu(x3, negative_slope=0.02, inplace=True)
        x4 = self.fc4(x3)
        x4 = F.leaky_relu(x4, negative_slope=0.02, inplace=True)
        x4 = torch.cat((x4, valuez), dim=2)
        x5 = self.fc5(x4)
        x5 = F.leaky_relu(x5, negative_slope=0.02, inplace=True)
        x6 = self.fc6(x5)
        x6 = F.leaky_relu(x6, negative_slope=0.02, inplace=True)
        x7 = self.fc7(x6)
        x7 = F.leaky_relu(x7, negative_slope=0.02, inplace=True)
        x8 = self.fc8(x7)
        x8 = torch.tanh(x8)

        return x8

def checkpoint(config, epoch, Encoder=None, ImplicitFun=None, InverseImplicitFun=None):
    model_path = config.checkpoint_dir + '/' + config.cate_name + '/' + '/Corr-' + str(epoch) + '.pth'

    if not os.path.exists(config.checkpoint_dir + '/' + config.cate_name):
        os.makedirs(config.checkpoint_dir + '/' + config.cate_name)
    if InverseImplicitFun==None:
        torch.save({
                    'Encoder_state_dict': Encoder.state_dict(),
                    'ImplicitFun_state_dict': ImplicitFun.state_dict()
                    }, model_path)
    else:
        torch.save({
                    'Encoder_state_dict': Encoder.state_dict(),
                    'ImplicitFun_state_dict': ImplicitFun.state_dict(),
                    'InverseImplicitFun_state_dict': InverseImplicitFun.state_dict()
                    }, model_path) 