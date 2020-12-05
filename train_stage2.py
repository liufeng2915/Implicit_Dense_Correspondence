

import argparse
import numpy as np 
from model import *
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import datetime
import time
from dataset import *
from loss import *


parser = argparse.ArgumentParser()
parser.add_argument('--cate_name', type=str, default='chair', help='category name')
parser.add_argument('--data_dir', type=str, default='./data/ShapeNet/', help='Root directory of dataset') 
parser.add_argument('--resolution', type=int, default=64, help='voxel resolution')
parser.add_argument('--epoch', type=int, default=1000, help='Epoch to train')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate of for adam')
parser.add_argument('--beta1', type=float, default=0.5, help='Momentum term of adam')
parser.add_argument('--batch_size', type=int, default=6, help='training batch size')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='Directory name to save the checkpoints')
parser.add_argument('--pretrain_model', type=bool, default=False, help='load pretrained model')
parser.add_argument('--pretrain_model_name', type=str, default='Corr-249.pth', help='pretrained model')
parser.add_argument('--cuda', type=bool, default=True, help='True for GPU, False for CPU')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--logs_dir', type=str, default='./logs', help='Root directory of samples')
parser.add_argument('--signature', default=str(datetime.datetime.now()))


# #
config = parser.parse_args()
if config.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
torch.manual_seed(config.seed)
if config.cuda:
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = True
print(config)

sig = config.signature
writer = SummaryWriter('%s/%s/%s' % (config.logs_dir, config.cate_name, sig))


# # data loader
data_list = generate_list(config.data_dir+config.cate_name)
train_data = ShapeNet(data_list=data_list, data_reso=config.resolution)
train_loader = DataLoader(train_data, num_workers=4, batch_size=config.batch_size, shuffle=True)


# # network
Encoder = Encoder()
ImplicitFun = ImplicitFun() 
InverseImplicitFun = InverseImplicitFun()
print(Encoder, ImplicitFun, InverseImplicitFun)

# load pre-trained model
counter = 0
if config.pretrain_model:
    counter = int(config.pretrain_model_name.split('-')[-1][:-4])
    all_model = torch.load(config.checkpoint_dir + '/' + config.cate_name + '/' + config.pretrain_model_name)
    Encoder.load_state_dict(all_model['Encoder_state_dict'])
    ImplicitFun.load_state_dict(all_model['ImplicitFun_state_dict'])
    counter = counter + 1

# gpu or cpu 
if config.cuda:
    Encoder = Encoder.cuda()
    ImplicitFun = ImplicitFun.cuda()
    InverseImplicitFun = InverseImplicitFun.cuda()


# # optimizer
optimizer = optim.Adam(list(Encoder.parameters())+list(ImplicitFun.parameters())+list(InverseImplicitFun.parameters()), lr=config.learning_rate, betas=[config.beta1, 0.999])


# ----------------------------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    start_time = time.time()
    show_loss = 0
    num_iter = 0
    for epoch in range(counter, config.epoch):

        for it, data in enumerate(train_loader):
            points, values, shape = data
            if config.cuda:
                points = Variable(points).cuda()
                values = Variable(values).cuda()
                shape = Variable(shape).cuda()

            optimizer.zero_grad()
            latent = Encoder(shape)
            _,esti_values = ImplicitFun(latent, points)
            loss_occ = occupancy_loss(esti_values, values)

            branch_values,_ = ImplicitFun(latent, shape)
            esti_shape = InverseImplicitFun(latent, branch_values)
            loss_sr = selfrec_loss(esti_shape, shape)                # self-reconstruction loss

            loss = loss_occ + loss_sr
            loss.backward()
            optimizer.step()

            show_loss = show_loss + loss.item()
            num_iter = num_iter + 1
            writer.add_scalar('loss/loss', loss, num_iter)
            writer.add_scalar('loss/loss_occupancy', loss_occ, num_iter)
            writer.add_scalar('loss/loss_self_reconstruction', loss_sr, num_iter)
            if (epoch%10)==0:
                writer.add_histogram('embedding/z', latent, num_iter)

            if (it % 10) == 0:
                print("Epoch: [%4d/%5d] Time: %4.1f, Loss: %.4f, loss_occupancy: %.4f, loss_self_reconstruction: %.4f"
                    %(epoch, it, time.time()-start_time, show_loss/num_iter, loss_occ.item(), loss_sr.item()))

        if (epoch+1)%10==0:
            checkpoint(config, epoch, 'stage2', Encoder, ImplicitFun, InverseImplicitFun)