

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
parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='Directory name to save the checkpoints')
parser.add_argument('--pretrain_model', type=bool, default=False, help='load pretrained model')
parser.add_argument('--pretrain_model_name', type=str, default='Corr-249.pth', help='pretrained model')
parser.add_argument('--cuda', type=bool, default=True, help='True for GPU, False for CPU')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--logs_dir', type=str, default='./logs', help='Root directory of samples')
parser.add_argument('--cd_weight', type=float, default=10, help='weight1')
parser.add_argument('--emd_weight', type=float, default=1, help='weight2')
parser.add_argument('--normal_weight', type=float, default=0.01, help='weight3')
parser.add_argument('--smooth_weight', type=float, default=0.1, help='weight4')
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

sig = str(datetime.datetime.now()) + config.signature
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
            points, values, shapes = data
            if config.cuda:
                points = Variable(points).cuda()
                values = Variable(values).cuda()
                shapes = Variable(shapes).cuda()
                coord = shapes.clone().detach().requires_grad_(True)  # for calculating surface normals


            optimizer.zero_grad()
            latent = Encoder(shapes)
            _,esti_values = ImplicitFun(latent, points)
            loss_occ = occupancy_loss(esti_values, values)               # occupancy loss

            branch_values,_ = ImplicitFun(latent, shapes)
            self_rec_shape = InverseImplicitFun(latent, branch_values)
            loss_sr = selfrec_loss(self_rec_shape, shapes)                # self-reconstruction loss

            cross_rec_shapes = InverseImplicitFun(latent, torch.cat((branch_values[1:,:,:], branch_values[:1,:,:]), dim=0))
            # surface normals
            _,input_ = ImplicitFun(latent, coord)
            input_normals = gradient(input_, coord)
            input_normals = F.normalize(input_normals, p=2, dim=2).detach()
            _,cr_ = ImplicitFun(latent, cross_rec_shapes)
            cr_normals = gradient(cr_, cross_rec_shapes)
            cr_normals = F.normalize(cr_normals, p=2, dim=2).detach()

            loss_cd, loss_normal = CD_normal_loss(cross_rec_shapes, shapes, cr_normals, input_normals)
            loss_emd = EMD_loss(cross_rec_shapes, shapes)
            loss_smooth = smooth_loss(cross_rec_shapes, shapes)
            loss_cr = config.cd_weight*loss_cd + config.emd_weight*loss_emd + config.normal_weight*loss_normal + config.smooth_weight*loss_smooth

            loss = loss_occ + loss_sr + loss_cr
            loss.backward()
            optimizer.step()

            show_loss = show_loss + loss.item()
            num_iter = num_iter + 1

            writer.add_scalar('loss/loss', loss, num_iter)
            writer.add_scalar('loss/loss_occupancy', loss_occ, num_iter)
            writer.add_scalar('loss/loss_self_reconstruction', loss_sr, num_iter) 
            writer.add_scalar('loss/loss_cross_reconstruction', loss_cr, num_iter)
            writer.add_scalar('loss/loss_normal', loss_normal, num_iter)
            writer.add_scalar('loss/loss_smooth', loss_smooth, num_iter)  
            writer.add_scalar('loss/loss_cd', loss_cd, num_iter)
            writer.add_scalar('loss/loss_emd', loss_emd, num_iter)
            
            if (it%10)==0:
                writer.add_histogram('embedding/latent', latent, num_iter)

            if (it % 20) == 0:
                print("Epoch: [%4d/%5d] Time: %4.1f, Loss: %.4f, loss_occupancy: %.4f, loss_self_reconstruction: %.4f, loss_normal: %.4f, loss_smooth: %.4f, loss_cd: %.4f, loss_emd: %.4f"
                    %(epoch, it, time.time()-start_time, show_loss/num_iter, loss_occ.item(), loss_sr.item(), loss_normal.item(), loss_smooth.item(), loss_cd.item(), loss_emd.item()))

            if (it+1)%100==0:
                checkpoint(config, epoch, 'stage3', Encoder, ImplicitFun, InverseImplicitFun)
