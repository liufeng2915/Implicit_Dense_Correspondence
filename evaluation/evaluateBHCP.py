

import os
import numpy as np 
import sys
sys.path.append('../')
from model import ImplicitFun, Encoder, InverseImplicitFun
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasetBHCP import*
import scipy.io
import scipy.spatial

## 
cate_name = 'helicopter' # chair, plane, bike helicopter

## data
data_dir = '../data/BHCP/' + cate_name + '/'
data_list = get_pairwise_list(data_dir)
eval_data = BHCP(data_list=data_list)
train_loader = DataLoader(eval_data, num_workers=2, batch_size=1, shuffle=False)

# network
ImplicitFun = ImplicitFun()  
Encoder = Encoder() 
InverseImplicitFun = InverseImplicitFun()
if cate_name == 'helicopter':
    all_model = torch.load('../models/plane.pth')
else:
    all_model = torch.load('../models/' + cate_name + '.pth')
ImplicitFun.load_state_dict(all_model['ImplicitFun_state_dict'])
Encoder.load_state_dict(all_model['Encoder_state_dict'])
InverseImplicitFun.load_state_dict(all_model['InverseImplicitFun_state_dict'])
print(InverseImplicitFun)

# gpu or cpu 
ImplicitFun = ImplicitFun.cuda()
Encoder = Encoder.cuda()
InverseImplicitFun = InverseImplicitFun.cuda()

# ----------------------------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    thres = np.arange(0,0.26,0.01)

    dis_list = np.array([])
    for it, data in enumerate(train_loader):

        print("Paired sample: [%d/%d]"%(it, len(train_loader.dataset)))
        shape, land_a, land_b, name_a, name_b = data
        shape = Variable(shape.squeeze(0).cuda())

        latent = Encoder(shape)
        branch_values, _ = ImplicitFun(latent, shape)
        cross_rec_shapes = InverseImplicitFun(latent, torch.cat((branch_values[1:,:,:], branch_values[:1,:,:]), dim=0))
        
        e_shape_a = cross_rec_shapes[0:1].squeeze(0).data.cpu().numpy()
        e_shape_b = cross_rec_shapes[1:2].squeeze(0).data.cpu().numpy()
        shape_a = shape[0:1].squeeze(0).data.cpu().numpy()
        shape_b = shape[1:2].squeeze(0).data.cpu().numpy()
        land_a = land_a.squeeze(0).data.cpu().numpy()
        land_b = land_b.squeeze(0).data.cpu().numpy()

        # # KNN
        nn_land_on_shape_b = shape_b[scipy.spatial.KDTree(e_shape_a).query(land_a)[1]]
        nn_land_on_shape_a = shape_a[scipy.spatial.KDTree(e_shape_b).query(land_b)[1]]
        dis_land_a = np.sqrt(np.sum((land_a - nn_land_on_shape_a)**2, axis=1))
        dis_land_b = np.sqrt(np.sum((land_b - nn_land_on_shape_b)**2, axis=1))

        # # Normlization
        diameter_shape_a = np.sqrt(np.sum((np.max(shape_a, axis=0)-np.min(shape_a, axis=0))**2))
        diameter_shape_b = np.sqrt(np.sum((np.max(shape_b, axis=0)-np.min(shape_b, axis=0))**2))
        dis_land_a = dis_land_a/diameter_shape_a
        dis_land_b = dis_land_b/diameter_shape_b

        dis_list = np.append(dis_list, dis_land_a, axis=0)
        dis_list = np.append(dis_list, dis_land_b, axis=0)

    # #
    dis = []
    for i in range(len(thres)):
        temp_num = np.where(dis_list<=thres[i])[0]
        dis.append(temp_num.shape[0]/dis_list.shape[0])

    scipy.io.savemat('results/ours_' + cate_name + '_aligned.mat', {'distance':dis, 'threshold':thres})
    

