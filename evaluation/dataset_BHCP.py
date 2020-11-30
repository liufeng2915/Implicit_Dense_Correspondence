
from torch.utils.data import DataLoader
import scipy.io
import numpy as np
import glob

class BHCP(object):
    def __init__(self, data_list):
        self.data_list = data_list

    def load_data(self, path1, path2):

        mat_file1 = scipy.io.loadmat(path1)
        shape1 = mat_file1['points'].astype(np.float32)
        land1 = mat_file1['land'].astype(np.float32)

        mat_file2 = scipy.io.loadmat(path2)
        shape2 = mat_file2['points'].astype(np.float32)
        land2 = mat_file2['land'].astype(np.float32)

        shape = np.concatenate((np.expand_dims(shape1, axis=0), np.expand_dims(shape2, axis=0)), axis=0)

        return shape, land1, land2

    def __getitem__(self, index):

        file_path = self.data_list[index]
        name1 = file_path.split(' ')[0]
        name2 = file_path.split(' ')[1]
        shape, land1, land2 = self.load_data(name1, name2)

        # #
        valid_idx1 = np.where(land1[:, 0]!=-1)[0]
        valid_idx2 = np.where(land2[:, 0]!=-1)[0]
        inter_idx = np.intersect1d(valid_idx1, valid_idx2)
        valid_land1 = land1[inter_idx, 4:7]
        valid_land2 = land2[inter_idx, 4:7]

        return shape, valid_land1, valid_land2, name1.split('/')[-1][:-4], name2.split('/')[-1][:-4]

    def __len__(self):
        return len(self.data_list)

##
def get_pairwise_list(path):

    temp_list = sorted(glob.glob(path+'*.mat'))
    data_list = []
    for i in range(len(temp_list)):
        for j in range(i+1, len(temp_list)):
            name = temp_list[i]+ ' ' + temp_list[j]
            data_list.append(name)

    return data_list