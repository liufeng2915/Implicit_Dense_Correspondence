
from torch.utils.data import DataLoader
import scipy.io
import numpy as np
import glob

class ShapeNet(object):
    def __init__(self, data_list, data_reso):
        self.data_list = data_list
        self.data_reso = data_reso    

    def load_data(self, path):

        mat_file = scipy.io.loadmat(path)
        data = mat_file['data_'+str(self.data_reso)]
        points = data[:,:3]
        values = data[:,3:4]
        shape = mat_file['shape']

        # data aug.
        scale_rand = np.random.rand(1,3)+0.5
        diameter = np.sqrt(np.sum((np.max(shape,axis=0)-np.min(shape,axis=0))**2))
        scale_shape = shape*scale_rand
        scale_diameter = np.sqrt(np.sum((np.max(scale_shape,axis=0)-np.min(scale_shape,axis=0))**2))
        scale_points = points*scale_rand
        scale_shape = scale_shape/scale_diameter*diameter
        scale_points = scale_points/scale_diameter*diameter

        return scale_points.astype(np.float32), values.astype(np.float32), scale_shape.astype(np.float32)

    def __getitem__(self, index):

        file_path = self.data_list[index]
        points, values, shape = self.load_data(file_path)

        return points, values, shape

    def __len__(self):
        return len(self.data_list)


def generate_list(data_path):

    data_list = sorted(glob.glob(path+'*.mat'))
    return data_list