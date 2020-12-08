
import torch
import torch.nn.functional as F

from metrics.ChamferDistance import*
from metrics.emd_module  import*
nnd_dist = ChamferDistance()
emd_dist = emdModule()
criterion = torch.nn.MSELoss()

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def cosine_loss(N1, N2):
    loss = N1*N2
    loss = loss.sum(-1)
    loss = torch.abs(loss)
    loss = 1-loss
    loss = torch.mean(1-F.cosine_similarity(N1, N2))
    return torch.mean(loss)

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):

    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    '''
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    '''
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points



def farthest_point_sample(xyz, npoint):
    '''
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    '''
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    '''
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    '''
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz):
    '''
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    '''
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)  #B*N*3, M*1
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    #grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    #torch.cuda.empty_cache()

    return new_xyz, fps_idx, grouped_xyz, idx

#------------------------------------------------------------------------------------------------------------#

def occupancy_loss(esti_values, values):
    return criterion(esti_values, values)

def selfrec_loss(esti_shapes, shapes):
    return criterion(esti_shapes, shapes)

def CD_normal_loss(esti_shapes, shapes, esti_normals, normals):
    dist1, dist2, idx1, idx2 = nnd_dist(esti_shapes, shapes)
    loss_cd = torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))

    corr_normals = torch.gather(normals, 1, idx1.long().unsqueeze(-1).repeat(1,1,3))
    loss_normal = cosine_loss(esti_normals, corr_normal)
    return loss_cd, loss_normal

def EMD_loss(esti_shapes, shapes):
    dist, assigment = emd_dist(esti_shapes, shapes, 0.005, 50)
    loss_emd = torch.sqrt(dist).mean(1).mean()
    return loss_emd

npatch = 512
radius = 0.1
nsample =  16
def smooth_loss(esti_shapes, shapes):
    _,_,_,idx = sample_and_group(npatch, radius, nsample, torch.cat((shapes[1:,:,:], shapes[:1,:,:])))
    offset_vectors = esti_shapes - torch.cat((shapes[1:,:,:], shapes[:1,:,:])).detach()
    patch_offset_vectors = index_points(offset_vectors.unsqueeze(-1), idx).squeeze(-1)
    loss_smooth = torch.mean(1-F.cosine_similarity(patch_offset_vectors[:,:,:-1,:], patch_offset_vectors[:,:,1:,:], 3))
    return loss_smooth
