import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from knn_cuda import KNN
from einops import repeat

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint,random=True):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    prev_dist = torch.ones(B, N).to(device) * 1e10
    if random: 
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    else:
        farthest = torch.tensor([0]).to(device)
    
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        current_dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = current_dist < prev_dist
        prev_dist[mask] = current_dist[mask]
        farthest = torch.max(prev_dist, -1)[1]
    return centroids
#point= fps(point, npoint)


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
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

def normalize_patch(grouped_xyz, trans_norm=True, scale_norm=True, eps=1e-12):
    '''
    Input:
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, local region points XYZs
    Output:
        normed_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized local region points XYZs to zero mean and unit norm
        patch_mean: (batch_size, npoint, 1, 3) TF tensor, local region mean
        patch_norm: (batch_size, npoint, 1, 1) TF tensor, local region norm
    '''
    patch_mean = torch.mean(grouped_xyz, dim=2, keepdim=True)  # batch_size x npoint x 1 x 3
    if trans_norm:
        shifted_xyz = grouped_xyz - patch_mean
    else:
        shifted_xyz = grouped_xyz

    point_norm = torch.norm(shifted_xyz, dim =3, keepdim=True)  # batch_size x npoint x nsample x 1
    patch_norm = torch.max(point_norm, axis=2, keepdim=True)  # B x npoint x 1 x 1
    #patch_norm = torch.max(patch_norm, eps)  # for safe division

    if scale_norm:
        normed_xyz = shifted_xyz / patch_norm
    else:
        normed_xyz = shifted_xyz

    return normed_xyz, patch_mean, patch_norm

def denormalize_patch(normed_xyz, patch_mean, patch_norm, trans_norm=True, scale_norm=True):
    '''
    Input:
        normed_xyz: (batch_size, npoint, nsample, 3) TF tensor, local region points XYZs
        patch_mean: (batch_size, npoint, 1, 3) TF tensor, local region target mean
        patch_norm: (batch_size, npoint, 1, 1) TF tensor, local region target norm
    Output:
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, denormalized local region points XYZs
    '''

    if scale_norm:
        shifted_xyz = normed_xyz * patch_norm
    else:
        shifted_xyz = normed_xyz

    if trans_norm:
        grouped_xyz = shifted_xyz + patch_mean
    else:
        grouped_xyz = shifted_xyz

    return grouped_xyz


def sample_and_group(num_in_point, npoint, radius, nsample, xyz, points, bacth_size,knn= False, trans_norm=True, scale_norm=True, returnfps=False,use_xyz=True,global_fetuers=False ,seed_choice = 'FPS'):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    
    if seed_choice == 'FPS':
        fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    
    if seed_choice == "Random":
        fps_idx = np.random.randint( num_in_point-1, size=(bacth_size,npoint))


    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    
    torch.cuda.empty_cache()

    if knn:
        _, idx = KNN(nsample, transpose_mode=False)(xyz.permute(0,2,1).contiguous(), new_xyz.permute(0,2,1).contiguous())
    else: 
        idx = query_ball_point(radius, nsample, xyz, new_xyz)

    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()

    ##should add all 3 feature of grouped_xyz 
    # grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_xyz_norm, patch_mean, patch_norm = normalize_patch(grouped_xyz, trans_norm, scale_norm)
    
    
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        if use_xyz:
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz_norm
    
    if global_fetuers:
        contcat_seeds = repeat(new_xyz, 'B N C -> B k N C', k=nsample)
        new_points = torch.cat([grouped_xyz_norm, contcat_seeds], dim=-1)

    return new_xyz, new_points, idx, grouped_xyz_norm, grouped_xyz, patch_mean, patch_norm









def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, num_in_point, npoint, radius, nsample, in_channel1 ,in_channel2, mlp,mlp2, group_all,knn,trans_norm , scale_norm , use_xyz,use_nchw,global_fetuers,seed_choice,batch_size):
        super(PointNetSetAbstraction, self).__init__()
        self.num_in_point = num_in_point
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.global_fetuers = global_fetuers
        self.mlp_convs2 = nn.ModuleList()
        self.mlp_bns2 = nn.ModuleList()
        self.batch_size= batch_size
        self.knn = knn
        self.trans_norm = trans_norm
        self.scale_norm = scale_norm
        self.mlp2 = mlp2
        self.use_xyz = use_xyz
        self.use_nchw = use_nchw
        self.seed_choice = seed_choice
        last_channel = in_channel1
        last_channel2 = in_channel2
        
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        
        for out_channel2 in mlp2:
            self.mlp_convs2.append(nn.Conv2d(last_channel2, out_channel2, 1))
            self.mlp_bns2.append(nn.BatchNorm2d(out_channel2))
            last_channel2 = out_channel2

        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)#should update
        else:
            new_xyz, new_points, idx, grouped_xyz, grouped_xyz_orig, patch_mean, patch_norm  = sample_and_group(self.num_in_point ,self.npoint, self.radius, self.nsample, xyz, points,self.batch_size, self.knn ,self.trans_norm, self.scale_norm, self.use_xyz ,global_fetuers =self.global_fetuers, seed_choice =self.seed_choice)


        # new_xyz: sampled points position data, [B, npoint, C]g
        # new_points: samplif self.use_nchw :

        new_points = new_points.permute(0, 3, 2, 1)#new 
            #new_points = new_points.permute(0, 3, 1, 2)# [B, C+D, nsample,npoint] 
     
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            # new_points =  F.relu(bn(conv(new_points)))
            new_points= F.relu(bn(conv(new_points)))
       
        # if self.use_nchw :
        new_points = new_points.permute(0, 2, 3, 1)
       
        new_points,_ = torch.max(new_points, dim= 2, keepdim= True)
        # print(new_points.shape)
       

        if self.mlp2 is not None: 
            new_points = new_points.permute(0, 3, 1, 2)
            for i, conv in enumerate(self.mlp_convs2):
                bn2 = self.mlp_bns2[i]
                if (i!=(len(self.mlp_convs2)-1)):
                    new_points =  F.relu(bn2(conv(new_points)))
                else: 
                    new_points = conv(new_points)
        
       
        new_points = new_points.permute(0, 2, 3, 1)
        new_points = torch.max(new_points, 2)[0]





        #new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points, idx, grouped_xyz, grouped_xyz_orig, patch_mean, patch_norm

        
        # new_points,_ = torch.max(new_points, dim= 2, keepdim= True)#
        # if self.mlp2 is not None: 
        #     for i, conv in enumerate(self.mlp_convs2):
        #         bn2 = self.mlp_bns2[i]
        #         #At TF the relu is only for the last layer? 
        #         new_points =  F.relu(bn2(conv(new_points)))
        
        # new_xyz = new_xyz.permute(0, 2, 1)
        

        # #they squze the new_points at TF
        
        # return new_xyz, new_points, idx, grouped_xyz, grouped_xyz_orig, patch_mean, patch_norm


        



class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

