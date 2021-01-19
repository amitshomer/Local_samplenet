import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_util2 import PointNetSetAbstraction
from models.pointnet_util2 import denormalize_patch
from src.soft_projection_localnet import SoftProjection
import torch
##from src.soft_projection import SoftProjection
from src.chamfer_distance import ChamferDistance




class Local_samplenet(nn.Module):
    def __init__(self,num_class ,
        bottleneck_size,
        batch_size,
        num_in_point = 1024,  
        num_output_points=32 ,
        normal_channel=False,
        global_fetuers=False,
        npatch=4,
        nsample_per_patch=256,
        seed_choice= 'FPS',
        projections_group_size=7,
        trans_norm=True, 
        scale_norm=True,
        input_shape="bcn",
        output_shape="bcn",
        skip_projection=False):

        super(Local_samplenet,self).__init__()
        self.npatch = npatch
        self.npoint_per_patch = num_output_points/npatch
        in_channel = 6 if normal_channel or global_fetuers else 3
        self.normal_channel = normal_channel
        self.batch_size = batch_size
        self.nsample_per_patch = nsample_per_patch
        self.bottleneck_size =bottleneck_size
        
        self.sa1 = PointNetSetAbstraction( num_in_point = num_in_point, num_output_points=num_output_points, npoint=self.npatch, radius=0.2, nsample=self.nsample_per_patch, in_channel1=in_channel,in_channel2=self.bottleneck_size , mlp=[64, 64,64,128,self.bottleneck_size],mlp2=[256, 256, 256, 3*int(self.npoint_per_patch)], group_all=False, knn=True, trans_norm=trans_norm, scale_norm=scale_norm,use_xyz=False,use_nchw=True,global_fetuers=global_fetuers,seed_choice=seed_choice, batch_size=batch_size )
        # self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        ##### before remove- dnot delete yet, need for load old weights
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)
        #####

        self.project = SoftProjection(projections_group_size)
        self.trans_norm = trans_norm
        self.scale_norm =scale_norm
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.skip_projection = skip_projection

    def forward(self, xyz):
        end_points = {}
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points, l1_indices, l1_patches, l1_patches_orig, l1_patch_mean, l1_patch_norm, spoint_seeds = self.sa1(xyz, norm)
        
        #Itai part
        end_points['patches_orig'] = l1_patches_orig
        end_points['batch_patches_orig'] = torch.reshape(l1_patches_orig, [self.batch_size * self.npatch, -1, 3])
        end_points['patches_mean'] = l1_patch_mean
        end_points['patches_norm'] = l1_patch_norm

        end_points['patches_normed'] = l1_patches
        end_points['batch_patches_normed'] = torch.reshape(l1_patches, [self.batch_size * self.npatch, -1, 3])
        
        # sampled points from patches (num_output_points/npoint are sampled from each patch)
        net = torch.reshape(l1_points, [self.batch_size, self.npatch, -1, 3])
        
        end_points['patches_normed_simplified'] = net
        end_points['batch_patches_normed_simplified'] = torch.reshape(net, [self.batch_size * self.npatch, -1, 3])
        
        # denormalize patches (there are npoint patches for each point cloud)
        net = denormalize_patch(net, l1_patch_mean, l1_patch_norm, trans_norm=self.trans_norm, scale_norm=self.scale_norm)
        end_points['patches_denormed_simplified'] = net
        end_points['batch_patches_denormed_simplified'] = torch.reshape(net, [self.batch_size * self.npatch, -1, 3])

        # collect simplified points from patches (num_output_points from each point cloud)
        simplified_points = torch.reshape(net, [self.batch_size, -1, 3])
        end_points['simplified_points'] = simplified_points


        # project simplified points
        batch_patches_normed = end_points['batch_patches_normed']
        batch_patches_normed_simplified = end_points['batch_patches_normed_simplified']
        
        batch_patches_normed_projected, projection_weights, dist = self.project(batch_patches_normed, batch_patches_normed_simplified)
        patches_normed_projected = torch.reshape(batch_patches_normed_projected, [self.batch_size, self.npatch, -1, 3])
       
        end_points['patches_normed_projected'] = patches_normed_projected
        end_points['batch_patches_normed_projected'] = batch_patches_normed_projected
        
        # denormalize patches (there are npoint patches for each point cloud)
        patches_denormed_projected = denormalize_patch(patches_normed_projected, l1_patch_mean, l1_patch_norm, trans_norm=self.trans_norm, scale_norm=self.scale_norm)
        batch_patches_denormed_projected = torch.reshape(patches_denormed_projected, [self.batch_size * self.npatch, -1, 3])
        
        end_points['patches_denormed_projected'] = patches_denormed_projected
        end_points['batch_patches_denormed_projected'] = batch_patches_denormed_projected

        #collect projected points from patches (num_output_points from each point cloud)
        projected_points = torch.reshape(patches_denormed_projected, [self.batch_size, -1, 3])

        return projected_points, end_points,spoint_seeds


    def get_simplification_loss(self, ref_pc, samp_pc, pc_size, gamma=1, delta=0):
        # if self.skip_projection or not self.training:
        if not self.training:
            return torch.tensor(0).to(ref_pc)
        # ref_pc and samp_pc are B x N x 3 matrices
        if self.input_shape == "bcn":
            ref_pc = ref_pc.permute(0, 2, 1).contiguous()
            samp_pc = samp_pc.permute(0, 2, 1).contiguous()

        cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
        max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
        max_cost = torch.mean(max_cost)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        cost_p2_p1 = torch.mean(cost_p2_p1)
        loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1

        if self.output_shape == "bcn":
            ref_pc = ref_pc.permute(0, 2, 1).contiguous()
            samp_pc = samp_pc.permute(0, 2, 1).contiguous()

        return loss
   
    def get_sampleseed_loos(self, ref_pc, samp_pc, pc_size, gamma=1, delta=0):
        # if self.skip_projection or not self.training:
        if not self.training:
            return torch.tensor(0).to(ref_pc)
        # ref_pc and samp_pc are B x N x 3 matrices
        if self.input_shape == "bcn":
            ref_pc = ref_pc.permute(0, 2, 1).contiguous()
            samp_pc = samp_pc.permute(0, 2, 1).contiguous()

        cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
        max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
        max_cost = torch.mean(max_cost)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        cost_p2_p1 = torch.mean(cost_p2_p1)
        
        # loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
        loss = cost_p2_p1
        
        if self.output_shape == "bcn":
            ref_pc = ref_pc.permute(0, 2, 1).contiguous()
            samp_pc = samp_pc.permute(0, 2, 1).contiguous()

        return loss

    def get_projection_loss(self):
            sigma = self.project.sigma()
            if self.skip_projection or not self.training:
                return torch.tensor(0).to(sigma)
            return sigma
    
    def get_local_loss(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


# class get_local_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()

#     def forward(self, pred, target, trans_feat):
#         total_loss = F.nll_loss(pred, target)

#         return total_loss
