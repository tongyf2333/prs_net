import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch3d.transforms import quaternion_multiply,quaternion_invert
from mesh2voxel import create_voxel_grid
from merge_mesh import merge_mesh
import trimesh

def chamfer_distance_manual(pc1, pc2):
    B, N, _ = pc1.shape
    M = pc2.shape[1]
    pc1_expand = pc1.unsqueeze(2)  
    pc2_expand = pc2.unsqueeze(1) 
    dist = torch.norm(pc1_expand - pc2_expand, dim=3)
    min_dist1, _ = torch.min(dist, dim=2)
    min_dist2, _ = torch.min(dist, dim=1)
    cd = min_dist1.mean(dim=1) + min_dist2.mean(dim=1)
    return cd.mean()

def get_plane_sympoints(points,plane):
    #points [b,n,3]
    #plane [b,4]
    #output: sympoints [b,n,3]
    b, n, _ = points.shape
    normals = plane[:, :3]  # (b, 3)
    d = plane[:, 3].unsqueeze(1)  # (b,1)
    normals_norm = torch.norm(normals, dim=1, keepdim=True)  # (b,1)
    unit_normals = normals / normals_norm  # (b,3)
    d_p = torch.bmm(points, unit_normals.unsqueeze(2)).squeeze(2) + d  # (b,n)
    reflected = points - 2 * d_p.unsqueeze(2) * unit_normals.unsqueeze(1)  # (b,n,3)
    return reflected

def get_rot_sympoints(points,rot):
    #points [b,n,3]
    #rot [b,4]
    #output: sympoints [b,n,3]
    batch_size,num_points,_=points.shape
    tmp=torch.zeros((batch_size,num_points,4),device=points.device)
    tmp[:,:,1:4]=points
    res=quaternion_multiply(quaternion_multiply(rot.unsqueeze(1),tmp),quaternion_invert(rot.unsqueeze(1)))
    return res[:,:,1:4]

def get_sym(points,plane,rot):
    #points [b,n,3]
    #plane [b,4]
    #rot [b,4]
    #output: sympoints [b,n,3]
    sym_points_plane=get_plane_sympoints(points,plane)
    sym_points_rot=get_rot_sympoints(points,rot)
    return sym_points_plane,sym_points_rot

def GetNearest(points,near_points,min_bound,voxel_size):
    #points [b,n,3]
    #near_points[b,32,32,32,3]
    #min_bound [b,3]
    #voxel_size [b]
    #output: nearest_points [b,n,3]
    b, n, _ = points.shape
    idx = torch.floor((points - min_bound.unsqueeze(1)) / voxel_size.view(b, 1, 1)).long()  # [b,n,3]
    idx = torch.clamp(idx, 0, 31) 
    near_points_flat = near_points.view(b, -1, 3)  # [b, 32*32*32, 3]
    linear_idx = idx[..., 0] * 32 * 32 + idx[..., 1] * 32 + idx[..., 2]  # [b,n]
    linear_idx = linear_idx.unsqueeze(-1).expand(-1, -1, 3)  # [b, n, 3]
    nearest_points = torch.gather(near_points_flat, dim=1, index=linear_idx)  # [b, n, 3]
    return nearest_points

def SymmetricLoss(points,near_points,min_bound,voxel_size):
    #points [b,n,3]
    #near_points[b,32,32,32,3]
    #min_bound [b,3]
    #voxel_size [b]
    #output: loss
    nearest_points = GetNearest(points, near_points, min_bound, voxel_size)
    dis=points-nearest_points #[b,n,3]
    loss=torch.mean(torch.sum(torch.norm(dis,dim=2),dim=1))
    return loss       

if __name__ == "__main__":
    # Example usage
    points = torch.randn(10, 1000, 3)  # [b, n, 3]
    plane = torch.randn(10, 4)         # [b, 4]
    rot = torch.randn(10, 4)           # [b, 4]

    sym_points_plane, sym_points_rot = get_sym(points, plane, rot)
    print("Symmetric Points (Plane):", sym_points_plane)
    print("Symmetric Points (Rotation):", sym_points_rot)
    # Example for SymmetricLoss
    near_points = torch.randn(10, 32, 32, 32, 3)  # [b, 32, 32, 32, 3]
    min_bound = torch.randn(10, 3)                # [b, 3]
    voxel_size = torch.randn(10)                  # [b]
    loss = SymmetricLoss(points, near_points, min_bound, voxel_size)
    print("Symmetric Loss:", loss.item())
