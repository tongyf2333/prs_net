import torch

from regloss import RegularizationLoss
from symloss import SymmetricLoss,get_sym,chamfer_distance_manual

class AllLoss(torch.nn.Module):
    def __init__(self, device, weight,hasaxis=True, use_chamfer=False):
        super(AllLoss, self).__init__()
        self.device = device
        self.hasaxis = hasaxis
        self.weight= weight
        self.use_chamfer = use_chamfer
        self.reg_loss_fn = RegularizationLoss(device,hasaxis)
        self.sym_loss_fn = SymmetricLoss

    def forward(self, points,near_point,plane_x, plane_y, plane_z, rot_x, rot_y, rot_z):
        # points: [b,n,3]
        # voxel_grid: [b,c,h,w,d]
        # voxel_size: [b]
        # min_bound: [b,3]
        # near_point: [b,n,3]        
        sym_points_plane_x, sym_points_rot_x = get_sym(points, plane_x, rot_x)
        sym_points_plane_y, sym_points_rot_y = get_sym(points, plane_y, rot_y)
        sym_points_plane_z, sym_points_rot_z = get_sym(points, plane_z, rot_z)

        loss = torch.tensor([0.0],device=self.device,dtype=torch.float32).to(self.device)
        loss += self.weight * self.reg_loss_fn(plane_x[..., 0:3], plane_y[..., 0:3], plane_z[..., 0:3], rot_x, rot_y, rot_z)
        
        if self.use_chamfer:
            loss_plane_x = chamfer_distance_manual(sym_points_plane_x, points)
            loss_plane_y = chamfer_distance_manual(sym_points_plane_y, points)
            loss_plane_z = chamfer_distance_manual(sym_points_plane_z, points)
            loss += loss_plane_x + loss_plane_y + loss_plane_z
        else:
            loss_plane_x = self.sym_loss_fn(sym_points_plane_x, near_point)
            loss_plane_y = self.sym_loss_fn(sym_points_plane_y, near_point)
            loss_plane_z = self.sym_loss_fn(sym_points_plane_z, near_point)
            loss += loss_plane_x + loss_plane_y + loss_plane_z
        
        if self.hasaxis:
            if self.use_chamfer:
                loss_rot_x= chamfer_distance_manual(sym_points_rot_x, points)
                loss_rot_y= chamfer_distance_manual(sym_points_rot_y, points)
                loss_rot_z= chamfer_distance_manual(sym_points_rot_z, points)
            else:
                loss_rot_x = self.sym_loss_fn(sym_points_rot_x, near_point)
                loss_rot_y = self.sym_loss_fn(sym_points_rot_y, near_point)
                loss_rot_z = self.sym_loss_fn(sym_points_rot_z, near_point)
            loss += loss_rot_x + loss_rot_y + loss_rot_z
        return loss

# Example usage
if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = torch.randn(2, 1024, 3).to(device)
    voxel = torch.randn(2, 3, 32, 32, 32).to(device)
    near_point = torch.randn(2, 32, 32, 32, 3).to(device)
    plane_x = torch.randn(2, 4).to(device)
    plane_y = torch.randn(2, 4).to(device)
    plane_z = torch.randn(2, 4).to(device)
    rot_x = torch.randn(2, 4).to(device)
    rot_y = torch.randn(2, 4).to(device)
    rot_z = torch.randn(2, 4).to(device)

    all_loss_fn = AllLoss(device, weight=1.0).to(device)
    loss = all_loss_fn(points, voxel, near_point, plane_x, plane_y, plane_z, rot_x, rot_y, rot_z)
    print("Loss:", loss.item())