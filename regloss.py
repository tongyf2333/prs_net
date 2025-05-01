import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def l2_norm(x):
    return F.normalize(x, p=2, dim=1)

def quaternion_to_axis_angle_batch(q_batch):    
    w=q_batch[:,0]
    x=q_batch[:,1]
    y=q_batch[:,2]
    z=q_batch[:,3]
    theta_batch=2*torch.acos(w)
    sin_half_theta_batch=torch.sqrt(1-w**2)
    axis_batch=torch.zeros_like(q_batch[:,1:])
    mask=sin_half_theta_batch>1e-6
    axis_batch[mask]=torch.stack([x[mask],y[mask],z[mask]],dim=1)/sin_half_theta_batch[mask].unsqueeze(1)
    return axis_batch,theta_batch

class RegularizationLoss(nn.Module):
    def __init__(self, device,weight=0.01,hasaxis=True):
        super(RegularizationLoss, self).__init__()
        self.weight = weight
        self.device = device
        self.hasaxis=hasaxis

    def forward(self, plane_x, plane_y, plane_z, rot_x, rot_y, rot_z):
        #input: plane_x [b,3], plane_y [b,3], plane_z [b,3], rot_x [b,4], rot_y [b,4], rot_z [b,4]
        #output: loss
        batch_size=plane_x.shape[0]
        if self.hasaxis:
            axis_x,_=quaternion_to_axis_angle_batch(rot_x)
            axis_y,_=quaternion_to_axis_angle_batch(rot_y)
            axis_z,_=quaternion_to_axis_angle_batch(rot_z)
            M2=torch.stack([l2_norm(axis_x),l2_norm(axis_y),l2_norm(axis_z)],dim=1).reshape(batch_size,3,3)
            M2=torch.bmm(M2,M2.transpose(1,2))-torch.eye(3).unsqueeze(0).expand(batch_size,-1,-1).to(self.device)
        M1=torch.stack([l2_norm(plane_x),l2_norm(plane_y),l2_norm(plane_z)],dim=1).reshape(batch_size,3,3)
        M1=torch.bmm(M1,M1.transpose(1,2))-torch.eye(3).unsqueeze(0).expand(batch_size,-1,-1).to(self.device)
        if self.hasaxis:
            loss=torch.mean(torch.sum(torch.sum(M1**2+M2**2,dim=2),dim=1))
        else:
            loss=torch.mean(torch.sum(torch.sum(M1**2,dim=2),dim=1))
        return loss
    

if __name__ == "__main__":
    # Example usage
    plane_x = torch.randn(10, 3)
    plane_y = torch.randn(10, 3)
    plane_z = torch.randn(10, 3)
    rot_x = torch.randn(10, 4)
    rot_y = torch.randn(10, 4)
    rot_z = torch.randn(10, 4)

    loss_fn = RegularizationLoss()
    loss = loss_fn(plane_x, plane_y, plane_z, rot_x, rot_y, rot_z)
    print("Regularization Loss:", loss.item())