import torch
from preprocess import MyDataset
from network import Network
from allloss import AllLoss
from torch.utils.data import DataLoader
import torch.optim as optim
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#dataset
dataloader_test=DataLoader(
    MyDataset('meshes/test',rotate=False),
    batch_size=1,
    shuffle=True
)

#settings
hasaxis=False
use_chamfer=False
weight=25

#network
net=Network(device,verbose=True).to(device)
loss=AllLoss(device,hasaxis=hasaxis,weight=weight,use_chamfer=use_chamfer).to(device)
net.load_state_dict(torch.load("models/model_weights_final.pth"))# best model 200 iter

net.eval()

for points, voxel_grid, voxel_size, min_bound,near_point in dataloader_test:
    points=points.to(device)
    voxel_grid=voxel_grid.to(device)
    voxel_size=voxel_size.to(device)
    min_bound=min_bound.to(device)
    near_point=near_point.to(device)
    plane_x,plane_y,plane_z,rot_x,rot_y,rot_z=net(voxel_grid)
    plane_x,plane_y,plane_z,rot_x,rot_y,rot_z=net(voxel_grid)
    cur_loss=loss(points,voxel_size,min_bound,near_point,plane_x,plane_y,plane_z,rot_x,rot_y,rot_z)
    print("Total Test Loss:", cur_loss.item())
    print("Plane_x:",plane_x)
    print("Plane_y:",plane_y)
    print("Plane_z:",plane_z)