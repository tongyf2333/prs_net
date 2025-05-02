import torch
from preprocess import MyDataset
from network import Network
from allloss import AllLoss
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import argparse
from regloss import quaternion_to_axis_angle_batch
from argparse import ArgumentParser
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser=ArgumentParser()
parser.add_argument('--train_dir',type=str,default='meshes/bottle',help='path to training data')
parser.add_argument('--test_dir',type=str,default='meshes/bottle',help='path to testing data')
parser.add_argument('--model_dir',type=str,default='models',help='path to save model')
parser.add_argument('--epoch',type=int,default=300,help='number of epochs')
parser.add_argument('--batch_size',type=int,default=1,help='batch size')
parser.add_argument('--weight',type=float,default=25,help='weight for regularization loss')
parser.add_argument('--hasaxis',type=str2bool,default=True,help='whether to use axis loss')
parser.add_argument('--use_chamfer',type=str2bool,default=False,help='whether to use chamfer distance')
parser.add_argument('--rotate_train',type=str2bool,default=False,help='whether to rotate the training mesh')
parser.add_argument('--rotate_test',type=str2bool,default=False,help='whether to rotate the testing mesh')
args=parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed(1)

#dataset
print("train_dir:",args.train_dir)
print("test_dir:",args.test_dir)
print("model_dir:",args.model_dir)
print("epoch:",args.epoch)
print("batch_size:",args.batch_size)
print("weight:",args.weight)
print("hasaxis:",args.hasaxis)
print("use_chamfer:",args.use_chamfer)
print("rotate_train:",args.rotate_train)
print("rotate_test:",args.rotate_test)

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

dataloader_train=DataLoader(
    MyDataset(args.train_dir,rotate=args.rotate_train),
    batch_size=args.batch_size,
    shuffle=True
)
dataloader_test=DataLoader(
    MyDataset(args.test_dir,rotate=args.rotate_test),
    batch_size=1,
    shuffle=True
)

#network
net=Network(device).to(device)
loss=AllLoss(device,weight=args.weight,hasaxis=args.hasaxis,use_chamfer=args.use_chamfer).to(device)
#optimizer
optimizer= optim.Adam(
    net.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
)

net.train()
start_time=time.time()

for i in range(args.epoch):
    for points, voxel_grid, voxel_size, min_bound,near_point in dataloader_train:
        points=points.to(device)
        voxel_grid=voxel_grid.to(device)
        voxel_size=voxel_size.to(device)
        min_bound=min_bound.to(device)
        near_point=near_point.to(device)
        optimizer.zero_grad()
        plane_x,plane_y,plane_z,rot_x,rot_y,rot_z=net(voxel_grid)
        cur_loss=loss(points,voxel_size,min_bound,near_point,plane_x,plane_y,plane_z,rot_x,rot_y,rot_z)
        cur_loss.backward()
        optimizer.step()
        if i%10==0:
            print("Epoch:", i, "Loss:", cur_loss.item())
    if i%20==0:
        torch.save(net.state_dict(), os.path.join(args.model_dir, f'model_weights_epoch_{i}.pth'))
        print("Model saved at epoch:", i)

print("Training time:", time.time()-start_time)

torch.save(net.state_dict(), os.path.join(args.model_dir, 'model_weights_final.pth'))
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
    if args.hasaxis:
        axis_x,_=quaternion_to_axis_angle_batch(rot_x)
        axis_y,_=quaternion_to_axis_angle_batch(rot_y)
        axis_z,_=quaternion_to_axis_angle_batch(rot_z)
        print("Axis_x:",axis_x)
        print("Axis_y:",axis_y)
        print("Axis_z:",axis_z)