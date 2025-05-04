import torch
from dataset import MyDataset
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
parser.add_argument('--test_dir',type=str,default='preprocessed/test',help='path to test data')
parser.add_argument('--model_dir',type=str,default='models/model_weights_final.pth',help='path to load model')
parser.add_argument('--batch_size',type=int,default=1,help='batch size')
parser.add_argument('--weight',type=float,default=25,help='weight for regularization loss')
parser.add_argument('--hasaxis',type=str2bool,default=False,help='whether to use axis loss')
parser.add_argument('--use_chamfer',type=str2bool,default=False,help='whether to use chamfer distance')
args=parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dataset
print("test_dir:",args.test_dir)
print("model_dir:",args.model_dir)
print("batch_size:",args.batch_size)
print("weight:",args.weight)
print("hasaxis:",args.hasaxis)
print("use_chamfer:",args.use_chamfer)

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

dataloader_test=DataLoader(
    MyDataset(args.test_dir),
    batch_size=args.batch_size,
    shuffle=True
)

#network
net=Network(device).to(device)
loss=AllLoss(device,weight=args.weight,hasaxis=args.hasaxis,use_chamfer=args.use_chamfer).to(device)
net.load_state_dict(torch.load(args.model_dir))
net.eval()


for nearest,voxel,sampled in dataloader_test:
    nearest=nearest.to(device)
    voxel=voxel.to(device)
    sampled=sampled.to(device)
    plane_x,plane_y,plane_z,rot_x,rot_y,rot_z=net(voxel)
    cur_loss=loss(sampled,nearest,plane_x,plane_y,plane_z,rot_x,rot_y,rot_z)
    print("loss:",cur_loss.item())
    print("plane_x:",plane_x)
    print("plane_y:",plane_y)
    print("plane_z:",plane_z)
    if args.hasaxis:
        print("rot_x:",rot_x)
        print("rot_y:",rot_y)
        print("rot_z:",rot_z)
    