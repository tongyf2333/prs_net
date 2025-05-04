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
parser.add_argument('--train_dir',type=str,default='preprocessed/chair',help='path to training data')
parser.add_argument('--model_dir',type=str,default='models',help='path to save model')
parser.add_argument('--epoch',type=int,default=300,help='number of epochs')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--weight',type=float,default=25,help='weight for regularization loss')
parser.add_argument('--hasaxis',type=str2bool,default=False,help='whether to use axis loss')
parser.add_argument('--use_chamfer',type=str2bool,default=False,help='whether to use chamfer distance')
parser.add_argument('--lr',type=float,default=0.01,help='learning rate')
args=parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)
torch.cuda.manual_seed(1)

#dataset
print("train_dir:",args.train_dir)
print("model_dir:",args.model_dir)
print("epoch:",args.epoch)
print("batch_size:",args.batch_size)
print("weight:",args.weight)
print("hasaxis:",args.hasaxis)
print("use_chamfer:",args.use_chamfer)
print("learning rate:",args.lr)

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

dataloader_train=DataLoader(
    MyDataset(args.train_dir),
    batch_size=args.batch_size,
    shuffle=True
)

#network
net=Network(device).to(device)
loss=AllLoss(device,weight=args.weight,hasaxis=args.hasaxis,use_chamfer=args.use_chamfer).to(device)
#optimizer
optimizer= optim.Adam(
    net.parameters(),
    lr=args.lr,
    betas=(0.9, 0.999),
)

net.train()
start_time=time.time()

for i in range(args.epoch):
    for nearest,voxel,sampled in dataloader_train:
        nearest=nearest.to(device)
        voxel=voxel.to(device)
        sampled=sampled.to(device)
        optimizer.zero_grad()
        plane_x,plane_y,plane_z,rot_x,rot_y,rot_z=net(voxel)
        cur_loss=loss(sampled,nearest,plane_x,plane_y,plane_z,rot_x,rot_y,rot_z)
        cur_loss.backward()
        optimizer.step()
        if i%10==0:
            print("Epoch:", i, "Loss:", cur_loss.item())
    if i%20==0:
        torch.save(net.state_dict(), os.path.join(args.model_dir, f'model_weights_epoch_{i}.pth'))
        print("Model saved at epoch:", i)

print("Training time:", time.time()-start_time)

torch.save(net.state_dict(), os.path.join(args.model_dir, 'model_weights_final.pth'))