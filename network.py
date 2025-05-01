import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(torch.nn.Module):
    def __init__(self,device,verbose=False):
        super(Network, self).__init__()
        self.verbose=verbose
        self.conv=nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(4),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.Conv3d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
        ).to(device)
        self.plane_predictors=nn.ModuleList()
        self.rot_predictors=nn.ModuleList()
        init_planes=[]
        init_planes.append(torch.tensor([1,0,0,0],dtype=torch.float32).to(device))
        init_planes.append(torch.tensor([0,1,0,0],dtype=torch.float32).to(device))
        init_planes.append(torch.tensor([0,0,1,0],dtype=torch.float32).to(device))
        init_rots=[]
        init_rots.append(torch.tensor([0,1,0,0],dtype=torch.float32).to(device))
        init_rots.append(torch.tensor([0,0,1,0],dtype=torch.float32).to(device))
        init_rots.append(torch.tensor([0,0,0,1],dtype=torch.float32).to(device))
        for i in range(3):
            self.plane_predictors.append(nn.Sequential(
                nn.Linear(64, 32),
                nn.LeakyReLU(negative_slope=0.2,inplace=True),
                nn.Linear(32, 16),
                nn.LeakyReLU(negative_slope=0.2,inplace=True),
                nn.Linear(16, 4)
            ).to(device))
            self.rot_predictors.append(nn.Sequential(
                nn.Linear(64, 32),
                nn.LeakyReLU(negative_slope=0.2,inplace=True),
                nn.Linear(32, 16),
                nn.LeakyReLU(negative_slope=0.2,inplace=True),
                nn.Linear(16, 4)
            ).to(device))
            self.plane_predictors[i][4].bias.data.copy_(init_planes[i])
            self.rot_predictors[i][4].bias.data.copy_(init_rots[i])
        def normalizer(x):
            return x/(1e-12+torch.norm(x[:,:3],dim=1,keepdim=True))
        self.normalizer=normalizer

    def forward(self, input):
        #predicting latent code
        input_=input.unsqueeze(1)
        x=self.conv(input_)
        if self.verbose:
            print("x.shape:",x.shape)
            print("x.val:",torch.min(x),torch.max(x))
        x=x.view(x.size(0),-1)
        #predicting plane
        plane_x=self.plane_predictors[0](x)
        plane_y=self.plane_predictors[1](x)
        plane_z=self.plane_predictors[2](x)
        if self.verbose:
            print("plane_x.shape:",plane_x.shape)
            print("plane_x.val:",plane_x)
            print("plane_y.shape:",plane_y.shape)
            print("palne_y.val",plane_y)
            print("plane_z.shape:",plane_z.shape)
            print("plane_z.val",plane_z)
        plane_x=self.normalizer(plane_x)
        plane_y=self.normalizer(plane_y)
        plane_z=self.normalizer(plane_z)
        #predicting rotation
        rot_x=self.rot_predictors[0](x)
        rot_y=self.rot_predictors[1](x)
        rot_z=self.rot_predictors[2](x)
        if self.verbose:
            print("rot_x.shape",rot_x.shape)
            print("rot_x.val:",rot_x)
            print("rot_y.shape",rot_y.shape)
            print("rot_y.val:",rot_y)
            print("rot_z.shape",rot_z.shape)
            print("rot_z.val:",rot_z)
        rot_x=F.normalize(rot_x,dim=1,p=2)
        rot_y=F.normalize(rot_y,dim=1,p=2)
        rot_z=F.normalize(rot_z,dim=1,p=2)
        return plane_x,plane_y,plane_z,rot_x,rot_y,rot_z
    
if __name__ == "__main__":
    net=Network()
    input=torch.randn(10,32,32,32)
    plane_x,plane_y,plane_z,rot_x,rot_y,rot_z=net(input)
    print(plane_x.shape)
    print(plane_y.shape)
    print(plane_z.shape)
    print(rot_x.shape)
    print(rot_y.shape)
    print(rot_z.shape)