import torch
import torchsde
from torch import nn
import numpy as np
import pdb
from core.nets import MLPL, Hypernetwork, LinearNet, StableResNODEMLP


def find_params(SML):
    
    params = None
    
    if SML.strategy == "HNC":
        
        params = SML.poly.hhnet.weight_generator.parameters()

    elif SML.strategy == "ENC":

        params = SML.poly.polyf.parameters()
        
    return params 



class HNC_strategy(nn.Module):
    def __init__(self,dim,outputdim, layers,batch,device, skip,  act, clip,std_scales, amat=None):
        super().__init__()
        self.batch=batch
        self.dim=dim
        self.amat=amat
        self.clip=clip
        self.device=device
        self.std_scales=std_scales
        self.polyf = LinearNet(in_size=self.dim, out_size=outputdim)

        self.hhnet = Hypernetwork( hyper_dims = 1+self.dim, target_network = self.polyf, layers=(2+self.dim,)*layers, hyperfan=True, skip=skip, act=act)

    def forward(self, t, y):
        
        ts=torch.full((self.batch, 1), t).to(self.device)
     
        hyper_inp=torch.cat(( (ts-0.5)*self.std_scales[0],self.amat*self.std_scales[1]),dim=1)
   
        generated_vmap_params= (torch.vmap(self.hhnet.generate_params)(hyper_inp)).clamp_(-self.clip, self.clip)
  
        cforce = torch.vmap(self.hhnet.forward)(y[:,0:self.dim],generated_vmap_params )
        return cforce  
        
class ENC_strategy(nn.Module):
    def __init__(self, dim,outputdim, layers,batch,device, skip, act, std_scales, amat= None):
        super().__init__()
        self.batch=batch
        self.dim=dim
        self.amat=amat
        self.std_scales=std_scales
        self.device = device
        if skip:
            self.polyf = StableResNODEMLP(in_size=1+2*self.dim,out_size=outputdim,mlp_size=2+2*self.dim,num_layers=layers, act=act)  
        else:
            self.polyf = MLPL(in_size=1+2*self.dim,out_size=outputdim,layers=(2+2*self.dim,)*layers,act=act)  
        
    def forward(self, t, y):
        ts=torch.full((self.batch, 1), t).to(self.device)
        
        cforce=self.polyf(torch.cat(( (ts-0.5), self.amat ,y[:,0:self.dim]),dim=1))
        
        return cforce  
    



class HyperCoeffsLinearControlStochasticLQRImpl(torch.nn.Module):

    noise_type = "diagonal"
    sde_type = "stratonovich"

    def __init__(self, A, B, batch_size,  device, hlayers, strategy, skip=False, act="SiLU",clip=10**2,std_scales=(1,1,1)):
       
        super(HyperCoeffsLinearControlStochasticLQRImpl, self).__init__()
        self.batch=batch_size
        self.At = A.to(device)
        self.Bt = B.to(device)
       
        self.strategy= strategy
        self.dim=A.size(0)
        outputdim= B.size(1)
     
        
        hlayers=hlayers

        if strategy == "HNC":
            
            self.poly = HNC_strategy(self.dim,outputdim, hlayers,batch_size,device,skip,act,clip,std_scales)
        
        elif strategy == "ENC":

            self.poly = ENC_strategy(self.dim,outputdim, hlayers,batch_size,device, skip,act, std_scales)
   
    def f(self, t, y):

        forces = self.poly(t,y)

        f = torch.einsum('kji,ki->kj', self.At, y[:,0:self.dim]) + torch.einsum('ji,ki->kj', self.Bt, forces)
     
        f2 = (torch.sum((forces**2),dim=1)).unsqueeze(1)
       
        return torch.cat([f,f2],dim=1)

    def g(self, t, y):
        
        noise=0.00*torch.ones_like(y[:,0:self.dim]) 
        zero_pad=0*y[:,0].unsqueeze(1)
       
        return torch.cat([noise,zero_pad],dim=1)


