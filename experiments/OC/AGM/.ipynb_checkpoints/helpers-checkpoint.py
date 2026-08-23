
import fire
import matplotlib.pyplot as plt
import torch
import torch.optim.swa_utils as swa_utils
import torchcde
import torchsde
from torch import nn
import copy
import matplotlib.pyplot as plt
from torch import profiler
from tqdm import tqdm
import numpy as np
import pdb
from torchdiffeq import odeint
#from hilo_mpc import Model, NMPC, UKF
import pynvml
import geoopt
from typing import Sequence,Optional, Iterable, Any, Tuple, Dict
# static hypernetwork

import networkx as nx
import scipy


def laplacian_from_adjacency(adj_matrix):
    # Degree matrix (diagonal matrix with degree of nodes)
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    # Laplacian matrix: L = D - A
    laplacian = degree_matrix - adj_matrix
    return laplacian

def criticalK(adjm, data_size, scale):
    L=laplacian_from_adjacency(adjm)
    #L=np.matmul(adjm,np.transpose(adjm))
    
    eigvals = scipy.linalg.eigh(L, eigvals_only=True)

    eigvals = np.sort(eigvals)

    # Get largest and second smallest
    second_smallest = eigvals[1]
    largest = eigvals[-1]
   
    K= data_size * (np.pi**2/4) * np.sqrt(data_size) * largest/ (scale*second_smallest**2)

    return K, L


def calculate_norm(params):
    norm=0
    norm0=0
    for i in range(0,len(params)):
       
        norm+=torch.sum((params[i].grad)**2)
        norm0+=torch.sum((params[i])**2)
    return norm,norm0

def calculate_diffs(ys_tray):
    energies=ys_tray[:,:,-1]
   
    diffs=energies[1:,:]-energies[0:-1,:]
    return diffs

def objective(y,A):
    ymat=y.unsqueeze(2)
 
    ytr=ymat.repeat(1,1,y.size(1))
    ytr1=torch.transpose(ytr, 2,1)

    phasematrix=ytr1-ytr

    forcematrix=torch.sin(phasematrix)  
    forcematrix=torch.square(forcematrix)*A
 
    return torch.sum(forcematrix, (1,2))/2


def order_param(y, L):
    ymat=y.unsqueeze(2)
    

    ysin= torch.sin(y)
    ycos= torch.cos(y)
    L=L.unsqueeze(0)
    L=L.repeat(y.size(0),1,1)

    quad_form_sin = torch.einsum('bi,bij,bj->b', ysin, L, ysin)
    quad_form_cos = torch.einsum('bi,bij,bj->b', ycos, L, ycos)

    r = torch.ones([y.size(0)]).to(y.device) - (quad_form_sin + quad_form_cos)/(y.size(1)**2)
    
    return r

def global_param(y, L):
    ymat=y.unsqueeze(2)
    
  
    ysin= torch.sin(y)
    ycos= torch.cos(y)

    quad_form_sin = torch.einsum('tbi,ij,tbj->tb', ysin, L, ysin)
    quad_form_cos = torch.einsum('tbi,ij,tbj->tb', ycos, L, ycos)

    quad_forms=(quad_form_sin + quad_form_cos)/(y.size(-1)**2)
    r= torch.ones_like(quad_forms)-quad_forms
    return r


def sqr_adj(dimm):
    dim=np.sqrt(dimm)
    dim=int(dim)
    adj=np.zeros((dim*dim,dim*dim))

    for i in range(0, dim*dim):
        for j in range(0, dim*dim):

            if (j+1)==i and (i%dim != 0):
                adj[i,j]=1
            if (j-1)==i and (j%dim != 0):
                adj[i,j]=1
            if (j-dim)==i:
                adj[i,j]=1
            if (j+dim)==i:
                adj[i,j]=1
  
    return adj

def erdos_graph(dim, p):
   
    
    G=nx.erdos_renyi_graph(dim,p)
    A = nx.adjacency_matrix(G)

    return A.toarray()

def watts_graph(dim, k, p):

    G= nx.watts_strogatz_graph(dim,k,p)
    A= nx.adjacency_matrix(G)

    return A.toarray()

def adj_lattice(dim, graph, *args):
    
    if graph=="square":
     
        return sqr_adj(dim)
                
    elif graph == "erdos":

        return erdos_graph(dim, args[0])

    elif graph == "watts":

        return watts_graph(dim, args[1], args[0])


class KuramotoAGM(nn.Module):
    noise_type = "diagonal"
    sde_type = "stratonovich"
    
    def __init__(self, batch, dim, K, A, freqs,force_field,dt):
        super(KuramotoAGM, self).__init__()
        self.batch=batch
        self.dim=dim
        self.K=K
        self.freqs=freqs
        self.A = A
        
        
        self.dt=dt
        self.u= force_field
    def f(self, t, y):
     
       
        index = (t / self.dt).floor().long()
        cforce=self.u[index,:,:]
     
       
        ymat=y[:,0:self.dim].unsqueeze(2)
        
        ytr=ymat.repeat(1,1,self.dim)
        ytr1=torch.transpose(ytr, 2,1)
   
        phasematrix=ytr1-ytr
      
        forcematrix=torch.sin(phasematrix)*self.A
      
        force=self.K*cforce*torch.sum(forcematrix,2)/self.dim
   
        force+=self.freqs

        f1=(torch.sum((cforce)**2,dim=1)).unsqueeze(1)
     
        return torch.cat([force,f1],dim=1)

    def g(self, t, y):

        return torch.zeros_like(y)

class AGM_ODE(nn.Module):
    def __init__(self, batch, dim, K, A, freqs,force_field,dt):
        super(AGM_ODE, self).__init__()
        self.batch=batch
        self.dim=dim
        self.K=K
        self.freqs=freqs
        self.A = A
        
        
        self.dt=dt
        self.u= force_field
    def forward(self, t, y):
     
        if t<1:
            index = (t / self.dt).floor().long()
        else:
            index= ((t-0.001)/self.dt).floor().long()
        cforce=self.u[index,:,:]
     
       
        ymat=y[:,0:self.dim].unsqueeze(2)
        
        ytr=ymat.repeat(1,1,self.dim)
        ytr1=torch.transpose(ytr, 2,1)
   
        phasematrix=ytr1-ytr
      
        forcematrix=torch.sin(phasematrix)*self.A
      
        force=self.K*cforce*torch.sum(forcematrix,2)/self.dim
   
        force+=self.freqs

        f1=(torch.sum((cforce)**2,dim=1)).unsqueeze(1)
     
        return torch.cat([force,f1],dim=1)



def adjoint_init(y,dim):
 
    ymat=y.unsqueeze(2)
        
    ytr=ymat.repeat(1,1,dim)
    ytr1=torch.transpose(ytr, 2,1)

    phasematrix=ytr-ytr1
  
    forcematrix=torch.sin(2*phasematrix)
    inits=torch.sum(forcematrix,2)/2
    return inits

def grad_estimate(pfield,y,dim,K):
 
    ymat=y[:,:,:].unsqueeze(3)
    ytr=ymat.repeat(1,1,1,dim)
    ytr1=torch.transpose(ytr,3,2)
    phasematrix=ytr1-ytr
    forcematrix=torch.sin(phasematrix)

    grad=torch.sum(K*pfield*torch.sum(forcematrix,3)/dim,2).unsqueeze(2)
    return grad

def combine_vector_field(temporal, components):

    # Broadcast tensors
    w = temporal[:, :, None]  # (t_size, batch_size, 1)
    u = components[None, :, :]  # (1, batch_size, state_size)
    F =  w * u  # (t_size, batch_size, state_size)
    
    return F

def grad_estimate_vec(pfield,y,temporal,component,dim,R,K):
 
    ymat=y[:,:,:].unsqueeze(3)
    ytr=ymat.repeat(1,1,1,dim)
    ytr1=torch.transpose(ytr,3,2)
    phasematrix=ytr1-ytr
    forcematrix=torch.sin(phasematrix)
    S= pfield*torch.sum(forcematrix,3)
 
    comp_grad=(component*S)
    scal_grad=(temporal*S)

    return comp_grad, scal_grad

class KuramotoAdjoint(nn.Module):
    noise_type = "diagonal"
    sde_type = "stratonovich"
    
    def __init__(self, batch, dim, K, freqs, A,force_field,x_field, dt, device):
        super(KuramotoAdjoint, self).__init__()
        self.batch=batch
        self.dim=dim
        self.K=K
        self.freqs=freqs
        self.device= device
        self.A=A
     
        self.dt=dt
        self.x=x_field
        self.u= force_field
    def f(self, t, y):

        index = (t / self.dt).floor().long()
       
        cforce=self.u[-(index+1),:,:]
     
        x=self.x[-(index+1),:,:]
      
  
        mask = ~torch.eye(self.dim, dtype=torch.bool, device=x.device)
        theta_i = x.unsqueeze(2)  # [B, N, 1]
        theta_k = x.unsqueeze(1)  # [B, 1, N]
        phase_diff = theta_k - theta_i  # [B, N, N]

        cos_delta = self.A*torch.cos(phase_diff)
        masked_cos = cos_delta.masked_fill(~mask, 0.0)  # [B, N, N]

          
     
        term1 = torch.sum(masked_cos *y.unsqueeze(1), dim=1)
        term1 = - (self.K/ self.dim) * cforce*term1
     
    
        term2= torch.sum(masked_cos , dim=1)
        term2 = (self.K/ self.dim) *cforce   * term2 * y

        dp = term1 + term2
        return -dp

    def g(self, t, y):
         
        return torch.zeros_like(y)        

class ODE_Adjoint(nn.Module):

    def __init__(self, batch, dim, K, freqs, A,force_field,x_field, dt, device):
        super(ODE_Adjoint, self).__init__()
        self.batch=batch
        self.dim=dim
        self.K=K
        self.freqs=freqs
        self.device= device
        self.A=A
     
        self.dt=dt
        self.x=x_field
        self.u= force_field
    def forward(self, t, y):

        if t<1:
            index = (t / self.dt).floor().long()
        else:
            index= ((t-0.001)/self.dt).floor().long()
        cforce=self.u[-(index+1),:,:]
     
        x=self.x[-(index+1),:,:]
      
  
        mask = ~torch.eye(self.dim, dtype=torch.bool, device=x.device)
        theta_i = x.unsqueeze(2)  # [B, N, 1]
        theta_k = x.unsqueeze(1)  # [B, 1, N]
        phase_diff = theta_k - theta_i  # [B, N, N]

        cos_delta = self.A*torch.cos(phase_diff)
        masked_cos = cos_delta.masked_fill(~mask, 0.0)  # [B, N, N]

          
     
        term1 = torch.sum(masked_cos *y.unsqueeze(1), dim=1)
        term1 = - (self.K/ self.dim) * cforce*term1
     
    
        term2= torch.sum(masked_cos , dim=1)
        term2 = (self.K/ self.dim) *cforce   * term2 * y

        dp = term1 + term2
        return -dp


def adabelief_update(force_field, grad, m, s, s_max, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Performs one AdaBelief + AMSGrad update step on `force_field`.

    Args:
        force_field (Tensor): The parameter tensor to be updated.
        grad (Tensor): The gradient of the loss w.r.t. force_field.
        m (Tensor): First moment vector (mean of gradient).
        s (Tensor): Second moment vector (belief in gradient).
        s_max (Tensor): Max of second moment vector (AMSGrad correction).
        t (int): Time step (starts at 0).
        lr (float): Learning rate.
        beta1 (float): Exponential decay rate for first moment.
        beta2 (float): Exponential decay rate for second moment.
        eps (float): Small constant for numerical stability.

    Returns:
        Updated (force_field, m, s, s_max, t)
    """
    t += 1

    # First moment estimate
    m = beta1 * m + (1 - beta1) * grad

    # Belief estimate: deviation of grad from its mean (m)
    grad_diff = grad - m
    s = beta2 * s + (1 - beta2) * grad_diff.pow(2)

    # Bias correction
    m_hat = m / (1 - beta1 ** t)
    s_hat = s / (1 - beta2 ** t)

    # AMSGrad correction
    s_max = torch.maximum(s_max, s_hat)

    # Update step
    force_field = force_field - lr * m_hat / (s_hat.sqrt() + eps)
    return force_field, m, s, s_max, t

def adabelief_update_manifold(force_field, grad, m, s, s_max, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-9):
    """
    Performs one AdaBelief + AMSGrad update step on `force_field`.

    Args:
        force_field (Tensor): The parameter tensor to be updated.
        grad (Tensor): The gradient of the loss w.r.t. force_field.
        m (Tensor): First moment vector (mean of gradient).
        s (Tensor): Second moment vector (belief in gradient).
        s_max (Tensor): Max of second moment vector (AMSGrad correction).
        t (int): Time step (starts at 0).
        lr (float): Learning rate.
        beta1 (float): Exponential decay rate for first moment.
        beta2 (float): Exponential decay rate for second moment.
        eps (float): Small constant for numerical stability.

    Returns:
        Updated (force_field, m, s, s_max, t)
    """
    t += 1
    
    manifold = geoopt.Sphere()
    u_manifold = geoopt.ManifoldTensor(force_field, manifold=manifold)
    grad = manifold.proju(u_manifold,grad )
    # First moment estimate
    m = beta1 * m + (1 - beta1) * grad

    # Belief estimate: deviation of grad from its mean (m)
    grad_diff = grad - m
    s = beta2 * s + (1 - beta2) * grad_diff.pow(2)

    # Bias correction
    m_hat = m / (1 - beta1 ** t)
    s_hat = s / (1 - beta2 ** t)

    # AMSGrad correction
    s_max = torch.maximum(s_max, s_hat)

    # Update step

    v_tangent = manifold.proju(u_manifold,- lr * m_hat / (s_hat.sqrt() + eps) )
    force_field_upd= manifold.expmap(force_field,  v_tangent)

    m_upd=manifold.transp(force_field, force_field_upd, m)

    return force_field_upd, m_upd, s, s_max, t

