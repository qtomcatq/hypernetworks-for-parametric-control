import pynvml
import torch
import torch.optim.swa_utils as swa_utils
from torch import nn
import numpy as np
import scipy
from scipy.linalg import expm, eigh
import control
from typing import Sequence,Optional, Iterable, Any, Tuple, Dict
import pdb
import networkx as nx
import jax
import jax.numpy as jnp
from jax import vmap, jit, lax
from scipy import signal
from jax.scipy.linalg import expm
jax.config.update("jax_enable_x64", True)


def loss_compute(ys_tray, state_size, weight):

    dif=torch.sum(ys_tray[-1,:,0:state_size]**2,dim=1)

    dif1=weight*ys_tray[-1,:,-1]   
    diff=dif+dif1

    lossvec= torch.mean(diff) 
    logtrick= torch.mean(torch.log(diff))
    stdvec=torch.std(diff)

    
    energy= torch.mean(dif1)
    stdener= torch.std(dif1)
    return lossvec, logtrick, stdvec, energy, stdener

def loss_compute_kur(ys_tray, state_size,batch_size, weight,A):

    obj= objective(ys_tray[-1,:,0:state_size],A)
    dif= weight*ys_tray[-1,:,-1]
  
    lossvec=obj+dif
    loss=torch.mean(lossvec)
    ste_loss=torch.std(lossvec)/np.sqrt(batch_size)
    logtrick= torch.mean(torch.log(lossvec))
    return loss, ste_loss, logtrick
        


def compute_gradnorm(SML):
    total_sq_norm=0
    
    for p in SML.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_sq_norm += param_norm.item() ** 2
                
    return total_sq_norm
def sample_linear(dim,batch_size, device):
    
    amat = torch.rand((int(batch_size/2), dim), device=device)-0.5
    amat = torch.cat([amat, -amat], dim=0)
    x0_raw = torch.randn(int(batch_size/2), dim, device=device) 
    x0_raw = torch.cat([x0_raw,-x0_raw],dim=0)
    norms = torch.linalg.norm(x0_raw, dim=1, keepdim=True) 
    x0 = x0_raw / norms  

    return amat, x0

def sample_kuramoto(state_size,batch_size, dist, scale, scale_kin, device):
    
    x0_raw = torch.randn(int(batch_size/2), state_size, device=device) 
    x0_raw = torch.cat([x0_raw,-x0_raw],dim=0)
    ys_exp = torch.cat((x0_raw/scale_kin,torch.zeros(batch_size, 1).to(device)),dim=1).to(device) 
    low_cdf = dist.cdf(torch.tensor(-3).to(device))
    high_cdf = dist.cdf(torch.tensor(3).to(device))
    
    # 2. Sample Uniformly in CDF space and transform back (ICDF)
    u = torch.rand(int(batch_size/2), state_size, device=device)
    freq1 = dist.icdf(low_cdf + u * (high_cdf - low_cdf))
    
    # 3. Concatenate antithetic pairs
    freqs = torch.cat((freq1, -freq1), dim=0)/scale
    
    return freqs, ys_exp





def save_model(score, ema_loss, ema_model, SML, path,strategy,layers):
    if score>np.sqrt(ema_loss):
            score=np.sqrt(ema_loss)
            torch.save(ema_model.state_dict(), path+"/" + strategy + "ema" + str(layers)+ ".pth")
            torch.save(SML.state_dict(), path+"/" + strategy  + str(layers)+ ".pth")
        
    return score


def update_ema_model(model, ema_model, decay):
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data


def ema_update_trackers(j,alpha, lossvec,energy,stdener,stdvec,ema_loss,ema_ener,enr_std,ema_std):

    if j==0:
        ema_loss=lossvec.cpu().detach().numpy()
        ema_ener=energy.cpu().detach().numpy()
        enr_std=stdener.cpu().detach().numpy()
        ema_std=stdvec.cpu().detach().numpy()           
    else:
        ema_loss=ema_loss*alpha+lossvec.cpu().detach().numpy()*(1-alpha)
        ema_ener=ema_ener*alpha+energy.cpu().detach().numpy()*(1-alpha)
        enr_std=enr_std*alpha+stdener.cpu().detach().numpy()*(1-alpha)
        ema_std=ema_std*alpha+stdvec.cpu().detach().numpy()*(1-alpha)
    return [ema_loss, ema_ener, enr_std, ema_std]
    
def mass_append(lossvec,stdvec,energy,stdener, batch_size, ema_loss, ema_ener, enr_std, ema_std, eperf, stperf, eloss, stloss, emaperf, emaener, emastperf, emastener,total_sq_norm,gradnorms ):
    
    eperf.append(lossvec.cpu().detach().numpy()) 
    stperf.append(stdvec.cpu().detach().numpy()/(np.sqrt(batch_size)))
    eloss.append(energy.cpu().detach().numpy())
    stloss.append(stdener.cpu().detach().numpy()/(np.sqrt(batch_size)))
    emaperf.append(ema_loss) 
    emaener.append(ema_ener)
    emastperf.append(ema_std)
    emastener.append(enr_std)
    gradnorms.append(total_sq_norm)
    return [eperf, stperf, eloss, stloss, emaperf, emaener,emastperf, emastener, gradnorms]

def normal(tensor):
    tensor = (tensor - tensor.mean(dim=0)) / torch.where(tensor.std(dim=0, unbiased=False) == 0, torch.tensor(1.0, device=tensor.device), tensor.std(dim=0, unbiased=False))
    return tensor

def generate_system(dim):
    while True:
        A, B = np.random.randn(dim, dim), np.random.randn(dim, dim)
        A, B = A / np.max(np.abs(np.linalg.eigvals(A))), B / np.max(np.abs(np.linalg.eigvals(B)))
        if np.linalg.matrix_rank(control.ctrb(A, B)) == dim:
            return A, B
        

def generate_system_ez(dim):
    
    A, B = np.random.randn(dim, dim), np.random.randn(dim, dim)
    A, B = A / np.max(np.abs(np.linalg.eigvals(A))), B / np.max(np.abs(np.linalg.eigvals(B)))
    
    return A, B


def random_companion_batch(batch_size, N, pole_range=(-2.0, -0.1)):
    """
    Generate a batch of controllable companion-form (A,B) pairs
    with real poles uniformly sampled from a given range.
    Fully vectorized (no Python loops).
    """
    # 1. Sample random poles for each system
    poles = np.random.uniform(pole_range[0], pole_range[1], size=(batch_size, N))
    
    # 2. Compute polynomial coefficients for each batch (vectorized via np.poly)
    # np.poly operates per-row? No — it’s per-vector, so we’ll use Vandermonde trick
    # For each system, characteristic polynomial: ∏(s - p_i)
    s_powers = np.arange(N, -1, -1)  # exponents N ... 0
    coeffs = np.empty((batch_size, N))
    for i in range(batch_size):  # unavoidable short np.poly call per batch, can use np.poly for speed if smaller batch
        coeffs[i] = np.poly(poles[i])[1:]
    # → shape (batch_size, N+1), first column is 1 (leading coefficient)
    
    # If you truly want *no* Python loop, use np.poly1d + vectorize (slightly slower for small batches):
    # coeffs = np.vectorize(np.poly, signature='(n)->(m)')(poles)
    # 3. Extract and negate coefficients (skip leading 1)
    amat = -coeffs[:, ::-1]
    
    # 4. Build companion matrices in batch (fully vectorized)
    # base companion (same structure for all)
    shift = np.eye(N, k=1)
    A_batch = np.broadcast_to(shift, (batch_size, N, N)).copy()
    A_batch[:, -1, :] = amat  # insert last row
    
    # 5. Build B batch
    B = np.zeros((N, 1))
    B[-1, 0] = 1.0
    B_batch = np.broadcast_to(B, (batch_size, N, 1)).copy()
    
    return A_batch, B_batch, poles


def generate_system_canon(dim, pole_range=(-0.2, -0.1)):

    A = np.zeros((dim, dim))
    if dim > 1:
        A[:-1, 1:] = np.eye(dim - 1)    # superdiagonal of 1s
   
    # Step 4: Build canonical input vector
    B = np.zeros((dim, 1))
    B[-1, 0] = 1.0

  
    return A, B



def get_gpu_memory():
    pynvml.nvmlInit()
    num_devices = pynvml.nvmlDeviceGetCount()
    memory_info = []
    
    for i in range(num_devices):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_info.append({
            'device': i,
            'total_memory': mem_info.total / (1024 ** 2),  # Convert bytes to MB
            'free_memory': mem_info.free / (1024 ** 2),    # Convert bytes to MB
            'used_memory': mem_info.used / (1024 ** 2)     # Convert bytes to MB
        })
    
    pynvml.nvmlShutdown()
    return memory_info
    
def objective(y,A):
    ymat=y.unsqueeze(2)
 
    ytr=ymat.repeat(1,1,y.size(1))
    ytr1=torch.transpose(ytr, 2,1)

    phasematrix=ytr1-ytr

    forcematrix=torch.sin(phasematrix)  
    forcematrix=torch.square(forcematrix)*A
 
    return torch.sum(forcematrix, (1,2))/2


def lipswish(x):
    return 0.909 * jnn.silu(x)

def extract_unique_diagonals(matrices):
    n = matrices[0].shape[0]  # Get the size from the first matrix
    return [
        np.concatenate([
            np.array([matrix[r, c] for r, c in zip(range(0, n-i), range(i, n))]) 
            for i in range(1, n)
        ])
        for matrix in matrices
    ]


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

def global_param(y, L):
    ymat=y.unsqueeze(2)
    
  
    ysin= torch.sin(y)
    ycos= torch.cos(y)

    quad_form_sin = torch.einsum('tbi,ij,tbj->tb', ysin, L, ysin)
    quad_form_cos = torch.einsum('tbi,ij,tbj->tb', ycos, L, ycos)

    quad_forms=(quad_form_sin + quad_form_cos)/(y.size(-1)**2)
    r= torch.ones_like(quad_forms)-quad_forms
    return r

def order_param(y, L):
  
 

    ysin= torch.sin(y)
    ycos= torch.cos(y)
    L=L.unsqueeze(0)
    L=L.repeat(y.size(0),1,1)

    quad_form_sin = torch.einsum('bi,bij,bj->b', ysin, L, ysin)
    quad_form_cos = torch.einsum('bi,bij,bj->b', ycos, L, ycos)

    r = torch.ones([y.size(0)]).to(y.device) - (quad_form_sin + quad_form_cos)/(y.size(1)**2)
    
    return torch.mean(r)


def erdos_graph(dim, p):
   
    
    G=nx.erdos_renyi_graph(dim,p)
    A = nx.adjacency_matrix(G)

    return A.toarray()

def watts_graph(dim, k, p):

    G= nx.watts_strogatz_graph(dim,k,p)
    
    A= nx.adjacency_matrix(G)

    return A.toarray(), G

def adj_lattice(dim, graph, *args):
    
    if graph=="square":
     
        return sqr_adj(dim)
                
    elif graph == "erdos":

        return erdos_graph(dim, args[0])

    elif graph == "watts":

        return watts_graph(dim, args[1], args[0])

def batchsize_selector(i,j):
    if j==0:
        b=1
    else:
        b=256*(j+1)+128*(i+1)       
    return b


def iterr(i,k):
                
    if k==0:
        if i==0:
            iter=2500
            batch=128
        else:
            iter=1
            batch=128
    else:
        if i==0:
            iter=250
            batch=256*(k+1)+128*(i+1)
        elif i==1:
            iter=250
            batch=256*(k+1)+128*(i+1)
        else:
            iter=128
            batch=256*(k+1)+128*(i+1)
    return iter, batch

def Kgain(A,B,omega,gamma,x0, nsteps,total,P0):
    
    V=np.eye(np.shape(x0)[0])*(gamma**2)*(total/nsteps)
    W=np.eye(np.shape(x0)[0])*(omega**2)/(total/nsteps)
    
    N = nsteps
    
    P = [None] * N
   
    #P[0]= 0*np.matmul(x0.reshape((-1, 1)),np.expand_dims(x0,axis=0))
    P[0]=P0
    C = np.eye(np.shape(x0)[1])
    # For i = 0, ..., N - 1
    for i in range(N-1):
       
        # Calculate the optimal feedback gain K
     
        P[i+1]= A @ P[i] @ A.T - (A @ P[i] @ C.T) @ np.linalg.pinv(C @ P[i] @ C.T +W) @ (C @ P[i] @ A.T) + V
            
    L= [None] * N
   
    for i in range(N):
        # Calculate the optimal feedback gain K
        L[i]= (P[i] @ C.T) @ np.linalg.pinv(C @ P[i] @ C.T + W)
    
    return L

def lqr( Q, R, A, B, nsteps, Qf):
    """
    Discrete-time linear quadratic regulator for a nonlinear system.
 
    Compute the optimal control inputs given a nonlinear system, cost matrices, 
    current state, and a final state.
     
    Compute the control variables that minimize the cumulative cost.
 
    Solve for P using the dynamic programming method.
 
    :param actual_state_x: The current state of the system 
        3x1 NumPy Array given the state is [x,y,yaw angle] --->
        [meters, meters, radians]
    :param desired_state_xf: The desired state of the system
        3x1 NumPy Array given the state is [x,y,yaw angle] --->
        [meters, meters, radians]   
    :param Q: The state cost matrix
        3x3 NumPy Array
    :param R: The input cost matrix
        2x2 NumPy Array
    :param dt: The size of the timestep in seconds -> float
 
    :return: u_star: Optimal action u for the current state 
        2x1 NumPy Array given the control input vector is
        [linear velocity of the car, angular velocity of the car]
        [meters per second, radians per second]
    """
    # We want the system to stabilize at desired_state_xf.
  
 
    # Solutions to discrete LQR problems are obtained using the dynamic 
    # programming method.
    # The optimal solution is obtained recursively, starting at the last 
    # timestep and working backwards.
    # You can play with this number
    N = nsteps
 
    # Create a list of N + 1 elements
    P = [None] * (N + 1)
     
   
 
    # LQR via Dynamic Programming
    P[N] = Qf
 
    # For i = N, ..., 1
    for i in range(N, 0, -1):
 
        # Discrete-time Algebraic Riccati equation to calculate the optimal 
        # state cost matrix
     
        P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
            R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)      
        
    # Create a list of N elements
    K = [None] * N
   
   
    # For i = 0, ..., N - 1
    for i in range(N):
 
        # Calculate the optimal feedback gain K
        K[i] = -np.linalg.pinv(R + B.T @ P[i+1] @ B) @ (B.T @ P[i+1] @ A)
 
        
 
    # Optimal control input is u_star
   
 
    return K
   

def lqr_batch_jax(Q, R, A, B, nsteps, Qf):
    """
    Batched finite-horizon discrete-time LQR (JAX).
    Matches the standard discrete Riccati recursion used in the NumPy implementation.

    Args:
      Q: (batch, N, N) or (N, N)
      R: (batch, M, M) or (M, M)
      A: (batch, N, N)
      B: (batch, N, M)
      nsteps: int
      Qf: (batch, N, N) or (N, N)

    Returns:
      Ks: (batch, nsteps, M, N)  feedback gains K_t for t=0..nsteps-1
    """
    # Force arrays to jax arrays and float64
    Q = jnp.asarray(Q)
    R = jnp.asarray(R)
    A = jnp.asarray(A)
    B = jnp.asarray(B)
    Qf = jnp.asarray(Qf)

    batch = A.shape[0]
    N = A.shape[1]
    M = B.shape[2]

    # Broadcast Q, R, Qf if given as matrices
    if Q.ndim == 2:
        Q = jnp.broadcast_to(Q, (batch, N, N))
    if R.ndim == 2:
        R = jnp.broadcast_to(R, (batch, M, M))
    if Qf.ndim == 2:
        Qf = jnp.broadcast_to(Qf, (batch, N, N))

    # Precompute transposes
    At = jnp.swapaxes(A, -1, -2)   # (batch, N, N)
    Bt = jnp.swapaxes(B, -1, -2)   # (batch, M, N)

    # vectorized pinv over batch
    batched_pinv = vmap(jnp.linalg.pinv, in_axes=0, out_axes=0)

    def backward_step(P_next, _):
        # P_next: (batch, N, N)
        # S = R + B^T P_next B  -> (batch, M, M)
        S = R + Bt @ P_next  @ B                  # (batch, M, M)

        # compute inverse/psinv of S
        S_inv = batched_pinv(S)          # (batch, M, M)

        # term = B^T P_next A  -> (batch, M, N)
        term = Bt @ (P_next @ A)         # (batch, M, N)

        # K_t = - S_inv @ term
        K_t = - (S_inv @ term)           # (batch, M, N)

        
        # compute P_next @ A - P_next @ B @ S_inv @ (B^T @ P_next @ A)
        Pterm = P_next @ A  - (P_next @ B) @ (S_inv @ term)  # (batch, N, N)
        P_t = Q + At @ Pterm                         # (batch, N, N)

        return P_t, K_t

    # Run scan starting from Qf (this computes P_{N-1}, P_{N-2}, ..., P_0 in sequence)
    # lax.scan will produce sequence [P_t(1), P_t(2), ...] where first step uses init=Qf
    P_seq, K_seq = lax.scan(backward_step, Qf, None, length=nsteps)
    # P_seq shape: (nsteps, batch, N, N)
    # K_seq shape: (nsteps, batch, M, N)

    # K_seq[0] corresponds to K_{N-1} (first backward step). We want chronological order K_0..K_{N-1}
    K_seq = jnp.flip(K_seq, axis=0)      # now shape (nsteps, batch, M, N)
    Ks = jnp.swapaxes(K_seq, 0, 1)       # (batch, nsteps, M, N)

    return Ks


def lqr_batch_jax_heun(Q, R, A, B, nsteps, Qf):
    Q = jnp.asarray(Q)
    R = jnp.asarray(R)
    A = jnp.asarray(A)
    B = jnp.asarray(B)
    Qf = jnp.asarray(Qf)

    batch, N, _ = A.shape
    M = B.shape[2]
    h = 1 / nsteps

    if Q.ndim == 2:
        Q = jnp.broadcast_to(Q, (batch, N, N))
    if R.ndim == 2:
        R = jnp.broadcast_to(R, (batch, M, M))
    if Qf.ndim == 2:
        Qf = jnp.broadcast_to(Qf, (batch, N, N))

    At = jnp.swapaxes(A, -1, -2)
    Bt = jnp.swapaxes(B, -1, -2)

    batched_pinv = vmap(jnp.linalg.pinv, in_axes=0, out_axes=0)
    R_inv = batched_pinv(R)
    def riccati_rhs(P):
        BtP = Bt @ P
        
        PB = P @ B
        feedback = PB @ (R_inv @ BtP)
        return -(At @ P + P @ A - feedback + Q)

    def backward_step(P_next, _):
        k1 = riccati_rhs(P_next)
        P_pred = P_next - h * k1
        k2 = riccati_rhs(P_pred)
        P_t = P_next - 0.5 * h * (k1 + k2)

        K_t = -(R_inv  @ (Bt @ P_t))
        return P_t, K_t

    P_seq, K_seq = lax.scan(backward_step, Qf, None, length=nsteps)

    K_seq = jnp.flip(K_seq, axis=0)
    Ks = jnp.swapaxes(K_seq, 0, 1)

    return Ks


def grammian(A,B,total):
    N=1000
    dt=total/N
    t = np.linspace(0, total, N, endpoint=False)
    Bp=np.matmul(B, np.transpose(B))
    ans=0*A
    for i in range(0, N):
        tc=t[i]
        A1=expm(A*tc)
        A2=expm(np.transpose(A)*tc)
        ft=np.matmul(A1,Bp)
     
        ans=ans+dt*np.matmul(ft,A2)
    return ans

def grammian_single(A, B, total):
    """Calculates the Gramian integral for single matrices A and B."""
    N = 1000
    dt = total / N
    t = jnp.linspace(0, total, N, endpoint=False)
    Bp = B @ B.T
    def body(i, acc):
        tc = t[i]
        A1 = expm(A * tc)
        A2 = expm(A.T * tc)
        return acc + dt * (A1 @ Bp @ A2)
    return jax.lax.fori_loop(0, N, body, jnp.zeros_like(A))

def vt(A,x0,total):
    A1=expm(A*total)
 
    v1=-np.matmul(A1,x0[0])
    return v1

def load_clean_state_dict(model, path):
    state_dict = torch.load(path)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")  # fix if saved with DataParallel
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict)