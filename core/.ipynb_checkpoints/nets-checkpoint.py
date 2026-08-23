import torch
import torch.optim.swa_utils as swa_utils
import torchsde
from torch import nn
import numpy as np
import pdb
from typing import Sequence,Optional, Iterable, Any, Tuple, Dict
from hypernn.torch import TorchHyperNetwork
import math

def LipSwish(x):
  
    return 0.909 * torch.nn.functional.silu(x)


class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers):
        super().__init__()

        model = [torch.nn.Linear(in_size, mlp_size),
                 torch.nn.ReLU()]  

        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            model.append(torch.nn.ReLU())  

        model.append(torch.nn.Linear(mlp_size, out_size))
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)

class MLPL(torch.nn.Module):
    def __init__(self, in_size, out_size, layers):
        super().__init__()

        model = [torch.nn.Linear(in_size, layers[0]),
                 torch.nn.SiLU()] 

        for j in range(len(layers) - 1):
            model.append(torch.nn.Linear(layers[j], layers[j + 1]))
            model.append(torch.nn.SiLU())  

        model.append(torch.nn.Linear(layers[-1], out_size))
        # model.append(torch.nn.Tanh())  # optional output nonlinearity

        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)

class StableResNODEMLP(nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers):
        super().__init__()

        self.inp = nn.Linear(in_size, mlp_size)
        self.blocks = nn.ModuleList([
            nn.Linear(mlp_size, mlp_size) for _ in range(num_layers)
        ])
        self.out = nn.Linear(mlp_size, out_size)

        self.act = torch.nn.SiLU()
        self.dt = 1.0 / num_layers

        for layer in self.blocks:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.inp(x)
        for layer in self.blocks:
            h = layer(self.act(x))
            x = x + self.dt * h
        return self.out(x)


class LinearNet(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        nnlist=[]
        nnlist.append(nn.Linear(in_size, out_size, bias=False))         
        self.linear_modules = nn.ModuleList(nnlist)
        
        
    def forward(self, x):

        y = self.linear_modules[-1](x)
        return y

def init_hyperfan(
    hypernet,
    target_net,
    method='harmonic',
    use_xavier=True,
    input_variance=1.0,
    normal=True
):
    """
    Initialize a (possibly linear) hypernetwork using hyperfan initialization
    to generate weights for a target network.

    Args:
        hypernet (nn.Module): Hypernetwork (can be linear or deep).
        target_net (nn.Module): Target network (MLP, Linear, etc.).
        method (str): 'in', 'out', or 'harmonic' scaling rule.
        use_xavier (bool): Use Xavier (True) or He (False) scaling.
        input_variance (float): Variance of hypernetwork input embeddings.
        normal (bool): Use normal or uniform distribution.

    Returns:
        nn.Module: Initialized hypernetwork.
    """

    if method not in ['in', 'out', 'harmonic']:
        raise ValueError(f"Invalid method: {method}. Choose 'in', 'out', or 'harmonic'.")

    # ---- Collect all linear layers in hypernetwork ----
    linear_layers = [m for m in hypernet.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        raise ValueError("Hypernetwork must contain at least one nn.Linear layer.")

    # ---- Collect shapes of target network weights/biases ----
    if hasattr(target_net, 'target_shapes'):
        target_shapes = target_net.target_shapes
    else:
        target_shapes = []
        for m in target_net.modules():
            if isinstance(m, nn.Linear):
                target_shapes.append(tuple(m.weight.shape))
                if m.bias is not None:
                    target_shapes.append(tuple(m.bias.shape))
        if not target_shapes:
            raise ValueError("Target network must have at least one nn.Linear layer.")

    total_out_size = sum(int(torch.prod(torch.tensor(s))) for s in target_shapes)
    output_layer = linear_layers[-1]

    if output_layer.out_features != total_out_size:
        raise ValueError(
            f"Hypernetwork output size ({output_layer.out_features}) must "
            f"match total target parameter size ({total_out_size})."
        )

    # ---- Initialize hidden layers (if any) ----
    if len(linear_layers) > 1:
        for layer in linear_layers[:-1]:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
            std = math.sqrt(1 / fan_in) if use_xavier else math.sqrt(2 / fan_in)
            if normal:
                nn.init.normal_(layer.weight, 0, std)
            else:
                a = math.sqrt(3) * std
                nn.init.uniform_(layer.weight, -a, a)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    # ---- Initialize output layer (main hyperfan scaling) ----
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(output_layer.weight)
    c_relu = 1 if use_xavier else 2  # correction factor for nonlinearity
    meta = [
        {'name': 'weight' if len(s) > 1 else 'bias', 'layer': i // 2}
        for i, s in enumerate(target_shapes)
    ]

    # ---- Compute proper scaling for each generated block ----
    s_ind = 0
    for out_shape, m in zip(target_shapes, meta):
        e_ind = s_ind + int(torch.prod(torch.tensor(out_shape)))

        if m['name'] == 'weight':
            m_fan_in, m_fan_out = nn.init._calculate_fan_in_and_fan_out(torch.zeros(out_shape))
            var_in = c_relu / (m_fan_in * fan_in * input_variance)
            var_out = c_relu / (m_fan_out * fan_in * input_variance)
        else:
            # Bias case: smaller variance
            m_fan_out = out_shape[0]
            m_fan_in = m_fan_out
            var_in = c_relu / (2 * fan_in * input_variance)
            var_out = max(0, c_relu * (1 - m_fan_in / m_fan_out) / (fan_in * input_variance))

        if method == 'in':
            var = var_in
        elif method == 'out':
            var = var_out
        else:  # harmonic mean
            var = 2 / (1 / var_in + 1 / var_out) if var_out > 0 else var_in

        std = math.sqrt(var)
        a = math.sqrt(3) * std

        # --- Handle both linear and multi-layer case uniformly ---
        if len(linear_layers) == 1:
            layer = output_layer  # linear hypernet case
        else:
            layer = output_layer  # multi-layer case still same final layer

        if normal:
            nn.init.normal_(layer.weight[:, s_ind:e_ind], 0, std)
        else:
            nn.init.uniform_(layer.weight[:, s_ind:e_ind], -a, a)

        s_ind = e_ind

    if output_layer.bias is not None:
        nn.init.zeros_(output_layer.bias)

    return hypernet


class MLPHypernetwork(TorchHyperNetwork):
    def __init__(
        self,
        target_network: nn.Module,
        hyper_dims: int,
        layers: list,
        num_target_parameters: Optional[int] = None,
        
        
        weight_chunk_dim: Optional[int] = None,
        hyperfan=False,
        skip=False
    ):
        super().__init__(
                    target_network = target_network,
                    num_target_parameters = num_target_parameters,
                )
 
        self.hyper_dims = hyper_dims
        self.layers = layers
        self.skip= skip
        self.weight_generator = self.make_weight_generator()   
        
 
        if hyperfan:
            
            self.weight_generator = init_hyperfan(self.weight_generator, target_network)
        
    def make_weight_generator(self) -> nn.Module:
      
        if len(self.layers)==0:  
            return LinearNet(self.hyper_dims, self.num_target_parameters)
        else:
            if self.skip:
                return StableResNODEMLP(self.hyper_dims, self.num_target_parameters,self.layers[0], len(self.layers))
            else:
                return MLPL(self.hyper_dims, self.num_target_parameters, self.layers)

    
    def generate_params(
        self, inp: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
  
        generated_params = (self.weight_generator(inp).view(-1))
               
        return generated_params  
        
    def forward(
        self,
        inp,
        
        generated_params: Optional[torch.Tensor] = None,
        has_aux: bool = False,
        assert_parameter_shapes: bool = True,
        generate_params_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):

        aux_output = {}
        if generated_params is None:
            generated_params = self.generate_params(
                **generate_params_kwargs
            )

        return self.target_forward(
            inp,
            generated_params=generated_params,
            assert_parameter_shapes=assert_parameter_shapes,
            **kwargs,
        ) 