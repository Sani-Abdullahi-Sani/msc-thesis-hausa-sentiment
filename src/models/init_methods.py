"""
Weight initialization methods module.
Provides different initialization methods for neural network weights.
"""

import math
import torch
import torch.nn.init as init

def xavier_init(tensor, gain=1.0):
    """
    Xavier (Glorot) initialization.
    
    Args:
        tensor: Tensor to initialize
        gain: Gain factor
        
    Returns:
        Initialized tensor
    """
    return init.xavier_uniform_(tensor, gain=gain)

def he_init(tensor, gain=1.0):
    """
    He initialization.
    
    Args:
        tensor: Tensor to initialize
        gain: Gain factor
        
    Returns:
        Initialized tensor
    """
    return init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='relu')

def generalized_init(tensor, beta, gain=1.0, mode='fan_in'):
    """
    Generalized initialization method.
    
    Args:
        tensor: Tensor to initialize
        beta: Beta factor for scaling
        gain: Gain factor
        mode: Fan mode ('fan_in' or 'fan_out')
        
    Returns:
        Initialized tensor
    """
    fan = init._calculate_correct_fan(tensor, mode)
    bound = beta * math.sqrt(gain / fan)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
    
def init_weights(model, method='xavier', beta=2.0):
    """
    Initialize weights of a model using the specified method.
    
    Args:
        model: PyTorch model
        method: Initialization method ('xavier', 'he', or 'generalized')
        beta: Beta factor for generalized initialization
        
    Returns:
        Model with initialized weights
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            if method == 'xavier':
                xavier_init(param.data)
            elif method == 'he':
                he_init(param.data)
            elif method == 'generalized':
                generalized_init(param.data, beta=beta)
            else:
                raise ValueError(f"Unknown initialization method: {method}")
                
        elif 'bias' in name:
            param.data.zero_()
    
    return model