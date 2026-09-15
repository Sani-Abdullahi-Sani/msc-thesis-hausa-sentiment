import math
import torch
import torch.nn.init as init

def xavier_init(tensor, gain=1.0):
    """
    Xavier (Glorot) initialization.
    
    Args:
        tensor: Weight tensor to initialize
        gain: Gain factor for initialization
    
    Returns:
        Initialized tensor
    """
    return init.xavier_uniform_(tensor, gain=gain)

def he_init(tensor, gain=1.0):
    """
    He initialization.
    
    Args:
        tensor: Weight tensor to initialize  
        gain: Gain factor for initialization
    
    Returns:
        Initialized tensor
    """
    return init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='relu')

def generalized_init(tensor, beta, gain=1.0, mode='fan_in'):
    """
    Generalized initialization method - uniform beta for all layers.
    
    Args:
        tensor: Weight tensor to initialize
        beta: Beta scaling factor
        gain: Gain factor for initialization
        mode: Fan mode ('fan_in', 'fan_out')
    
    Returns:
        Initialized tensor
    """
    fan = init._calculate_correct_fan(tensor, mode)
    bound = beta * math.sqrt(gain / fan)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def generalized_init_imbalanced(tensor, beta_input, beta_output, layer_type, gain=1.0, mode='fan_in'):
    """
    Generalized initialization with imbalanced beta values.
    
    This implements the research idea where output layers are initialized with 
    larger values than input layers to promote different training regimes.
    
    Args:
        tensor: Weight tensor to initialize
        beta_input: Beta value for input/early layers (typically smaller, e.g., 1.0)
        beta_output: Beta value for output/late layers (typically larger, e.g., 3.0-4.0)
        layer_type: String indicating layer position ('input', 'middle', 'output')
        gain: Gain factor for initialization  
        mode: Fan mode ('fan_in', 'fan_out')
    
    Returns:
        Initialized tensor
    """
    # Choose beta based on layer type
    if layer_type == 'input':
        beta = beta_input
    elif layer_type == 'output':
        beta = beta_output
    elif layer_type == 'middle':
        # For middle layers, use interpolated value
        beta = (beta_input + beta_output) / 2
    else:
        raise ValueError(f"Unknown layer_type: {layer_type}. Use 'input', 'middle', or 'output'")
    
    # Apply generalized initialization with chosen beta
    fan = init._calculate_correct_fan(tensor, mode)
    bound = beta * math.sqrt(gain / fan)
    
    with torch.no_grad():
        tensor.uniform_(-bound, bound)
    
    return tensor

def get_layer_type(module_name, module, is_embedding=False, is_classifier=False):
    """
    Determine layer type for imbalanced initialization.
    
    Args:
        module_name: Name of the module (for debugging)
        module: The actual module
        is_embedding: Whether this is the embedding layer
        is_classifier: Whether this is the final classifier layer
    
    Returns:
        String: 'input', 'middle', or 'output'
    """
    if is_embedding:
        return 'input'
    elif is_classifier:
        return 'output'
    else:
        # For transformer internal layers, classify as middle
        return 'middle'

class ImbalancedInitializer:
    """
    Helper class to manage imbalanced initialization across a model.
    
    This class tracks which layers have been initialized and ensures
    consistent application of the imbalanced initialization strategy.
    """
    
    def __init__(self, beta_input=1.0, beta_output=3.0, gain=1.0, mode='fan_in'):
        self.beta_input = beta_input
        self.beta_output = beta_output
        self.gain = gain
        self.mode = mode
        self.initialized_layers = []
    
    def initialize_layer(self, module, module_name="", is_embedding=False, is_classifier=False):
        """
        Initialize a single layer with appropriate beta value.
        
        Args:
            module: The module to initialize
            module_name: Name of the module (for logging)
            is_embedding: Whether this is an embedding layer
            is_classifier: Whether this is the classifier layer
        """
        if not isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            return
        
        # Determine layer type
        layer_type = get_layer_type(module_name, module, is_embedding, is_classifier)
        
        # Get the appropriate beta value
        if layer_type == 'input':
            beta = self.beta_input
        elif layer_type == 'output': 
            beta = self.beta_output
        else:  # middle
            beta = (self.beta_input + self.beta_output) / 2
        
        # Apply initialization
        generalized_init_imbalanced(
            module.weight, 
            self.beta_input, 
            self.beta_output, 
            layer_type, 
            self.gain, 
            self.mode
        )
        
        # Zero biases
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.data.zero_()
        
        # Log initialization
        layer_info = {
            'name': module_name,
            'type': layer_type,
            'beta': beta,
            'shape': list(module.weight.shape)
        }
        self.initialized_layers.append(layer_info)
        
        print(f"✓ Initialized {module_name} ({layer_type}) with β={beta:.1f}, shape={module.weight.shape}")
    
    def get_initialization_summary(self):
        """Get summary of all initialized layers."""
        summary = {
            'total_layers': len(self.initialized_layers),
            'input_layers': len([l for l in self.initialized_layers if l['type'] == 'input']),
            'middle_layers': len([l for l in self.initialized_layers if l['type'] == 'middle']),
            'output_layers': len([l for l in self.initialized_layers if l['type'] == 'output']),
            'beta_input': self.beta_input,
            'beta_output': self.beta_output,
            'imbalance_ratio': self.beta_output / self.beta_input
        }
        return summary

def apply_initialization(model, init_method='xavier', beta=2.0, beta_input=1.0, beta_output=3.0, gain=1.0):
    """
    Apply initialization to entire model.
    
    Args:
        model: PyTorch model to initialize
        init_method: Initialization method ('xavier', 'he', 'generalized', 'generalized_imbalanced')
        beta: Beta value for uniform generalized initialization
        beta_input: Beta value for input layers (imbalanced)
        beta_output: Beta value for output layers (imbalanced)
        gain: Gain factor
    
    Returns:
        Dictionary with initialization summary
    """
    if init_method == 'generalized_imbalanced':
        initializer = ImbalancedInitializer(beta_input, beta_output, gain)
        
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                # Determine layer characteristics
                is_embedding = isinstance(module, torch.nn.Embedding)
                is_classifier = ('classifier' in name.lower() or 'output' in name.lower() or 
                               name.endswith('.classifier'))
                
                initializer.initialize_layer(module, name, is_embedding, is_classifier)
        
        return initializer.get_initialization_summary()
    
    else:
        # Standard initialization methods
        initialized_count = 0
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                if init_method == 'xavier':
                    xavier_init(module.weight, gain)
                elif init_method == 'he':
                    he_init(module.weight, gain)
                elif init_method == 'generalized':
                    generalized_init(module.weight, beta, gain)
                else:
                    raise ValueError(f"Unknown initialization method: {init_method}")
                
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.zero_()
                
                initialized_count += 1
        
        return {
            'method': init_method,
            'total_layers': initialized_count,
            'beta': beta if init_method == 'generalized' else None
        }
