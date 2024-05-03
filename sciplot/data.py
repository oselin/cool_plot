"""
Description goes here. TODO

__author__  = "Pierfrancesco Oselin"
__credits__ = ["Pierfrancesco Oselin"]
__version__ = "1.0.0"
"""
try:
    import numpy as np
except:
    raise ImportError("Numpy library is needed for cool_plot library")

try:
    import torch
    torch_is_installed = True
except:
    torch_is_installed = False

class Data:
    def __init__(self, x:list|np.ndarray, y:list|np.ndarray, label:str, z:list|np.ndarray|None=None, color='auto', linestyle='solid', alpha:float=1.0, ):

        self.x         = np.array(x) if (isinstance(x, list)) else x
        self.y         = np.array(y) if (isinstance(y, list)) else y
        self.z         = np.array(z) if (isinstance(z, list)) else z
        self.label     = label
        self.color     = color
        self.linestyle = linestyle
        self.alpha     = alpha if (alpha <= 1.0) else 1.0

        # Handle tensors
        if (torch_is_installed):
            if (isinstance(self.x, torch.Tensor)):
                if (self.x.is_cuda): # if tensor is on GPU, move to CPU and remove gradient
                    self.x = self.x.cpu()
                self.x = self.x.detach().numpy()

            if (isinstance(self.y, torch.Tensor)):
                if (self.y.is_cuda): # if tensor is on GPU, move to CPU and remove gradient
                    self.y = self.y.cpu()
                self.y = self.y.detach().numpy()

            if (isinstance(self.z, torch.Tensor)):
                if (self.z.is_cuda): # if tensor is on GPU, move to CPU and remove gradient
                    self.z = self.z.cpu()
                self.z = self.z.detach().numpy()

    def flatten(self, ):
        if (len(self.x.shape) == 2 and 1 in self.x.shape): self.x = self.x.flatten()
        if (len(self.y.shape) == 2 and 1 in self.y.shape): self.y = self.y.flatten()
        if (self.z is not None and len(self.z.shape) == 2 and 1 in self.z.shape): self.z = self.z.flatten()
    
    def resize(self):
        
        tmp = [len(d) for d in [self.x, self.y, self.z] if d is not None]
        new_size = np.min(tmp)

        self.x = self.x[:new_size]
        self.y = self.y[:new_size]
        if (self.z is not None): self.z = self.z[:new_size]