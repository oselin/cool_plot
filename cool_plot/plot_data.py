"""
Description goes here. TODO

__author__  = "Pierfrancesco Oselin"
__credits__ = ["Pierfrancesco Oselin"]
__version__ = "1.0.0"
"""
import numpy as np

try:
    import torch
    torch_is_installed = True
except:
    torch_is_installed = False

class PlotData:
    def __init__(self, x:list|np.ndarray, y:list|np.ndarray, label:str, z:list|np.ndarray|None=None, color='auto', linestyle='solid', alpha:float=1.0, ):

        self.x         = x if (isinstance(x, np.ndarray)) else np.array(x)
        self.y         = y if (isinstance(y, np.ndarray)) else np.array(y)
        self.z         = z if (isinstance(z, np.ndarray) or z is None) else np.array(z)
        self.label     = label
        self.color     = color
        self.linestyle = linestyle
        self.alpha     = alpha if (alpha <= 1.0) else 1.0

    def to_numpy(self, ):

        if (torch_is_installed):
            if (isinstance(self.x, torch.Tensor)):
                if (self.x.is_cuda): self.x = self.x.cpu().detach().numpy()
                else: self.x = self.x.detach().numpy()

            if (isinstance(self.y, torch.Tensor)):
                if (self.y.is_cuda): self.y = self.y.cpu().detach().numpy()
                else: self.y = self.y.detach().numpy()

            if (isinstance(self.z, torch.Tensor)):
                if (self.z.is_cuda): self.z = self.z.cpu().detach().numpy()
                else: self.z = self.z.detach().numpy()

    def flatten(self, ):
        if (len(self.x.shape) == 2): self.x = self.x.flatten()
        if (len(self.y.shape) == 2): self.y = self.y.flatten()
        if (self.z is not None and len(self.z.shape) == 2): self.z = self.z.flatten()
    
    def resize(self):
        
        tmp = [len(d) for d in [self.x, self.y, self.z] if d is not None]
        new_size = np.min(tmp)

        self.x = self.x[:new_size]
        self.y = self.y[:new_size]
        if (self.z is not None): self.z = self.z[:new_size]