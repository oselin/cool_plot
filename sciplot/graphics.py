#!/usr/bin/env python3

from PIL import Image
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import glob


def is_jupyter():
    try:
        # Check if the module is running in a Jupyter environment
        get_ipython()
        return True
    except NameError:
        return False

# Set Matplotlib backend (use TKAgg if Qt is missing or corrupted)
# But a to define backend if called from Jupyter Notebook
import matplotlib
if (not is_jupyter()):
    try:
        matplotlib.use('TKAgg')
    except:
        matplotlib.use("Agg")

class PlotData:
    def __init__(self, x, y, label, z=None, color='auto', linestyle='solid', alpha=1.0, linewidth=1.0, text=None):

        self.x = x
        self.y = y
        self.z = z
        self.label = label
        self.color = color
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.alpha = alpha
        self.text = text
        self.is_scalar = np.isscalar(x) and np.isscalar(y)

        if (not self.is_scalar and self.text is not None):
            raise ValueError("Cannot assign text to non scalar (scatter) plot")

    def to_numpy(self):

        if (isinstance(self.x, torch.Tensor)):
            if (self.x.is_cuda): self.x = self.x.cpu().detach().numpy()
            else: self.x = self.x.detach().numpy()

        if (isinstance(self.y, torch.Tensor)):
            if (self.y.is_cuda): self.y = self.y.cpu().detach().numpy()
            else: self.y = self.y.detach().numpy()

        if (isinstance(self.z, torch.Tensor)):
            if (self.z.is_cuda): self.z = self.z.cpu().detach().numpy()
            else: self.z = self.z.detach().numpy()

    def flatten(self):
        if (len(self.x.shape) > 1): 
            non_singleton_dims = sum(dim_size != 1 for dim_size in self.x.shape)
            if (non_singleton_dims == 1): self.x = self.x.flatten()
            else: raise ValueError("x seems to have multiple dimensions")
        if (len(self.y.shape) > 1): 
            non_singleton_dims = sum(dim_size != 1 for dim_size in self.y.shape)
            if (non_singleton_dims == 1): self.y = self.y.flatten()
            else: raise ValueError("y seems to have multiple dimensions")
        if (self.z is not None and len(self.z.shape) > 1): 
            non_singleton_dims = sum(dim_size != 1 for dim_size in self.z.shape)
            if (non_singleton_dims == 1): self.z = self.z.flatten()
            else: raise ValueError("z seems to have multiple dimensions")
    
    def resize(self):
        tmp = [len(d) for d in [self.x, self.y, self.z] if d is not None]

        new_size = np.min(tmp)

        self.x = self.x[:new_size]
        self.y = self.y[:new_size]
        if (self.z is not None): self.z = self.z[:new_size]


def cool_plot(*data:PlotData, 
              xlim=None, ylim=None, title=None,
              xlabel=None, ylabel=None, zlabel=None,
              width=15, height=6,
              save=None, filename=None, show_grid=True, show_legend=True, legend_args=None):
        
    # Convert data to numpy to avoid error
    for d in data: 
        d.to_numpy() 
        d.flatten()

    if (title is None): title = "Comparison between reference and estimated values"


    plt.figure(figsize=(width, height))
   
    # Data might not be with the same length
    for d in data: 
        if not d.is_scalar:
            d.resize()

    colors = cm.viridis(np.linspace(0, 1, 2*len(data)))
    for i, d in enumerate(data):
        if not d.is_scalar:
            plt.plot(d.x, d.y, label=d.label, color=(colors[2*i] if d.color=='auto' else d.color), alpha=d.alpha, linestyle=d.linestyle, linewidth=d.linewidth)
        else:
            plt.scatter(d.x, d.y, label=d.label, color=(colors[2*i] if d.color=='auto' else d.color), alpha=d.alpha, linestyle=d.linestyle, linewidth=d.linewidth)
            if (d.text is not None): 
                plt.annotate(d.text, (d.x, d.y), bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.3),)
    
    if (xlabel is not None): plt.xlabel(xlabel)
    if (ylabel is not None): plt.ylabel(ylabel)

    if (show_legend): 
        plt.legend(**legend_args)
    if (show_grid): plt.grid()
    plt.title(title)

    # Apply limits if provided
    if (xlim is not None): plt.xlim(xlim[0], xlim[1])
    if (ylim is not None): plt.ylim(ylim[0], ylim[1])

    if (save is not None):
        if (filename is None): raise ValueError("Error: filename was not provided")
        plt.savefig(save + 'plots/' + filename + '.png', bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        plt.close("all")
    else:
        plt.show()


def save_as_gif(directory, sub_folder, fps=5, loop=0, extension='.png', pattern=[], last=3000, logger=None, ):
    "Helper function for saving GIFs"
    
    plot_folder = directory + sub_folder + "\\*"

    # Get all test-specific plots
    files = glob.glob(plot_folder)

    if (not isinstance(pattern, list)): pattern = [pattern]
    for pat in pattern:
        
        # Filter images according to pattern (i.e. plot type)
        filtered_files = [file for file in files if (extension in file and pat in file)]

        # Open filtered images
        imgs = [Image.open(file) for file in filtered_files]
        
        # Export as gif
        imgs[0].save(fp=directory + f'{pat}.gif', format='GIF', append_images=imgs[1:], save_all=True, duration=int(last/fps), loop=loop)
        if (logger is not None): logger.info(f"Plot {pat[1:-1]} exported successfully as GIF")
