"""
Description goes here. TODO

__author__  = "Pierfrancesco Oselin"
__credits__ = ["Pierfrancesco Oselin"]
__version__ = "1.0.0"
"""
from .data import Data
import os

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
except:
    raise ImportError("Matplotlib library is needed for cool_plot")

try:
    import numpy as np
except:
    raise ImportError("Numpy library is needed for cool_plot library")

def cool_plot(*data:Data, 
    xlim=None, ylim=None, title=None,
    xlabel=None, ylabel=None,
    width=15, height=6,
    filename=None,
    show_grid=True,
    use_latex=True,
    ):
        
    # Convert data to numpy arrays to avoid error
    for d in data: 
        d.flatten()

    if (title is None): title = "Comparison between reference and estimated values"

    # Create figure obect
    plt.figure(figsize=(width, height))
   
    # Data might not be with the same length
    for d in data: d.resize()

    colors = cm.viridis(np.linspace(0, 1, 2*len(data)))
    for i, d in enumerate(data):
        plt.plot(d.x, d.y, label=d.label, color=(colors[2*i] if d.color=='auto' else d.color), alpha=d.alpha, linestyle=d.linestyle)
    
    # Define axis labels
    if (xlabel is not None): plt.xlabel(xlabel)
    if (ylabel is not None): plt.ylabel(ylabel)

    # Set plot legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))

    # Enable grid
    if (show_grid): plt.grid()

    # Set title
    plt.title(title)

    # Apply limits if provided
    if (xlim is not None): plt.xlim(xlim[0], xlim[1])
    if (ylim is not None): plt.ylim(ylim[0], ylim[1])

    if (filename is not None):
        plt.savefig(filename + '.png', bbox_inches='tight', pad_inches=0.1, dpi=100, facecolor="white")
        plt.close("all")
    else:
        plt.show()