#!/usr/bin/env python3

import numpy as np
import torch, scipy, logging, os
import matplotlib.pyplot as plt
import fmpy
import torch

def create_dir(folder_base_name, folder_sub_name=None, job_id=None, create_plot_folder=True, create_param_folder=True):
    
    folder_number, folder_name = 1, ""
    if not os.path.exists(folder_base_name): os.makedirs(folder_base_name)

    if (folder_base_name[-1] != '/'): folder_base_name += '/'

    if (folder_sub_name is None and job_id is None): raise ValueError("folder_sub_name or job_id must be provided!")
    
    while True:
        if (folder_sub_name is not None):
            folder_name = f"{folder_base_name + folder_sub_name}{folder_number:02d}/"
            folder_number += 1
        elif (job_id is not None):
            folder_name = f"{folder_base_name}{job_id}/"

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            if (create_plot_folder): os.makedirs(folder_name + "plots")
            if (create_param_folder): os.makedirs(folder_name + "params")
            
            print(f"Created folder: {folder_name} and its structure.")
            break      
    
    return folder_name


# Data acquisition time span (for integration)
def time(data, time_interval):
    assert type(time_interval) is float

    if (isinstance(data, torch.Tensor)): 
        return torch.tensor(np.linspace(0, time_interval*data.size()[0], data.size()[0], endpoint=False))
    else:    
        return np.linspace(0, time_interval*data.size, data.size, endpoint=False)

# File name template
def filename(index): 
    return f"ABA_journey_{index}.csv"


# Find index that provide the nearest value
def find_nearest(array, value, end=False):
    idx = (np.abs(array - value)).argmin()

    if (end): idx += 1
    return idx


def adjust_freq(cutoff_freq, cutoff_mode, avg_speed, sensitivity=0.01):

    if (type(cutoff_freq) is list):
        if (cutoff_mode == 'auto'):
            max_freq = np.abs(avg_speed)/sensitivity
            cutoff_freq[1] = max_freq
    
    return cutoff_freq


def fft_to_plot(data, T, plot=False):
    """
    data: signal in time domain
    T: time period, needed for computing the frequencies span
    """
    # Reshape to monodimensional array if needed
    data = data.reshape(-1)

    # Compute fft
    data_fft = scipy.fft.fft(data)

    # Find the index at which negative frequencies start
    n  = int(len(data)/2 if not (len(data)%2) else (len(data)+1)/2)

    # Get frequencies to be used as x-axis
    xf  = scipy.fft.fftfreq(len(data), T)[:n]

    # Compute compensated frequ. modules (for plotting)
    compensated_module = 2.0/n * np.abs(data_fft[0:n])

    if (plot):
        plt.figure()
        plt.plot(xf, compensated_module)
        plt.show()

    return xf, compensated_module


def get_logger(logpath, filepath=None, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)

    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    if (filepath is not None):
        logger.info(filepath)
        with open(filepath, "r") as f:
            logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def get_model(model_name):
    from .model import model_trnf2_inv, initial_values
    # Choose model
    # if (model_name == 'true1'):
    #     model = model_trnf1
    # elif (model_name == 'true2'):
    #     model = model_trnf2
    # elif (model_name == '25params'):
    #     model = model_trnf_25
    # elif (model_name == '6params'):
    #     model = model_trnf_6
    # elif (model_name == 'inverse1'):
    #     model = model_trnf1_inv
    # elif (model_name == 'inverse2'):
    #     model = model_trnf2_inv
    # elif (model_name == 'simple'):
    #     model = simple_model
    # elif (model_name == 'simple_inverse'):
    #     model = simple_model_inv
    # else: model = model_trnf2
    model = model_trnf2_inv
    
    initial_params = initial_values(output=model_name, params='true')

    return initial_params, model


def mse(y, y_hat):
    # If both the inputs are tensors, the output has to be a tensor with gradient enabled
    if (isinstance(y, torch.Tensor) and isinstance(y_hat, torch.Tensor)):
        return torch.mean((torch.squeeze(y) - torch.squeeze(y_hat))**2)
    else:
        # If one of the two is tensor, and the other one no, return a numpy
        if (isinstance(y, torch.Tensor)): y = y.detach().numpy()
        if (isinstance(y_hat, torch.Tensor)): y_hat = y_hat.detach().numpy()

        # Avoid wrong calculation in case of multi-dimensional array
        if (len(y.shape) == 2): y = y.reshape(-1)
        if (len(y_hat.shape) == 2): y_hat = y_hat.reshape(-1)
        mse = np.mean((y - y_hat)**2)
        return mse

def load_params(filename):
    optimized_params = []
    # Load optimized parameters from file
    with open(filename, 'r') as file:
        for line in file:
            optimized_params.append(float(line.strip()))

    return np.array(optimized_params)


def load_history(filename):
    optimized_params = []
    # Load optimized parameters from file
    with open(filename, 'r') as file:
        for line in file:
            optimized_params.append([float(i) for i in line.split()])

    return np.array(optimized_params)


def read_fmu_params(filepath):
    fmu = fmpy.read_model_description(filepath)

    output = f'{"Variable name":<60} {"Causality":<20} {"Type":<20} {"Value":<20}\n'
    for variable in fmu.modelVariables:
        if variable.causality == 'parameter':                                   
            output += f'{variable.name:<60} {variable.causality:<20} {variable.type:<20} {variable.start:<20}\n'
    
    return output


def conv_torch(u:torch.Tensor, g:torch.Tensor, t):
    """
    Manual implementation of the convolution operator to support torch tensors
    - u: input
    - g: kernel
    """
    if not (isinstance(g, list) or isinstance(g, tuple)): g = [g]
    out = []
    for kernel in g:
        # Pre-allocate memory for performances
        yt = torch.zeros_like(t)
        for i in range(len(t)):
            # Time span on which evalute G and integrate. Note: tau \in [0, t]
            tau = np.arange(0, i)
            tmp = u[tau]*kernel[i - tau]
            yt[i] = torch.trapz(tmp, t[0:i])
        out.append(yt)
    
    return out

def get_magnitude_torch(tensor):
    if torch.equal(tensor, torch.zeros_like(tensor)):
        return 0  # Magnitude of a zero tensor is 0

    magnitude = int(torch.floor(torch.log10(tensor.abs())).item())
    return magnitude

def grad(f, x, h=None):
    """
    Return the gradient, as list of partial derivatives
    computed via (centered) finite differences
    """

    grads, hs = [], []
    if (h is None): # h is set to auto

        for x_i in x:
            # NOTE: for parameters with magnitude of 1e4, h=1e-3 is demonstrated to be significant
            # get the magnitude of the parameter
            coeff = get_magnitude_torch(x_i) * np.sqrt(np.finfo(float).eps) # -6
            hs.append(float(10**coeff))
    else:
        hs = [h for _ in x]

    for i in range(len(x)):
        # NOTE: the copy of x will be with requires_gradient=False
        x_p = x.clone()
        x_m = x.clone()

        x_p[i] += hs[i]
        x_m[i] -= hs[i]

        dfdi = (f(x_p) - f(x_m))/(2*hs[i])
        grads.append(dfdi)

    return grads


def normalize(vector:torch.Tensor)->torch.Tensor:
     return vector/(10**torch.ceil(torch.log10(torch.abs(vector))))


def min_max_normalization(vector:torch.Tensor)->torch.Tensor:

    min, max = torch.min(vector), torch.max(vector)
    v = (vector - min)/(max - min)

    return v


def find_duplicates(lst:list):
    # Create a dictionary to count occurrences
    count_dict = {}

    # Count each string
    for item in lst:
        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1

    # Find strings that appear more than once
    duplicates = [item for item, count in count_dict.items() if count > 1]

    # Display duplicates
    return duplicates


def remove_duplicates(lst:list):

    dups = find_duplicates(lst)

    for elem in dups:
        counter = 1

        for i in range(len(lst)):
            if lst[i] == elem:
                lst[i] += f"_{counter}"
                counter += 1
    
    return lst