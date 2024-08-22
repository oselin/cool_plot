#!/usr/bin/env python3

import numpy as np
import pandas as pd
from .utils import filename, adjust_freq, find_nearest, time
import yaml, scipy
from scipy.signal import butter

def butter_filter(data, cutoff_freq, fs, order=None, filter_type='lowpass', analog=False, filter_twice=True):

    if (order is None): order = 4

    # Normalize the cutoff frequency
    if (type(cutoff_freq) is list): 
        cutoff_freq = np.array(cutoff_freq, dtype='float64')
    cutoff_freq /= (0.5*fs) 

    # Generate Butterworth coefficients
    b, a = butter(order, cutoff_freq, btype=filter_type, analog=analog, output='ba', fs=None)

    # Filter data
    data = scipy.signal.lfilter(b, a, data)
    
    # Filter again on reverse ordered array to compensate the shift
    if (filter_twice): data = scipy.signal.lfilter(b, a, data[::-1])[::-1]
        
    return data


def integrate(x, time_period, initial_value=None):

    t = time(x, time_period)
    if (initial_value is not None):
        return scipy.integrate.cumtrapz(x, t, initial=initial_value)
    else:
        return scipy.integrate.cumtrapz(x, t)


def find_shift(data1_x, data1_y, data2_x, data2_y):
    """
    Find relative shift between two time-series data
    NOTE: data1 is the data to be shifted
    """
    assert type(data1_x) == np.ndarray
    assert type(data1_y) == np.ndarray
    assert type(data2_x) == np.ndarray
    assert type(data2_y) == np.ndarray

    # Compute the cross-covariance
    interpolated_fcn = scipy.interpolate.interp1d(data2_x, data2_y, fill_value='extrapolate')

    cross_covariance = np.correlate(data1_y - np.mean(data1_y), interpolated_fcn(data1_x) - np.mean(interpolated_fcn(data1_x)), mode='full')

    lags = np.arange(-len(data1_x) + 1, len(data1_x))

    dx = data1_x[1] - data1_x[0]

    # Find the lag with the maximum cross-covariance
    shift = lags[np.argmax(cross_covariance)]*dx

    return shift


def filtered_kinematic(acc:np.ndarray, Fs, FILTER_TYPE, CUTOFF_FREQ, FILTER_ORDER, filter_pos=True):
    """
    Estimate filtered acceleration, velocity and position given raw acceleration data
    and filter configuration parameters.
    Input: [np.ndarray] raw acceleration data
    Output: 3x[np.ndarray] filtered acceleration, velocity and position
    """

    T = 1/Fs

    # Filter acceleration signal with a  FILTER filter to remove unwanted frequencies
    acc_filt = butter_filter(acc, filter_type=FILTER_TYPE, cutoff_freq=CUTOFF_FREQ, order=FILTER_ORDER, fs=Fs)
    
    # Estimate velocity from acceleration
    vel      = integrate(acc_filt, T, initial_value=0)
    vel_filt = butter_filter(vel, filter_type=FILTER_TYPE, cutoff_freq=CUTOFF_FREQ, order=FILTER_ORDER, fs=Fs)

    # Estimate displacement from velocity
    pos      = integrate(vel_filt, T, initial_value = 0)
    
    # Handle filter options for position data
    if (filter_pos):
        if (FILTER_TYPE == 'bandpass'):
            pos_filt = butter_filter(pos, filter_type='highpass', cutoff_freq=CUTOFF_FREQ[0], order=FILTER_ORDER, fs=Fs)
        elif (FILTER_TYPE == 'highpass'):
            pos_filt = butter_filter(pos, filter_type='highpass', cutoff_freq=CUTOFF_FREQ, order=FILTER_ORDER, fs=Fs)
        else: # either low pass or stop band 
            pos_filt = pos.copy() # do not filter
    else: 
        pos_filt = pos.copy() # do not filter

    return acc_filt, vel_filt, pos_filt


def load_data(file_index, config_file, resampled_freq=None, resample_method='downsample', output="wavelet", normalize=True):

    # Verify that output mode is correct
    if (output not in ['wavelet', 'acceleration']): raise ValueError("Unkown output value provided")
    if (resample_method not in ['downsample', 'filter']): raise ValueError("Unkown resample method value provided")
    
    # Load project parameters
    with open(config_file, 'r') as f: 
        config = yaml.safe_load(f)

    DATA_ORIGIN         = config["data"]["data_origin"]
    FILEPATH            = config["data"]["file_path"]
    REFERENCE_FILE_NAME = config["data"]["reference_file_name"]

    # Filter settings
    filter_type         = config["data"]["filter_type"]
    filter_order        = config["data"]["filter_order"]
    cutoff_freq         = config["data"]["cutoff_freq"]
    cutoff_mode         = config["data"]["cutoff_mode"]

    # Settings for Continuous Wavelet Transform (CWT)
    num_widths          = config["data"]["num_widths"]

    # Settings for interpolation
    interp_method       = config["data"]["interpolation_method"]

    Fs                  = config["data"]["sampling_freq"]

    ## Load reference values
    rmf            = pd.read_csv(FILEPATH + REFERENCE_FILE_NAME)
    rmf_track_pos  = rmf.dist_on_track.values
    rmf_z_displ    = rmf.Height_left.values

    # Vertical reference displacement is provided in mm. Convert to meters
    rmf_z_displ = rmf_z_displ/1000

    # Compute the sampling period
    Ts = 1/Fs

    # file_index is assumed to be a list of files to be read. If not, convert file_index (a number) into a list
    if (not isinstance(file_index, list)): file_index = [file_index]

    # Initialize data structure to store the i-th imported values
    track_pos, output_data, reference, long_vel = [], [], [], []

    # Iterate over the list of file to be read and apply all the importing steps, illustrated in ./01.data_filtering.ipynb
    for i, file_idx in enumerate(file_index):

        ## Load file and get only the rows with actual values
        df = pd.read_csv(FILEPATH + filename(file_idx))
        indexes = df['distance_on_track'].notna().values

        ## Load axle-box acceleration, vehicle speed and position on the track
        AB_track_pos_i = df['distance_on_track'][indexes].to_numpy()
        AB_vel_i       = df['speed'][indexes].to_numpy()
        ABA_z_i        = df[DATA_ORIGIN][indexes].to_numpy()

        # NOTE: the locomotive might have traveled backword. If so, invert the data
        # Before filtering, check if reference is decreasing
        is_decreasing = np.all(np.diff(AB_track_pos_i) < 0)

        if (is_decreasing):
            AB_track_pos_i = AB_track_pos_i[::-1]
            AB_vel_i       = AB_vel_i[::-1]*-1
            ABA_z_i        = ABA_z_i[::-1]
        
        
        # Handle the cutoff mode (if set to auto, filter is modify to accomodate train speed)
        cutoff_freq_i = adjust_freq(cutoff_freq, cutoff_mode, AB_vel_i.mean())

        ## Get filtered acceleration, velocity and position
        ABA_z_filt_i, ABV_z_filt_i, ABP_z_filt_i = filtered_kinematic(ABA_z_i, FILTER_TYPE=filter_type, 
                        CUTOFF_FREQ=cutoff_freq_i, FILTER_ORDER=filter_order, Fs=Fs)

        ## Compute possible shift between reference and measurement signals and apply it
        shift = find_shift(rmf_track_pos, rmf_z_displ, AB_track_pos_i, ABP_z_filt_i)
        rmf_track_pos_i = (rmf_track_pos - shift).copy()
        
        ## Match the reference track span with the loaded df data
        AB_track_min, AB_track_max = find_nearest(AB_track_pos_i, rmf_track_pos_i[0]), find_nearest(AB_track_pos_i, rmf_track_pos_i[-1], end=True)
        
        # Since we have to match the input data sampling rate
        # Filter the reference signal with the same kind of filter, before interpolating
        rmf_z_displ_i = butter_filter(rmf_z_displ, filter_type=filter_type, cutoff_freq=cutoff_freq_i, order=filter_order, fs=Fs)
        
        # Interpolation of the reference data
        rmf_z_interpolation_i = scipy.interpolate.interp1d(rmf_track_pos_i, rmf_z_displ_i, fill_value="extrapolate", kind=interp_method)
        rmf_z_interp_i        = rmf_z_interpolation_i(AB_track_pos_i).copy()
        
        ## Redefinition of the dataset
        # Limit the loaded data and the interpolated measurement data in the span of known reference
        AB_track_pos_i = AB_track_pos_i[AB_track_min:AB_track_max]
        ABA_z_filt_i   = ABA_z_filt_i[AB_track_min:AB_track_max]
        rmf_z_interp_i = rmf_z_interp_i[AB_track_min:AB_track_max]
        AB_vel_i       = AB_vel_i[AB_track_min:AB_track_max]
        
        # Resample data to lower frequencies (i.e. downsample)
        if (resampled_freq):

            if (resample_method == 'downsample'):
                reduction_factor = int(Fs/resampled_freq)

                # Compute the new sampling period
                Ts = reduction_factor/Fs

                AB_track_pos_i = AB_track_pos_i[::reduction_factor]
                AB_vel_i       = AB_vel_i[::reduction_factor]
                ABA_z_filt_i   = scipy.signal.decimate(ABA_z_filt_i.copy(),   reduction_factor, ftype='fir', zero_phase=True, axis=0)
                rmf_z_interp_i = scipy.signal.decimate(rmf_z_interp_i.copy(), reduction_factor, ftype='fir', zero_phase=True, axis=0)
            else:
                ABA_z_filt_i =  butter_filter(ABA_z_filt_i, filter_type='lowpass', cutoff_freq=resampled_freq, order=5, fs=Fs)
        
        if (output == 'wavelet'):
            ## Define scales for the wavelet transform
            # The scales control the width of the wavelet and the range of frequencies to analyze.
            widths = np.arange(1, num_widths + 1) 

            ## Perform the Continuous Wavelet Transform
            ABA_cwt_i = scipy.signal.cwt(ABA_z_filt_i, scipy.signal.morlet2, widths)

            ## Get the amplitudes
            ABA_cwt_i = np.abs(ABA_cwt_i).T
            
            ## Calculate the total energy of the signal
            # This will be used as normalization factor
            E = np.sum(np.square(np.abs(ABA_z_filt_i)))/ABA_z_filt_i.size
            if (normalize): ABA_cwt_i = ABA_cwt_i/E

            output_data.append(ABA_cwt_i)
        else:
            output_data.append(ABA_z_filt_i)

        track_pos.append(AB_track_pos_i)
        reference.append(rmf_z_interp_i)
        long_vel.append(AB_vel_i)

     
    # Convert data to proper shape
    track_pos = np.hstack(track_pos)
    reference = np.hstack(reference)
    long_vel  = np.hstack(long_vel)

    if (output == 'wavelet'):
        output_data = np.vstack(output_data)
    else:
        output_data = np.hstack(output_data)

    return track_pos, output_data, reference, Ts, long_vel