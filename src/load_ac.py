import numpy as np

# Outlier detection algorithm
def moving_median_sliding(arr: np.ndarray, window_size: int, axis = 1):
    """Compute moving median and MAD of an array along a specified axis

    Arguments:
        `arr`   :   np.ndarray
            Array over which to calculate sliding median and MAD
        `window_size`   :   int
            Odd-integer (total) window size over which to computer sliding values
        `axis`  :   int
            Axis number over which to compute sliding values
    """
    # Symmetric padding
    pad_width = window_size // 2
    
    # Pad the array on both sides (the padding will be done symmetrically)
    padded_arr = np.pad(arr, [(0, 0) if i != axis else (pad_width, pad_width) for i in range(arr.ndim)], mode='edge')
    
    # Apply sliding window view to the padded array
    windows = np.lib.stride_tricks.sliding_window_view(padded_arr, window_shape=(window_size,), axis=axis)
    
    # Compute the sliding median and MAD    
    median = np.nanmedian(windows, axis = -1)
    mad    = np.nanmedian(np.abs(windows - median[..., None]), axis=-1)

    return median, mad

def stack_antenna_ac(mir_data: object, antenna_num: int, rx_num: int, 
                     flagging = True, window_size = 5, 
                     mad_dev = 5.0, fill_val = np.nan, 
                     normalization = True, return_el = True, 
                     spw_baselining = True, num_good_points = 40,
                     num_ignore_edge_chans = 1024):
    """Code to preprocess autocorrelation data from a telescope

    Arguments:
        `mir_data`  : MirParser object
            pyuvdata MirParser instance containing the autocorrelation data
        `antenna_num`   :   int
            Antenna number from which data should be loaded
        `rx_num`    :   int
            Receiver number to load data from (nominally RxA = 0, RxB = 1)
        `flagging`  :   bool
            Option for median/MAD-based outlier flagging moving median and MAD values per spectral window
        `window_size`   :   int
            Total window size over which to compute median and MAD (for outlier flags)
        `mad_dev`   :   float
            Number of (absolute) MAD devations away from median to flag as outlier
        `fill_val`  :   float
            Value to fill flagged data points with
        `normalization` : bool
            Normalize autocorrelation per spectral window (for FFT subtraction later)
        `return_el` :   bool
            Return elevation of telescope per integration?
        `spw_baselining`    :   bool
            Remove linear trend per spectral window? Minimizes jumps between spectral windows in edge channels
        `num_good_points`   :   int
            Number of good (i.e., non-nan and non-inf) channels at beginning and end of spectral window to fit linear trend
        `num_ignore_edge_chans` :   int
            Number of edge channels at beginning and end of spectral window to ignore
    """
    #MAD window above will only work if window size is odd
    if window_size % 2 == 0 and flagging == True:
        print("Window size must be even for median window flagging to work!")
        print("Input window size was: ", window_size)
        print("New window size is: ", window_size + 1)

        window_size += 1

    #Ensure all data is loaded
    mir_data.reset()

    #Get number of SPWs in data; SPW 0 is pseudocontinuum
    num_spws = np.unique(mir_data.sp_data['corrchunk']).size - 1

    #Assume number of channels is uniform throughout spws
    num_chs    = mir_data.ac_data['nch'][0]
    chan_pos  = np.arange(-num_chs // 2, num_chs // 2)

    #Loop through each spw
    for spw in range(1, num_spws + 1):

        mir_data.select(
                        where=[
                                ("sb", "eq", "u"),
                                ("corrchunk", "eq", spw),
                                ("ant", "eq", antenna_num),
                                ("antrx", "eq", rx_num),
                            ],
                            reset=True)
        
        #Load the data, parse it
        mir_data.load_data()

        #Cut off edge channels
        freq_res = mir_data.ac_data['fres'][0]*1e-3
        channel_freq_offsets = chan_pos * freq_res

        data_stack = np.vstack([item['data'] for item in mir_data.auto_data.values()])[:, num_ignore_edge_chans:-num_ignore_edge_chans]
        f_sky      = mir_data.ac_data['fsky'][0] + channel_freq_offsets[num_ignore_edge_chans:-num_ignore_edge_chans]

        #Flag outliers before concatenation
        if flagging:
            med, mad = moving_median_sliding(data_stack, window_size, 1)

            out_of_bounds = np.logical_or(np.less_equal(data_stack, med - mad_dev * mad), 
                                          np.greater_equal(data_stack, med + mad_dev * mad))
            
            data_stack[out_of_bounds] = fill_val
        
        #Normalize -- what exactly is happening here?
        if normalization:
            data_stack = data_stack/np.repeat(np.nanmean(data_stack,axis=0).reshape([1,-1]), data_stack.shape[0], axis=0)
            data_stack = data_stack/np.repeat(np.nanmean(data_stack,axis=1).reshape([-1,1]), data_stack.shape[1], axis=1)

            data_stack -= np.nanmean(data_stack)

        #Subtract linear baseline per spectral window 
        if spw_baselining:
            channel_numbers = np.arange(f_sky.size)

            for integ in range(data_stack.shape[0]):
                is_finite = np.isfinite(data_stack[integ, :])
                good_channels = np.append(channel_numbers[is_finite][:num_good_points], channel_numbers[is_finite][-num_good_points:])

                baseline_params = np.poly1d( np.polyfit(good_channels, data_stack[integ, good_channels], 1) )
                data_stack[integ, :] -= baseline_params(channel_numbers)
            
        #Concatenate processed SPW ACs together
        if spw == 1:
            stacked = 1 * data_stack
            freqs = 1 * f_sky
            spw_num = spw * np.ones(f_sky.shape)

            if return_el:
                elevation = mir_data.eng_data['actual_el']

        else:
            stacked = np.hstack((stacked, data_stack))
            freqs = np.hstack((freqs, f_sky))
            spw_num = np.hstack((spw_num, spw * np.ones(f_sky.shape)))

        #Get ready for the next spw
        mir_data.reset()

    #Return elevation if needed
    if return_el:
        return freqs, stacked, spw_num, elevation
    return freqs, stacked, spw_num