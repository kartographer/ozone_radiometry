import numpy as np
import scipy.linalg as linalg

def remove_standing_waves(stacked_data: np.ndarray, sky_freqs : np.ndarray, 
                         f0_s : np.ndarray, n_harmonics : int, freq_order : int, 
                         baseline_order : int, fft_window : float = 0.05 / 63.2, 
                         fft_resolution_multiplier : int = 3):
    """Fourier-based function to remove standing wave contribution from autocorrelation data
       and simulatneously fit a per-spw baseline to the data

    Arguments:
        `stacked_data`  :   np.ndarray
            3D autocorrelation data, axes are time x spw x channel
        `sky_freqs`     :   np.ndarray
            2D array of on-sky frequencies with axes spw x channel
        `f0_s`  :   np.ndarray or list
            Fundamental spatial frequencies for standing waves to find in FFT
        `n_harmonics`   :   int
            Number of harmonics to search in FFT for each fundamental frequency above
        `freq_order`    :   int
            Polynomial order to expand standing wave contribution in least-squares modeling
        `baseline_order`    :   int
            Polynomial order to fit for spectral baseline per spw
        `fft_window`    :   float
            Size of window around f0 * harmonics to search for maximal FFT peak
        `fft_resolution_multiplier` :   int
            Per-spw data are automatically padded to the next highest power of 2; this argument increases FFT spectral
            resolution by additional factors of 2
    """
    # Make local copy of data to work with
    data = 1 * stacked_data
    
    # Ensure all values inside FFT are real
    nan_flag = np.isnan(data) | np.isinf(data)
    data[nan_flag] = 0.0        

    # Make a list of frequencies to add in fitting
    f0_list = np.array([])

    for f0 in f0_s:
        f0_list = np.append(f0_list, np.arange(1, n_harmonics + 1) * f0)

    # Padding with zeros to bump up spectral resolution
    highest_n = int(np.ceil(np.log2(data.shape[2]))) + fft_resolution_multiplier

    left_pad = int(np.ceil(((2**highest_n) - data.shape[-1]) * 0.5))
    right_pad = int(np.floor(((2**highest_n) - data.shape[-1]) * 0.5))
    pad_slice = slice(left_pad, -right_pad)
                          
    # Stores *actual* FFT frequencies for model below
    fftfreqs = np.fft.fftfreq(int(2**highest_n), d = 1e3 * np.abs(np.diff(sky_freqs).flatten()[0]))

    # For convenience later
    n_integ, n_spw, n_chan = data.shape[0], data.shape[1], data.shape[2]

    # Stores padded sky frequencies for polynominal expansion
    channel_steps = np.arange(-left_pad, 2**highest_n - right_pad)
    channel_holds = np.ones((1, 2**highest_n))
    freqs_pad = sky_freqs[0, 0] + channel_holds * np.diff(sky_freqs[0, :]).flatten()[0] * channel_steps

    for spw in range(1, n_spw):
        freqs_pad = np.vstack((freqs_pad, sky_freqs[spw, 0] + channel_holds * np.diff(sky_freqs[spw, :]).flatten()[0] * channel_steps))

    # Normalized sky frequencies for faster convergence later
    f_sky_norm = 2 * (freqs_pad - np.min(freqs_pad[:, pad_slice])) / (np.max(freqs_pad[:, pad_slice]) - np.min(freqs_pad[:, pad_slice])) - 1
    f_sky_norm_real = f_sky_norm[:, left_pad : n_chan + left_pad]

    # Best-fit model and cleaned spectrum arrays
    model, cleaned = np.zeros(stacked_data.shape), np.zeros(stacked_data.shape)

    # Stores basis for LSQ
    # n_spw * n_chan (axis 0): total number of channels in fit
    # (baseline_order + 1) * n_spw + 2 * (freq_order + 1): per-spw baselining terms + standing wave coefficients
    A = np.zeros((n_spw * n_chan, (baseline_order + 1) * n_spw + 2 * (freq_order + 1) * len(f0_list)))
    
    # Build corner of A for independent spw-baseline fitting
    # Integration independent! So do it once only to speed up
    for spw in range(n_spw):
        for n in range(baseline_order + 1):
            A[spw * n_chan : (spw + 1) * n_chan, (baseline_order + 1) * spw + n] = 1 * f_sky_norm_real[spw, :]**n
    
    # Go through each integration in the cube
    for integ in range(n_integ):
        padded_integ = np.copy(data[integ, :])

        # Remove mean DC-baseline from each spw
        for i in range(n_spw):
            padded_integ[i, :] -= np.nanmean(padded_integ[i, :])
        
        # Data padding 
        # (0, 0) --> no padding along spw axis
        # (left_pad, right_pad) --> pad along channel axis
        padded_integ = np.pad(padded_integ, ((0, 0), (left_pad, right_pad)))

        # Padded FFT along spectral axis only
        padded_fft = np.fft.fft(padded_integ, axis = 1)
        
        # Sum powers in all spws, keeping channels untouched
        summed_psd = np.sum(np.abs(padded_fft)**2.0, axis = 0)
        
        # Add standing wave terms into A for fitting
        for idx, f0 in enumerate(f0_list):  
            model_fft = np.zeros(padded_fft.shape, dtype='complex128')
            
            #Make window around |f0 * (1 + harmonic)| to search for peaks using a priori information
            selection_window = (fftfreqs >= (f0 - fft_window)) & (fftfreqs <= (f0 + fft_window))
            peak_amp = np.nanmax(summed_psd[selection_window])
            peak_amp_position = np.where((summed_psd == peak_amp) & selection_window)[0][0]

            #Assume peak in each SPW is same spatial frequency but with different amplitudes
            for spw in range(n_spw):
                model_fft[spw, [peak_amp_position, -peak_amp_position]] = padded_fft[spw, [peak_amp_position, -peak_amp_position]]

            #Get standing wave pattern
            model_re = np.fft.ifft(model_fft.real, axis = 1).real[:, left_pad : n_chan + left_pad]
            model_im = np.fft.ifft(1j * model_fft.imag, axis = 1).real[:, left_pad : n_chan + left_pad]
            
            #Add terms into basis matrix
            for n in range(freq_order + 1):
                A[:, (baseline_order + 1) * n_spw + 2 * idx * (freq_order + 1) + 2 * n]     = model_im.flatten() * f_sky_norm_real.flatten()**n
                A[:, (baseline_order + 1) * n_spw + 2 * idx * (freq_order + 1) + 2 * n + 1] = model_re.flatten() * f_sky_norm_real.flatten()**n
                
        #Fit the LSQ
        res = linalg.lstsq(A, data[integ, :].flatten())[0]

        #Store model and cleaned spectrum
        model[integ, :] = np.matmul(A, res).reshape(data[integ, :].shape)
        cleaned[integ, :] = data[integ, :] - model[integ, :]

    #Reflag bad data
    model[nan_flag] = np.nan
    cleaned[nan_flag] = np.nan

    #Return model and cleaned spectral cubes
    return model, cleaned