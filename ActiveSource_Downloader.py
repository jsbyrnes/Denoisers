import os
import numpy as np
import h5py
import csv
import sys
import math
import argparse
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.signal.trigger import aic_simple
import time
import matplotlib.pyplot as plt
import re
from datetime import datetime
from scipy.optimize import minimize
from numba import jit
from scipy.ndimage import gaussian_filter1d

# Hardwired flags
PLOT_EVERYTHING = False
PLOT_ABOVE_THRESHOLD = False
THRESHOLD_GAIN_DB = 15  # Threshold for plotting based on gain in dB

@jit(nopython=True)
def piecewise_model_with_exp(params, t, window_data, sample_rate, min_t1):
    amp1, amp2, decay, floor, t1, t2 = params

    # Convert times to indices to enforce the sample separation
    t1_idx = int(t1 * sample_rate)
    t2_idx = int(t2 * sample_rate)

    # Large constant for penalizing invalid configurations
    large_constant = 1e12

    # Ensure the transition times are in order and the constraints on t1 and t2
    if not (min_t1 <= t1 < t2) or (t2_idx - t1_idx < 2) or (t2_idx > (window_data.size - 2)):
        return np.full_like(window_data, large_constant)  # Penalize invalid configurations

    if amp2 < amp1:  # Positive slope required between amp1 and amp2
        return np.full_like(window_data, large_constant)  # Penalize invalid slopes

    if decay < 0:
        return np.full_like(window_data, large_constant)  # Penalize invalid decay

    if floor > amp2:
        return np.full_like(window_data, large_constant)  # Penalize invalid floor

    model = np.zeros_like(t)  # Replace np.empty_like with np.zeros_like for numba compatibility

    # Build the piecewise model manually
    for i in range(len(t)):
        if t[i] < t1:
            model[i] = amp1  # Interval 1 (flat at amp1)
        elif t1 <= t[i] < t2:
            model[i] = amp1 + (amp2 - amp1) * (t[i] - t1) / (t2 - t1)  # Interval 2 (positive slope)
        else:
            # Interval 3: Exponential decay starting from amp2 and decaying toward floor
            delta_t = t[i] - t2
            model[i] = (amp2 - floor) * np.exp(-delta_t / decay) + floor

    # Return the residuals (difference between the model and the data)
    residuals = model - window_data
    return residuals

@jit(nopython=True)
def jacobian_with_exp(params, t, sample_rate):
    amp1, amp2, decay, floor, t1, t2 = params

    # Preallocate the Jacobian matrix (6 parameters in total)
    J = np.zeros((len(t), 6))

    # Compute slope for the second linear interval
    slope2 = (amp2 - amp1) / (t2 - t1) if t2 != t1 else 0  # Ensure no division by infinity

    # Populate the Jacobian matrix based on the piecewise conditions
    for i in range(len(t)):
        if t[i] < t1:
            J[i, 0] = 1  # derivative w.r.t amp1
        elif t1 <= t[i] < t2:
            # Linear interval between t1 and t2
            delta_t = t[i] - t1
            J[i, 0] = 1 - delta_t / (t2 - t1)  # derivative w.r.t amp1
            J[i, 1] = delta_t / (t2 - t1)      # derivative w.r.t amp2
            J[i, 4] = -slope2 + (amp2 - amp1) * delta_t / ((t2 - t1) ** 2)  # derivative w.r.t t1
            J[i, 5] = -(amp2 - amp1) * delta_t / ((t2 - t1) ** 2)           # derivative w.r.t t2
        else:
            # Exponential decay with a floor
            delta_t = t[i] - t2
            exp_term = np.exp(-delta_t / decay)

            J[i, 1] = exp_term  # derivative w.r.t amp2
            J[i, 2] = (amp2 - floor) * delta_t * exp_term / (decay ** 2)  # derivative w.r.t decay
            J[i, 3] = 1 - exp_term  # derivative w.r.t floor
            J[i, 5] = -(amp2 - floor) * exp_term / decay  # derivative w.r.t t2

    return J

def gauss_newton_lm(window_data, sample_rate, max_iter=50, damping=1, tol=1e-2):
    """
    Perform Gauss-Newton optimization with Levenberg-Marquardt damping to fit the slope function with exponential decay in the last interval.
    """
    t = np.linspace(0, len(window_data) / sample_rate, len(window_data))

    # Minimum t1 should be at least 1/5 of the total window length
    min_t1 = (1 / 4) * (len(window_data) / sample_rate)

    # Initial guess for the parameters (including exponential decay parameters)
    params = np.array([
        window_data[0],         # amp1 (initial flat amplitude)
        window_data[0] + 0.5,   # amp2 (amplitude after the positive slope segment)
        0.5, window_data[0],               # decay and floor for the exponential decay
        max(t[len(window_data) // 2], min_t1),  # t1: start of the positive slope segment
        t[len(window_data) // 2 + 5]   # t2: end of the positive slope segment
    ])

    # Initial residuals
    residuals = piecewise_model_with_exp(params, t, window_data, sample_rate, min_t1)

    # Gauss-Newton loop with damping (Levenberg-Marquardt)
    for iteration in range(max_iter):

        # Calculate the Jacobian matrix for the current parameters
        J = jacobian_with_exp(params, t, sample_rate)

        # Compute the normal equation (J^T J + damping * I) dx = -J^T residuals
        JTJ = J.T @ J
        damping_matrix = damping * np.eye(len(JTJ))
        lhs = JTJ + damping_matrix
        rhs = -J.T @ residuals

        # Solve for the parameter update (dx)
        try:
            dx = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            damping *= 10
            continue

        # Update the parameters
        params += dx
        residuals_n = piecewise_model_with_exp(params, t, window_data, sample_rate, min_t1)

        # If any residual is inf or nan, handle by increasing damping and skipping the update
        if np.sum(residuals_n**2) > np.sum(residuals**2):
            params -= dx  # Undo the update
            damping *= 2
            continue

        # Check for convergence
        if np.sum(residuals**2) - np.sum(residuals_n**2) < tol:
            #print(f"Convergence achieved in iteration {iteration}; dampening: {damping}")
            break

        # Update residuals and decrease damping if residuals are stable
        residuals = residuals_n
        if np.linalg.norm(residuals) < np.inf:
            damping /= 2

    # Calculate the final misfit
    final_residuals = piecewise_model_with_exp(params, t, window_data, sample_rate, min_t1)
    final_misfit = np.sum(final_residuals ** 2)

    return params, final_misfit

def plot_slope_fit(window_data, sample_rate, fit_params, final_misfit, method):
    """
    Plots the fitted slope function and the original envelope data.
    
    Args:
        window_data: The original envelope data.
        sample_rate: The sample rate of the data.
        fit_params: The fitted parameters [amp1, amp2, decay, floor, t1, t2].
        final_misfit: Final misfit value.
        method: The optimization method used ('Gauss-Newton').
    """
    amp1, amp2, decay, floor, t1, t2 = fit_params
    t = np.linspace(0, len(window_data) / sample_rate, len(window_data))

    # Reconstruct the fitted function
    fitted_curve = np.empty_like(t)

    for i in range(len(t)):
        if t[i] < t1:
            fitted_curve[i] = amp1  # Interval 1 (flat at amp1)
        elif t1 <= t[i] < t2:
            fitted_curve[i] = amp1 + (amp2 - amp1) * (t[i] - t1) / (t2 - t1)  # Interval 2 (positive slope)
        else:
            # Interval 3: Exponential decay with a floor
            delta_t = t[i] - t2
            fitted_curve[i] = (amp2 - floor) * np.exp(-delta_t / decay) + floor

    # Plot the original envelope and the fitted slope function
    plt.figure(figsize=(12, 6))
    plt.plot(t, window_data, label='Log Envelope', color='blue')
    plt.plot(t, fitted_curve, label=f'Fitted Function (Exponential Decay); decay:{decay:.4f}', color='red', linestyle='--')
    plt.axvline(x=t1, color='green', linestyle=':', label=f'Start of Interval 2 (t1: {t1:.2f}s)')
    plt.axvline(x=t2, color='orange', linestyle=':', label=f'Start of Exponential Decay (t2: {t2:.2f}s)')

    plt.title(f'Fitted Slope Function ({method}). Misfit: {final_misfit:.4f}; gain: {20*(amp2 - amp1):.4f} dB')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (log scale)')
    plt.legend()
    plt.grid(True)
    plt.show()

def fit_slope_function(window_data, sample_rate, plot=True):
    """
    Fits a four-interval slope function to the log10 envelope of the data in the sliding window.
    Uses Gauss-Newton optimization with Levenberg-Marquardt damping.
    
    Returns:
        Tuple: (Optimized parameters [amp1, amp2, amp3, amp4, t1, t2, t3], final misfit value).
    """
    fit_params, final_misfit = gauss_newton_lm(window_data, sample_rate)

    # Optionally plot the fitted result
    if PLOT_EVERYTHING or (PLOT_ABOVE_THRESHOLD and THRESHOLD_GAIN_DB < 20*(fit_params[1] - fit_params[0])):
        plot_slope_fit(window_data, sample_rate, fit_params, final_misfit, method="Gauss-Newton")

    return fit_params, final_misfit

def slope_picker(log_envelope, sample_rate, window_length=2.0, threshold_gain=2, start_time=None, end_time=None):
    """
    Slope picker that fits a four-interval slope function to the log10 envelope of the data 
    and reports gain in dB, resolving overlapping picks by selecting the maximum reduction in power.

    Args:
        log_envelope: numpy array with the data to process.
        sample_rate: Sample rate of the data.
        window_length: Length of the window in seconds for fitting.
        threshold_gain: Threshold for saving picks based on gain in dB.
        start_time: Start time (in seconds) to limit picker processing.
        end_time: End time (in seconds) to limit picker processing.
    
    Returns:
        List of valid picks where closely spaced picks are resolved by selecting the maximum reduction in power.
    """

    # Convert the window length to samples
    window_length_samples = int(window_length * sample_rate)
    half_window_samples = window_length_samples // 16
    
    # Define the start and end sample indices for fitting
    start_idx = int(start_time * sample_rate) if start_time is not None else 0
    end_idx = int(end_time * sample_rate) if end_time is not None else len(log_envelope) - window_length_samples
    
    picks = []
    
    # Slide the window across the data and fit the slope function
    for i in range(start_idx, end_idx, half_window_samples):
        window_data = log_envelope[i:i + window_length_samples]
        
        if len(window_data) < window_length_samples:
            break  # Stop if we're at the end and the window isn't full
        
        # Fit the slope function to the current window
        fit_params, misfit = fit_slope_function(window_data, sample_rate)

        if fit_params is not None:
            #amp1, amp2, amp3, amp4, t1, t2, t3 = fit_params
            amp1, amp2, decay, floor, t1, t2 = fit_params
            
            # Calculate the gain in the middle intervals in dB
            gain_db = 20 * (amp2 - amp1)
            
            # Save the pick if the gain in dB is above the threshold
            if gain_db > threshold_gain:
                pick_time = (i + int(t1 * sample_rate)) / sample_rate  # Convert sample index to time
                picks.append({
                    'start_time': pick_time,
                    'gain_amplitude_db': gain_db,
                    'fit_params': fit_params,
                    'misfit': misfit  # Reduction in the power of the signal
                })

    # Resolve closely spaced picks by keeping the one with the highest reduction in power
    if picks:
        resolved_picks = []
        current_pick = picks[0]

        for next_pick in picks[1:]:
            # If the next pick is within one window length of the current pick
            if next_pick['start_time'] - current_pick['start_time'] <= window_length:
                # Keep the pick with the higher reduction (larger reduction means better fit)
                if next_pick['misfit'] < current_pick['misfit']:
                    current_pick = next_pick
            else:
                # No overlap, so save the current pick and move to the next one
                resolved_picks.append(current_pick)
                current_pick = next_pick
        
        # Don't forget to add the last pick
        resolved_picks.append(current_pick)
        
        return resolved_picks

    return picks

#@jit(nopython=True)
def gaussian_kernel(size, sigma):
    """Generates a 1D Gaussian kernel."""
    kernel = np.empty(size)
    sum_val = 0.0
    for i in range(size):
        x = i - size // 2
        kernel[i] = np.exp(-0.5 * (x / sigma) ** 2)
        sum_val += kernel[i]
    return kernel / sum_val  # Normalize the kernel

#@jit(nopython=True)
def gaussian_filter(signal, sigma=3):
    """
    Applies a Gaussian filter to smooth the signal.
    
    Args:
        signal: Input signal (1D array).
        sigma: Standard deviation of the Gaussian kernel.
        kernel_size: Size of the Gaussian kernel (should be odd).
        
    Returns:
        Smoothed signal (1D array).
    """

    kernel_size = sigma*5

    if kernel_size % 2 == 1:
        kernel_size += 1

    kernel = gaussian_kernel(kernel_size, sigma)
    half_size = kernel_size // 2
    smoothed_signal = np.empty_like(signal)
    
    for i in range(len(signal)):
        weighted_sum = 0.0
        weight_total = 0.0
        for j in range(kernel_size):
            idx = i + j - half_size
            if 0 <= idx < len(signal):
                weighted_sum += signal[idx] * kernel[j]
                weight_total += kernel[j]
        
        smoothed_signal[i] = weighted_sum / weight_total if weight_total > 0 else signal[i]
    
    return smoothed_signal

def combined_slope_picker(tr_vertical, tr_h1=None, tr_h2=None, tr_pressure=None, sample_rate=200,
                          min_vel=1.5, max_vel=8, distance_km=0, water_depth_km=0, pre_time=15,
                          window_length=2.0, threshold_gain=0.5, search_window=0.1):
    
    def mean_filter(signal, window_size=20):
        """Applies a simple mean filter to smooth the signal."""
        return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

    def calculate_apparent_velocity(pick_time, distance_km, water_depth_km, pre_time):
        effective_time = pick_time - pre_time
        if effective_time <= 0:
            return None
        return math.sqrt(distance_km**2 + water_depth_km**2) / effective_time

    # 1. Preprocess and pick vertical channel
    vertical_envelope = np.log10(np.abs(tr_vertical.data) + np.finfo(float).eps)
    vertical_envelope_smoothed = gaussian_filter(vertical_envelope)
    
    picks_vertical = slope_picker(vertical_envelope_smoothed, sample_rate=sample_rate, window_length=window_length,
                                  threshold_gain=threshold_gain, start_time=pre_time)

    if not picks_vertical:
        return -1, -1, -1, -1, -1, -1  # No vertical picks found, return all -1s
    
    for pick in picks_vertical:
        pick_time = pick['start_time']
        apparent_velocity = calculate_apparent_velocity(pick_time, distance_km, water_depth_km, pre_time)
        
        if apparent_velocity is not None and min_vel <= apparent_velocity <= max_vel:
            valid_picks = [{
                'channel': 'vertical',
                'start_time': pick_time,
                'gain_amplitude_db': pick['gain_amplitude_db'],
                'apparent_velocity': apparent_velocity,
                'fit_params': pick['fit_params'],
                'pick_idx': int(pick_time * sample_rate)
            }]
            break
    else:
        return -1, -1, -1, -1, -1, -1  # No valid vertical picks with valid apparent velocity, return all -1s

    # 2. Preprocess and pick horizontal channels, constrained around the vertical pick time
    best_horizontal_pick = None
    if tr_h1 is not None or tr_h2 is not None:
        for tr_horizontal in [tr_h1, tr_h2]:
            if tr_horizontal is not None:
                # Restrict horizontal search to the time window around the vertical pick
                start_time_h = max(0, valid_picks[0]['start_time'] - window_length)
                end_time_h = valid_picks[0]['start_time'] + 5 * window_length
                
                horizontal_envelope = np.log10(np.abs(tr_horizontal.data) + np.finfo(float).eps)
                horizontal_envelope_smoothed = gaussian_filter(horizontal_envelope)
                
                picks_horizontal = slope_picker(horizontal_envelope_smoothed, sample_rate=sample_rate, window_length=window_length,
                                                threshold_gain=threshold_gain, start_time=start_time_h, end_time=end_time_h)
                
                for pick in picks_horizontal:
                    pick_time_h = pick['start_time']
                    apparent_velocity_h = calculate_apparent_velocity(pick_time_h, distance_km, water_depth_km, pre_time)
                    
                    if apparent_velocity_h is not None and min_vel <= apparent_velocity_h <= max_vel:
                        if (best_horizontal_pick is None or pick['gain_amplitude_db'] > best_horizontal_pick['gain_amplitude_db']):
                            best_horizontal_pick = {
                                'channel': 'horizontal',
                                'start_time': pick_time_h,
                                'gain_amplitude_db': pick['gain_amplitude_db'],
                                'apparent_velocity': apparent_velocity_h,
                                'fit_params': pick['fit_params']
                            }

    if not best_horizontal_pick or best_horizontal_pick['apparent_velocity'] > valid_picks[0]['apparent_velocity']:
        return -1, -1, -1, -1, -1, -1  # No valid horizontal picks, return all -1s

    # 3. Preprocess and pick pressure channel, restricted to only one fit at the vertical pick time
    pressure_gain_db = -1
    if tr_pressure is not None:
        pressure_envelope = np.log10(np.abs(tr_pressure.data) + np.finfo(float).eps)
        pressure_envelope_smoothed = gaussian_filter(pressure_envelope)
        
        # Run the picker only around the vertical pick time
        start_time_p = valid_picks[0]['start_time']-window_length
        picks_pressure = slope_picker(pressure_envelope_smoothed, sample_rate=sample_rate, window_length=window_length,threshold_gain=threshold_gain, start_time=start_time_p, end_time=start_time_p + 2*window_length)
        
        for pick in picks_pressure:
            pick_time_p = pick['start_time']
            apparent_velocity_p = calculate_apparent_velocity(pick_time_p, distance_km, water_depth_km, pre_time)
            
            if apparent_velocity_p is not None and min_vel <= apparent_velocity_p <= max_vel:
                if abs(pick_time_p - valid_picks[0]['start_time']) <= search_window:
                    vertical_gain_db = valid_picks[0]['gain_amplitude_db']
                    
                    if pick['gain_amplitude_db'] < 3 * vertical_gain_db:
                        pressure_gain_db = pick['gain_amplitude_db']
                        valid_picks.append({
                            'channel': 'pressure',
                            'start_time': pick_time_p,
                            'gain_amplitude_db': pressure_gain_db,
                            'apparent_velocity': apparent_velocity_p,
                            'fit_params': pick['fit_params']
                        })

    if pressure_gain_db == -1:
        return -1, -1, -1, -1, -1, -1  # No valid pressure picks, return all -1s

    # Return the pick time, index in the array, and dB of the gain for all three picks (vertical, horizontal, pressure)
    return (valid_picks[0]['start_time'], valid_picks[0]['pick_idx'], 
            valid_picks[0]['gain_amplitude_db'], 
            best_horizontal_pick['gain_amplitude_db'], 
            pressure_gain_db, 
            valid_picks[0]['apparent_velocity']  if tr_pressure else -1)

def aic_pick_all(tr_vertical, tr_h1=None, tr_h2=None, tr_pressure=None, sample_rate=200, 
                 min_vel=1.5, max_vel=10, distance_km=0, water_depth_km=0, pre_time=10, 
                 background_percentile=99, search_window=0.1):
    
    threshold_multiplier = 4
    max_early_samples = 5  # Allow horizontal picks to be at most 5 samples earlier than the vertical pick

    # Compute AIC for vertical, horizontal, and pressure channels
    sd_vertical = np.diff(aic_simple(tr_vertical.data), n=2)
    sd_h1 = np.diff(aic_simple(tr_h1.data), n=2) if tr_h1 is not None else None
    sd_h2 = np.diff(aic_simple(tr_h2.data), n=2) if tr_h2 is not None else None
    sd_pressure = np.diff(aic_simple(tr_pressure.data), n=2) if tr_pressure is not None else None

    # Determine the background noise level for each channel
    pre_time_samples = int(pre_time * sample_rate)
    background_start = pre_time_samples // 4  # Skip first quarter of pre-time window

    # Calculate background noise thresholds for each channel
    background_sd_vertical = sd_vertical[background_start:pre_time_samples]
    threshold_vertical = threshold_multiplier * np.percentile(background_sd_vertical, background_percentile)

    threshold_h1 = threshold_multiplier * np.percentile(sd_h1[background_start:pre_time_samples], background_percentile) if tr_h1 is not None else None
    threshold_h2 = threshold_multiplier * np.percentile(sd_h2[background_start:pre_time_samples], background_percentile) if tr_h2 is not None else None
    threshold_pressure = threshold_multiplier * np.percentile(sd_pressure[background_start:pre_time_samples], background_percentile) if tr_pressure is not None else None

    picks = []
    water_column_detected = False  # Flag to stop processing picks after a water column arrival is detected

    for i in range(1, len(sd_vertical) - 1):
        if water_column_detected:
            # Stop further processing after water column arrival
            break

        if sd_vertical[i] > threshold_vertical and sd_vertical[i - 1] < sd_vertical[i] > sd_vertical[i + 1]:
            # Calculate apparent velocity for the vertical pick
            pick_time = (i - 3) / sample_rate - pre_time
            apparent_velocity = calculate_apparent_velocity(pick_time, distance_km, water_depth_km)

            # Check if apparent velocity is within the acceptable range for the vertical component
            if min_vel <= apparent_velocity <= max_vel:
                valid = True

                # Check the last 5 seconds before the pick
                five_seconds_before_pick = int(5 * sample_rate)
                start_idx_pre_pick = max(0, i - five_seconds_before_pick)
                max_pre_pick = np.max(sd_vertical[start_idx_pre_pick:i])

                # Ensure the pick is more than twice as big as anything in the last 5 seconds
                #if sd_vertical[i] <= 2 * max_pre_pick:
                #    valid = False

                # Check H1 channel if provided
                if tr_h1 is not None:
                    start_idx_h1 = max(0, i - max_early_samples)
                    end_idx_h1 = min(len(sd_h1), i + int(pre_time * sample_rate // 4))
                    h1_peak = np.max(sd_h1[start_idx_h1:end_idx_h1])

                    # Calculate apparent velocity for H1 pick
                    h1_pick_time = (start_idx_h1 + np.argmax(sd_h1[start_idx_h1:end_idx_h1]) - 3) / sample_rate - pre_time
                    h1_apparent_velocity = calculate_apparent_velocity(h1_pick_time, distance_km, water_depth_km)

                    # Check if H1 peak is valid, apparent velocity is within range, and timing is valid
                    h1_pick_index = start_idx_h1  # Convert h1_pick_time to an index
                    if h1_peak < threshold_h1 or not (min_vel <= h1_apparent_velocity <= max_vel) or h1_pick_index < i - max_early_samples:
                        valid = False

                # Check H2 channel if provided
                if tr_h2 is not None:
                    start_idx_h2 = max(0, i - max_early_samples)
                    end_idx_h2 = min(len(sd_h2), i + int(pre_time * sample_rate // 4))
                    h2_peak = np.max(sd_h2[start_idx_h2:end_idx_h2])

                    # Calculate apparent velocity for H2 pick
                    h2_pick_time = (start_idx_h2 + np.argmax(sd_h2[start_idx_h2:end_idx_h2]) - 3) / sample_rate - pre_time
                    h2_apparent_velocity = calculate_apparent_velocity(h2_pick_time, distance_km, water_depth_km)

                    # Check if H2 peak is valid, apparent velocity is within range, and timing is valid
                    h2_pick_index = start_idx_h2  # Convert h2_pick_time to an index
                    if h2_peak < threshold_h2 or not (min_vel <= h2_apparent_velocity <= max_vel) or h2_pick_index < i - max_early_samples:
                        valid = False

                # Check Pressure channel if provided
                if tr_pressure is not None:
                    start_idx_pressure = max(0, i - int(search_window * sample_rate))
                    end_idx_pressure = min(len(sd_pressure), i + int(search_window * sample_rate))
                    pressure_peak = np.max(sd_pressure[start_idx_pressure:end_idx_pressure])

                    if pressure_peak < threshold_pressure:
                        valid = False

                    # Cull water column arrivals, which are extremely clean on the hydrophone
                    if pressure_peak > threshold_pressure and pressure_peak > 3 * sd_vertical[i]:
                        water_column_detected = True
                        continue  # Ignore this pick and don't consider further picks

                # If pick is valid, store the qualities
                if valid:
                    # Get normalized qualities for vertical, horizontal, and pressure channels
                    vertical_quality = sd_vertical[i] / threshold_vertical

                    # Horizontal quality is the larger of H1 and H2
                    if tr_h1 is not None and tr_h2 is not None:
                        horizontal_quality = max(h1_peak / threshold_h1, h2_peak / threshold_h2)
                    elif tr_h1 is not None:
                        horizontal_quality = h1_peak / threshold_h1
                    elif tr_h2 is not None:
                        horizontal_quality = h2_peak / threshold_h2
                    else:
                        horizontal_quality = -1  # No horizontal channel present

                    pressure_quality = pressure_peak / threshold_pressure if tr_pressure is not None else -1

                    # Store the pick with qualities
                    picks.append((pick_time, apparent_velocity, vertical_quality, horizontal_quality, pressure_quality, i))

    # Resolve conflicts between picks within 0.25 seconds of each other
    final_picks = []
    if picks:
        # Sort picks by pick time
        picks.sort(key=lambda x: x[0])

        # Iterate through picks and remove conflicts
        for i in range(len(picks)):
            current_pick = picks[i]

            # Check if the next pick is within 0.25 seconds of the current pick
            if i < len(picks) - 1 and abs(picks[i + 1][0] - current_pick[0]) <= search_window:
                # If there's a conflict, keep the one with the earliest pick time
                continue
            else:
                # No conflict, keep the pick
                final_picks.append(current_pick)

    # Return the best pick, or invalid (-1, -1, -1, -1, -1, -1) if no valid picks found
    if final_picks:
        final_picks.sort(key=lambda x: x[0])  # Sort by pick time (earliest first)
        return final_picks[0]

    return -1, -1, -1, -1, -1, -1

def calculate_apparent_velocity(pick_time, distance_km, water_depth_km):
    if pick_time == -1:
        return -1
    total_distance_km = math.sqrt(distance_km**2 + water_depth_km**2)
    return total_distance_km / pick_time if pick_time != 0 else -1

def load_station_data(station_file_path):
    stations = []
    with open(station_file_path, 'r') as station_file:
        for line in station_file:
            if line.startswith("#") or not line.strip():
                continue
            network, station, lat, lon, elevation, sitename, start_time, end_time = line.strip().split('|')
            stations.append({
                'network': network,
                'station': station,
                'latitude': float(lat),
                'longitude': float(lon),
                'elevation': float(elevation),
                'sitename': sitename,
                'start_time': UTCDateTime(start_time),
                'end_time': UTCDateTime(end_time)
            })
    return stations

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def load_config(config_file_path):
    config = {}
    with open(config_file_path, 'r') as config_file:
        for line in config_file:
            parts = line.strip().split()
            if len(parts) != 3:
                print(f"Skipping invalid config line: {line.strip()}")
                continue
            chunk, shotlog_path, station_path = parts
            config[int(chunk)] = (shotlog_path, station_path)
    return config

def check_and_load_header(file):
    first_line = file.readline().strip()
    if first_line.startswith('#'):
        header = file.readline().strip().split()
        header = [col.lstrip('#') for col in header]
        return header, True
    else:
        default_header = ['shotnumber', 'date', 'time', 'sourceLat', 'sourceLon', 'shipLat', 'shipLon', 'waterDepth']
        return default_header, False

def is_valid_shot_format(columns):
    """Validate if the shot log line contains the correct columns for shot number, time, date, and locations."""
    try:
        # Check if the first field is a shot number (numeric)

        shot_number = int(float(columns[0]))

        # Detect time: should have two colons
        time_column = next((col for col in columns if ":" in col), None)
        if time_column:
            # Truncate the fractional seconds to six digits if necessary
            if '.' in time_column:
                seconds_part, fractional_part = time_column.split('.')
                if len(fractional_part) > 6:
                    time_column = f"{seconds_part}.{fractional_part[:6]}"
            
            # Ensure valid time format
            datetime.strptime(time_column, "%H:%M:%S.%f")
        else:
            return False
        
        # Detect date: should contain a year and a separator (`/` or `-`)
        date_column = next((col for col in columns[1:] if re.search(r'[-/]', col) and col.count('-') == 2 or col.count('/') == 2), None)

        if date_column:
            # Ensure valid date format
            detect_date_format(date_column)
        else:
            return False
        
        # At least two columns for locations (lat/lon or concatenated)
        if len(columns) >= 7:
            return True
        return False
    except (ValueError, IndexError):
        return False

def detect_date_format(date_str):
    """Detects the format of the date string based on the position of the year."""
    if re.match(r"\d{4}[-/]\d{2}[-/]\d{2}", date_str):
        # YYYY-MM-DD or YYYY/MM/DD
        return "%Y-%m-%d" if '-' in date_str else "%Y/%m/%d"
    elif re.match(r"\d{2}[-/]\d{2}[-/]\d{4}", date_str):
        # DD-MM-YYYY or DD/MM/YYYY
        return "%d-%m-%Y" if '-' in date_str else "%d/%m/%Y"
    elif re.match(r"\d{2}[-/]\d{2}[-/]\d{2}", date_str):
        # MM-DD-YY or MM/DD/YY (2-digit year)
        return "%m-%d-%y" if '-' in date_str else "%m/%d/%y"
    else:
        raise ValueError("Unrecognized date format.")

def parse_shot_time(columns):
    """Parses shot time by detecting time and date order from columns."""
    # Identify time and date columns
    time_column = next((col for col in columns if ":" in col), None)
    date_column = next((col for col in columns[1:] if re.search(r'[-/]', col) and col.count('-') == 2 or col.count('/') == 2), None)

    if not time_column or not date_column:
        return None

    # Truncate the fractional seconds to six digits if necessary
    if '.' in time_column:
        seconds_part, fractional_part = time_column.split('.')
        if len(fractional_part) > 6:
            time_column = f"{seconds_part}.{fractional_part[:6]}"

    # Detect the date format
    try:
        date_format = detect_date_format(date_column)
    except ValueError as e:
        print(f"Date format error: {e}")
        return None

    # Combine the date and time into one string and create a UTCDateTime object
    shot_time_str = f"{date_column} {time_column}"

    try:
        shot_time = UTCDateTime.strptime(shot_time_str, f"{date_format} %H:%M:%S.%f")
    except Exception as e:
        print(f"Error parsing shot time: {e}")
        return None

    return shot_time

def parse_shot_info(line):
    """Parses shot information from a line in the shot log."""
    # Split the line by commas or spaces/tabs
    columns = line.strip().split(',')

    # Check if the format is valid for comma-separated values
    if not is_valid_shot_format(columns):
        # Try splitting by spaces/tabs
        columns = re.split(r'\s+', line.strip())

        # If not valid, try splitting by tabs specifically
        if not is_valid_shot_format(columns):
            columns = line.strip().split('\t')  # Handle tab-separated logs

            # Check if the format is valid for tab-separated values
            if not is_valid_shot_format(columns):
                return None  # Unrecognized format

    # Parse the shot time
    shot_time = parse_shot_time(columns)
    if shot_time is None:
        return None  # Invalid shot time
    
    # Handle latitude and longitude parsing (either as 2 or 4 columns)
    try:
        if len(columns) >= 7:
            # Check if we have N/S and E/W indicators
            if any(char in columns[4] for char in ['N', 'S', 'E', 'W']):
                # Handle lat/lon with N/S and E/W indicators
                shot_lat = float(columns[4][:-1]) * (-1 if 'S' in columns[4] else 1)
                shot_lon = float(columns[5][:-1]) * (-1 if 'W' in columns[5] else 1)
                ship_lat = float(columns[6][:-1]) * (-1 if 'S' in columns[6] else 1)
                ship_lon = float(columns[7][:-1]) * (-1 if 'W' in columns[7] else 1)
                
                # Correct values if out of range
                if abs(shot_lat) > 90 or abs(shot_lon) > 180:
                    shot_lat /= 10000
                    shot_lon /= 10000
                if abs(ship_lat) > 90 or abs(ship_lon) > 180:
                    ship_lat /= 10000
                    ship_lon /= 10000

            else:
                # Check if lat/lon columns are concatenated (e.g., 45.123-165.456)
                shot_lat_split = columns[3].split('-')
                if len(shot_lat_split) == 2 and shot_lat_split[0] and shot_lat_split[1]:
                    # Concatenated lat/lon values
                    shot_lat = float(shot_lat_split[0])
                    shot_lon = -float(shot_lat_split[1])  # Apply negative sign to longitude
                    
                    # Check ship lat/lon for concatenation as well
                    ship_lat_split = columns[4].split('-')
                    if len(ship_lat_split) == 2 and ship_lat_split[0] and ship_lat_split[1]:
                        ship_lat = float(ship_lat_split[0])
                        ship_lon = -float(ship_lat_split[1])
                    else:
                        # Regular lat/lon values for the ship
                        ship_lat = float(columns[4])
                        ship_lon = float(columns[5])
                else:
                    # Regular lat/lon values for the shot and ship
                    shot_lat = float(columns[3])
                    shot_lon = float(columns[4])
                    ship_lat = float(columns[5])
                    ship_lon = float(columns[6])
                    
                # Correct values if out of range
                if abs(shot_lat) > 90 or abs(shot_lon) > 180:
                    shot_lat /= 10000
                    shot_lon /= 10000
                if abs(ship_lat) > 90 or abs(ship_lon) > 180:
                    ship_lat /= 10000
                    ship_lon /= 10000

        else:
            # Handle cases where lat/lon is split across two columns (concatenated with a hyphen)
            shot_lat_split = columns[3].split('-')
            if len(shot_lat_split) == 2 and shot_lat_split[0] and shot_lat_split[1]:
                shot_lat = float(shot_lat_split[0])
                shot_lon = -float(shot_lat_split[1])  # Apply negative sign to longitude

                ship_lat_split = columns[4].split('-')
                if len(ship_lat_split) == 2 and ship_lat_split[0] and ship_lat_split[1]:
                    ship_lat = float(ship_lat_split[0])
                    ship_lon = -float(ship_lat_split[1])
                else:
                    ship_lat = float(columns[4])
                    ship_lon = float(columns[5])
            else:
                # Regular lat/lon values
                shot_lat = float(columns[3])
                shot_lon = float(columns[4])
                ship_lat = float(columns[5])
                ship_lon = float(columns[6])

    except (ValueError, IndexError) as e:
        print(f"Error parsing lat/lon: {e}")
        return None

    # Return parsed shot info
    shot_info = {
        'shotLat': shot_lat,
        'shotLon': shot_lon,
        'shipLat': ship_lat,
        'shipLon': ship_lon,
        'shotTime': shot_time
    }

    return shot_info

"""
def load_shotlog_with_sampling(shotlog_path, max_shots=5000):
    with open(shotlog_path, 'r') as file:
        file.readline()
        header = file.readline().strip().split()

        all_lines = file.readlines()
        total_shots = len(all_lines)
        
        if total_shots > max_shots:
            step = total_shots // max_shots
            sampled_lines = all_lines[::step]
            sampled_lines = sampled_lines[:max_shots]
        else:
            sampled_lines = all_lines
    
    return header, sampled_lines
"""

def load_shotlog_with_sampling(shotlog_path, max_shots=5000):
    with open(shotlog_path, 'r') as file:
        # Read all lines from the file
        all_lines = file.readlines()

        # Initialize header as None
        header = None
        valid_lines = []

        # Iterate through each line and filter out headers or invalid lines
        for line in all_lines:
            # Strip leading/trailing whitespace
            stripped_line = line.strip()

            # Skip empty lines
            if not stripped_line:
                continue

            # Determine delimiter
            delimiter = ',' if ',' in stripped_line else None

            # Skip lines that start with '#' or contain words where numbers are expected
            if stripped_line.startswith('#') or not stripped_line[0].isdigit():
                if header is None:  # Only assign header once
                    header = stripped_line.split(delimiter)
                continue

            # If the line looks valid, add it to the list of valid lines
            valid_lines.append(stripped_line)

        # Sample the valid lines if there are more than max_shots
        total_shots = len(valid_lines)
        if total_shots > max_shots:
            step = total_shots // max_shots
            sampled_lines = valid_lines[::step][:max_shots]
        else:
            sampled_lines = valid_lines

    return header, sampled_lines

def find_priority_channels(st, time_tolerance_samples=2):
    # Initialize the channel variables
    vertical_channel, h1_channel, h2_channel, pressure_channel = None, None, None, None
    dpz_channel, dp1_channel, dp2_channel = None, None, None  # Special case for DPZ/DP1/DP2
    channel_type = ""

    # Define channel priorities and valid channel endings
    priorities = ['E', 'H', 'B', 'D']  # Added 'D' to the priority list

    found_vertical, found_h1, found_h2 = False, False, False

    # Variables to track start times and sample rates
    start_time = None
    end_time = None
    sample_rate = None

    for priority in priorities:
        for tr in st:
            
            # Skip channels where the middle letter is 'N' (strong motion channels)
            if len(tr.stats.channel) == 3 and tr.stats.channel[1].upper() == 'N':
                continue

            # Ensure middle letter is either 'L', 'H', 'P', or 'D'
            if len(tr.stats.channel) == 3 and tr.stats.channel[1].upper() not in ['L', 'H', 'P', 'D']:
                continue

            # Skip if we already have the channel (e.g., Z, H1, H2)
            if found_vertical and 'Z' in tr.stats.channel.upper():
                continue
            if found_h1 and (tr.stats.channel.upper().endswith('1') or tr.stats.channel.upper().endswith('N')):
                continue
            if found_h2 and (tr.stats.channel.upper().endswith('2') or tr.stats.channel.upper().endswith('E')):
                continue
            if pressure_channel and tr.stats.channel.upper() in ['HDH', 'EDH']:
                continue

            # Check if the channel starts with the given priority
            if tr.stats.channel.startswith(priority):
                # Check start time, end time, and sample rate consistency
                if start_time is None:
                    start_time = tr.stats.starttime
                    end_time = tr.stats.endtime
                    sample_rate = tr.stats.sampling_rate
                else:
                    # Allow for small timing differences (up to 2 samples)
                    start_time_diff = abs(tr.stats.starttime - start_time)
                    end_time_diff = abs(tr.stats.endtime - end_time)

                    max_tolerance_time = time_tolerance_samples / sample_rate

                    if start_time_diff > max_tolerance_time or end_time_diff > max_tolerance_time or tr.stats.sampling_rate != sample_rate:
                        continue

                # Assign vertical and horizontal channels based on their naming convention
                if 'Z' in tr.stats.channel.upper() and not found_vertical:
                    vertical_channel = tr
                    found_vertical = True
                    channel_type = priority

                elif (tr.stats.channel.upper().endswith('1') or tr.stats.channel.upper().endswith('N')) and not found_h1:
                    h1_channel = tr
                    found_h1 = True

                elif (tr.stats.channel.upper().endswith('2') or tr.stats.channel.upper().endswith('E')) and not found_h2:
                    h2_channel = tr
                    found_h2 = True

                elif tr.stats.channel.upper() in ['HDH', 'EDH'] and pressure_channel is None:
                    pressure_channel = tr

            # Special case for DPZ/DP1/DP2 class stations
            if tr.stats.channel.upper() == 'DPZ':
                dpz_channel = tr
            elif tr.stats.channel.upper() == 'DP1':
                dp1_channel = tr
            elif tr.stats.channel.upper() == 'DP2':
                dp2_channel = tr

        # Stop if we've found all required channels or DPZ/DP1/DP2 combination
        if (found_vertical and found_h1 and found_h2) or (dpz_channel and dp1_channel and dp2_channel):
            break

    # If DPZ/DP1/DP2 are found, we only need those channels (no pressure)
    if dpz_channel and dp1_channel and dp2_channel:
        return dpz_channel, dp1_channel, dp2_channel, None, "DPZ/DP1/DP2"

    # Ensure that vertical and both horizontal channels were found and have consistent timing/sample rates
    if not (found_vertical and (found_h1 or found_h2)):
        return None, None, None, None, ""

    # Return the identified channels and channel type
    return vertical_channel, h1_channel, h2_channel, pressure_channel, channel_type

def generate_file_paths(chunk_number, file_count):
    formatted_chunk_number = f"{chunk_number:04d}0{file_count:02d}"  # First 4 digits from chunk number, next 2 for file count
    hdf5_file_path = f"waveforms{formatted_chunk_number}.hdf5"
    csv_file_path = f"metadata{formatted_chunk_number}.csv"
    return hdf5_file_path, csv_file_path

def check_file_size_and_rotate(hdf5_file, csv_file, data_group, data_format_group, csv_writer, file_count, chunk_number, csv_header):
    max_file_size_gb = 2
    max_file_size_bytes = max_file_size_gb * 1024 * 1024 * 1024

    current_file_size = os.path.getsize(hdf5_file.filename)

    if current_file_size >= max_file_size_bytes:
        hdf5_file.close()
        csv_file.close()

        file_count += 1
        hdf5_file_path, csv_file_path = generate_file_paths(chunk_number, file_count)

        # Open new HDF5 and CSV files
        hdf5_file = hdf5_file = h5py.File(hdf5_file_path, 'w')
        csv_file = open(csv_file_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)

        # Write the CSV header to the new file
        csv_writer.writerow(csv_header)
        csv_file.flush()

        # Create new data and format groups
        data_group = hdf5_file.create_group('data')
        data_format_group = hdf5_file.create_group('data_format')

        # Set the same attributes as before
        data_format_group.attrs['dimension_order'] = 'NCW'
        data_format_group.attrs['component_order'] = 'Z12H'
        data_format_group.attrs['sampling_rate'] = 200
        data_format_group.attrs['instrument_response'] = 'not restituted'

    return hdf5_file, csv_file, data_group, data_format_group, csv_writer, file_count

"""
def get_waveforms_from_clients(clients, station_data, start_time, end_time, max_retries=5):
    network = station_data['network']
    station = station_data['station']
    location = "*"  # Hardwired for all locations
    channels = "*Z,*1,*2,*N,*E,*H,*D"  # Hardwired channels

    # Iterate over the list of clients
    for client in clients:
        try:
            for retry_attempt in range(max_retries):
                try:
                    # Try to download the data with the current client
                    st = client.get_waveforms(network, station, location, channels, start_time, end_time, attach_response=True)
                    
                    # Filter out channels that don't start with E, B, H, or D
                    st = st.select(channel="E*") + st.select(channel="B*") + st.select(channel="H*") + st.select(channel="D*")

                    # Check that the traces have their responses
                    for tr in st:
                        if tr.stats.channel[0] in "EBHD" and not tr.stats.response:#seems funky but ok
                            raise ValueError(f"Missing response for channel: {tr.stats.channel}")

                    return st  # Return the stream if successful

                except ConnectionResetError:
                    # Retry after a random delay up to 5 minutes
                    wait_time = np.random.uniform(30, 300)  # 30 seconds to 5 minutes
                    print(f"ConnectionResetError encountered, retrying after {wait_time:.1f} seconds...")
                    time.sleep(wait_time)

                except Exception as e:

                    if 'No data' in str(e):
                        break                        
                    else:
                        print(f"Error encountered: {e}. Retrying (Attempt {retry_attempt + 1}/{max_retries})...")
                        time.sleep(1)  # Short delay before retrying

            # If retries are exhausted for this client
            print(f"Failed to retrieve data from client {client.base_url}. Moving to the next client.")

        except ValueError as e:
            # If there's a "No data" error, move to the next client immediately
            if 'No data' in str(e):
                print(f"No data found for station {station} on client {client.base_url}. Moving to the next client.")
                break  # Move to the next client
            else:
                raise  # Reraise other ValueErrors

    # If all clients fail
    raise RuntimeError("Failed to download data from all clients.")
"""

def get_and_process_waveforms(clients, station_data, start_time, end_time, filt=2.5, max_retries=10):
    network = station_data['network']
    station = station_data['station']
    location = "*"  # Hardwired for all locations
    channels = "*Z,*1,*2,*N,*E,*H,*D"  # Hardwired channels

    for client in clients:
        try:
            for retry_attempt in range(max_retries):
                try:
                    # Attempt to download the data with the current client
                    st = client.get_waveforms(network, station, location, channels, start_time, end_time, attach_response=True)
                    
                    # Filter out unwanted channels that don't start with E, B, H, or D
                    st = st.select(channel="E*") + st.select(channel="B*") + st.select(channel="H*") + st.select(channel="D*")

                    # Check that all traces have their responses
                    for tr in st:
                        if tr.stats.channel[0] in "EBHD" and not tr.stats.response:
                            raise ValueError(f"Missing response for channel: {tr.stats.channel}")

                    # Sort the stream by channel
                    st.sort(keys=['channel'])

                    # Find priority channels (vertical, horizontal, pressure)
                    vertical_channel, h1_channel, h2_channel, pressure_channel, channel_type = find_priority_channels(st)

                    # If no vertical channel is found, break and skip this station
                    if vertical_channel is None:
                        break

                    # Process all downloaded traces: detrend, resample, taper, and filter
                    for tr in [vertical_channel, h1_channel, h2_channel, pressure_channel]:
                        if not tr:
                            continue
                        tr.remove_response(output="VEL") 
                        tr.detrend("demean")
                        tr.resample(200.0)
                        tr.taper(0.01)
                        tr.filter("highpass", freq=filt)

                    # Return the processed channels if successful
                    return vertical_channel, h1_channel, h2_channel, pressure_channel, channel_type

                except Exception as e:
                    if 'No data' in str(e):
                        print(f"No data found for station {station} on client {client.base_url}. Moving to the next client.")
                        break  # Move to the next client if no data
                    elif 'ConnectionResetError' in str(e): # Retry after a random delay up to 3 minutes
                        wait_time = np.random.uniform(30, 180)
                        print(f"ConnectionResetError encountered, retrying after {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"Error encountered: {e}. Retrying (Attempt {retry_attempt + 1}/{max_retries})...")
                        time.sleep(1)  # Short delay before retrying

            # If retries are exhausted for this client
            print(f"Failed to retrieve data from client {client.base_url}. Moving to the next client.")

        except ValueError as e:
            # Handle missing data from clients and move to the next client
            if 'No data' in str(e):
                print(f"No data found for station {station} on client {client.base_url}. Moving to the next client.")
                break  # Move to the next client
            else:
                raise  # Reraise any other ValueErrors

    # If no clients succeed, raise an error
    raise RuntimeError(f"Failed to download data for station {station} from all clients.")

def download_data(chunk, config_file_path, pick_or_all=True, min_vel=1.6, max_vel=9, max_range=100, 
                  max_lines=1e10, filt=2.5, window_length=30, 
                  min_shot_interval=45, min_db=15, verbose=True, chunk_time=60*60*4):

    clients = []
    client_names = ["IRIS", "IRISPH5", "GEOFON", "USGS", "NOA", "NCEDC" ]

    for client_name in client_names:
        retry_count = 0
        while retry_count < 100:
            try:
                time.sleep(np.random.rand()*5)  # Wait before trying. System rejects rapid requests. 
                clients.append(Client(client_name))
                #print(f"Successfully added client {client_name}")
                break  # Exit the retry loop on success
            except Exception as e:
                print(f"Failed to initialize client {client_name} (Attempt {retry_count + 1}/10): {e}")
                retry_count += 1
                time.sleep(np.random.rand()*120)  # Wait before retrying

        if retry_count == 10:
            raise ValueError(f"Failed to initialize client {client_name} after 10 attempts.")

    config = load_config(config_file_path)
    
    if chunk not in config:
        raise ValueError(f"Chunk {chunk} not found in config file.")
    
    shotlog_path, station_path = config[chunk]

    stations = load_station_data(station_path)
    total_stations = len(stations)  # Track total number of stations
    
    header, sampled_lines = load_shotlog_with_sampling(shotlog_path, max_shots=max_lines)
    total_shots_in_log = len(sampled_lines)

    file_count = 1

    # Initial file paths
    hdf5_file_path, csv_file_path = generate_file_paths(chunk, file_count)

    # Open the files initially
    hdf5_file = h5py.File(hdf5_file_path, 'w')
    csv_file = open(csv_file_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)

    # Write the updated CSV header (without the shot log header)
    csv_header = [
        'source_latitude_deg', 'source_longitude_deg', 'source_origin_time',
        'source_shotlog_name', 'station_latitude_deg', 'station_longitude_deg', 
        'station_elevation_m', 'station_network', 'station_code', 'station_sitename', 
        'path_distance_km', 'path_apparent_velocity_km_s', 'trace_Z_snr_db', 
        'trace_H_snr_db', 'trace_P_snr_db', 'trace_pick_time', 
        'station_channeltype', 'trace_category', 'station_description', 'source_origin_time'
    ]
    csv_writer.writerow(csv_header)
    csv_file.flush()

    data_group = hdf5_file.create_group('data')
    data_format_group = hdf5_file.create_group('data_format')
    
    data_format_group.attrs['dimension_order'] = 'NCW'
    data_format_group.attrs['component_order'] = 'Z12H'
    data_format_group.attrs['sampling_rate'] = 200
    data_format_group.attrs['instrument_response'] = 'restituted'

    shots_downloaded = 0

    # Iterate through stations
    for station_idx, station_data in enumerate(stations, start=1):
        print(f"\nProcessing station {station_data['station']} ({station_idx}/{total_stations})...")

        shots_downloaded_per_station = 0
        failures = 0  # Don't try everything if this station isn't working

        last_shot_time = None  # To track the last valid shot time

        chunk_start = None
        chunk_end = None

        for idx, line in enumerate(sampled_lines):

            if failures > 10:
                print(f"Station seeing too many failed downloads, skipping")
                break

            # Example usage in a loop processing the shot log
            try:
                shot_info = parse_shot_info(line)

                # If parsing fails, move to the next line
                if shot_info is None:
                    continue

                # Calculate the total distance using Pythagorean theorem (hypotenuse)
                distance_km = haversine(station_data['latitude'], station_data['longitude'], shot_info['shotLat'], shot_info['shotLon'])
                water_depth_km = abs(station_data['elevation']) / 1000.0
                total_distance_km = math.sqrt(distance_km ** 2 + water_depth_km ** 2)

                shot_time = shot_info['shotTime']

                # Skip the shot if the shot time is not within the station's operational time
                if not station_data['start_time'] <= shot_time <= station_data['end_time']:
                    continue

                # If the distance is greater than the max range, skip this shot
                if distance_km > max_range:
                    continue

            except (ValueError, IndexError) as e:
                continue  # Skip this shot if there's a parsing error

            if not chunk_start or shot_time + 60 > chunk_end or last_shot_time > shot_time:
                #make a new chunk at start of a station, when you are at the end of the chunk, or if there is a skip back in the shot log

                chunk_start = shot_time - (3*2)*window_length
                chunk_end = chunk_start + chunk_time

                try:

                    Cvertical_channel, Ch1_channel, Ch2_channel, Cpressure_channel, channel_type = get_and_process_waveforms(clients, 
                        station_data, chunk_start, chunk_end, filt=filt, max_retries=5)
                    
                except:
                    print("Data retrieval failed, skipping.")
                    break
                
            # Check if last_shot_time is set and ensure the shots are within the minimum interval and 5-minute range
            if last_shot_time:
                time_difference = np.abs(shot_time - last_shot_time)

                # Reset the last shot time if the time difference exceeds 5 minutes (300 seconds)
                if time_difference > 300: #this happens because the shot log isn't always monotonic in time
                    last_shot_time = None
                elif time_difference < min_shot_interval and len(sampled_lines) > 20000:
                    continue

            # Update last_shot_time to the current shot time
            last_shot_time = shot_time

            try:
                # Determine the next shot time for error checking and limiting the download duration
                if idx < len(sampled_lines) - 1:
                    try:
                        next_shot_info = parse_shot_info(sampled_lines[idx + 1])
                        next_shot_time = next_shot_info['shotTime']

                        if next_shot_time is None:
                            raise ValueError("Next shot time parsing failed")
                    except Exception as e:
                        next_shot_time = shot_time + 120  # Default to 120 seconds if next shot time is invalid
                else:
                    next_shot_time = shot_time + 120  # Default for the last shot

                # Calculate travel time based on the minimum velocity (min_vel)
                travel_time = total_distance_km / min_vel

                # Adjust the end time to be shot_time + travel_time + window_length
                next_shot_time = shot_time + travel_time + window_length

                # Apply constraints: ensure the duration is at most 120 seconds
                requested_duration = next_shot_time - shot_time
                if requested_duration > 120:
                    next_shot_time = shot_time + 120

            except Exception as e:
                continue  # Skip to the next shot if the current shot time is invalid

            # Adjust the start time to be window_length before the shot
            start_time = shot_time - (3/2)*window_length

            shots_downloaded += 1
            shots_downloaded_per_station += 1

            # Print shot download info if verbose is True
            if verbose:
                print(f"\rProcessing data for station {station_data['station']} ({idx+1}/{total_shots_in_log})", end="")

            sys.stdout.flush()

            vertical_channel = Cvertical_channel.copy()
            vertical_channel.trim(starttime=start_time, endtime=next_shot_time)

            #These can be none, it's not a bug
            if Ch1_channel:
                h1_channel       = Ch1_channel.copy()
                h1_channel.trim(starttime=start_time, endtime=next_shot_time)
            else:
                h1_channel = None

            if Ch2_channel:
                h2_channel       = Ch2_channel.copy()
                h2_channel.trim(starttime=start_time, endtime=next_shot_time)
            else:
                h2_channel = None

            if Cpressure_channel:
                pressure_channel = Cpressure_channel.copy()
                pressure_channel.trim(starttime=start_time, endtime=next_shot_time)
            else:
                pressure_channel = None

            pick_time, pick_index, vertical_quality, horizontal_quality, pressure_quality, apparent_velocity = combined_slope_picker(
                vertical_channel, tr_h1=h1_channel, tr_h2=h2_channel, tr_pressure=pressure_channel, sample_rate=200,
                min_vel=min_vel, max_vel=max_vel, distance_km=distance_km, water_depth_km=water_depth_km, pre_time=(3/2)*window_length,
                window_length=3, threshold_gain=min_db, search_window=0.33
            )

            # If pick is invalid, continue
            if pick_or_all and pick_time == -1:
                continue

            # Calculate the sample indices for signal and noise
            #pick_time_in_seconds = int(pick_time * vertical_channel.stats.sampling_rate)
            total_window = int(window_length * vertical_channel.stats.sampling_rate)

            # Recenter pick at 1/2 into the signal window
            start_idx_signal = pick_index - total_window // 2
            end_idx_signal = start_idx_signal + total_window

            # Noise window starts at the beginning of the recentered signal window
            start_idx_noise = start_idx_signal - total_window
            end_idx_noise = start_idx_signal

            # Create waveforms for signal and noise
            signal_waveforms = np.array([
                tr.data[start_idx_signal:end_idx_signal] if tr is not None else np.zeros(end_idx_signal - start_idx_signal)
                for tr in [vertical_channel, h1_channel, h2_channel, pressure_channel]
            ])

            noise_waveforms = np.array([
                tr.data[start_idx_noise:end_idx_noise] if tr is not None else np.zeros(end_idx_noise - start_idx_noise)
                for tr in [vertical_channel, h1_channel, h2_channel, pressure_channel]
            ])

            # Write the signal metadata row
            signal_row = [
                shot_info['shotLat'], shot_info['shotLon'], shot_time, 
                os.path.basename(station_path).replace('.txt', ''), 
                station_data['latitude'], station_data['longitude'], station_data['elevation'], 
                station_data['network'], station_data['station'], station_data['sitename'], 
                distance_km, apparent_velocity, vertical_quality, 
                horizontal_quality, pressure_quality, pick_time, channel_type, 'S',
                station_data['sitename'], shot_time.isoformat()
            ]
            csv_writer.writerow(signal_row)
            csv_file.flush()

            # Write the noise metadata row with NaNs for source and path data, keeping station data intact
            noise_row = [
                float('nan'), float('nan'), float('nan'), 
                float('nan'),  # NaN for source_ and _path variables
                station_data['latitude'], station_data['longitude'], station_data['elevation'], 
                station_data['network'], station_data['station'], station_data['sitename'], 
                float('nan'), float('nan'), float('nan'),  # NaN for path data
                float('nan'),  # NaN for pick_time
                channel_type, 'N',
                station_data['sitename'], shot_time.isoformat()
            ]
            csv_writer.writerow(noise_row)
            csv_file.flush()

            # Write the signal and noise waveforms separately
            trace_id_signal = f"{station_data['station']}_{shots_downloaded}_S"
            trace_id_noise = f"{station_data['station']}_{shots_downloaded}_N"

            if trace_id_signal in data_group:
                print(f"Dataset {trace_id_signal} already exists. Skipping...")
            else:
                data_group.create_dataset(trace_id_signal, data=signal_waveforms, chunks=True)
                hdf5_file.flush()

            if trace_id_noise in data_group:
                print(f"Dataset {trace_id_noise} already exists. Skipping...")
            else:
                data_group.create_dataset(trace_id_noise, data=noise_waveforms, chunks=True)
                hdf5_file.flush()

            # Call the file rotation function and get updated file handles if needed
            #hdf5_file, csv_file, data_group, data_format_group, csv_writer, file_count = check_file_size_and_rotate(
            #    hdf5_file, csv_file, data_group, data_format_group, csv_writer, file_count, chunk, csv_header)

    # Close the final files
    csv_file.flush()
    csv_file.close()
    hdf5_file.flush()
    hdf5_file.close()

    print(f"\nAll shots processed and saved. Processed {shots_downloaded} shots.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download seismic data based on a chunk configuration.")
    parser.add_argument('--chunk', required=True, type=int, help='6-digit integer trace chunk number')
    parser.add_argument('--config', required=True, help='Path to the configuration file')
    parser.add_argument('--window_length', required=False, type=int, default=30, help='Window length in seconds centered on pick time')
    parser.add_argument('--filter', required=False, type=float, default=5, help='Corner for high-pass filter in Hz')
    parser.add_argument('--max_range', required=False, type=int, default=100, help='Maximum distance from the shot to consider, km')
    parser.add_argument('--min_db', required=False, type=float, default=15, help='Minimum signal-to-noise ratio in dB')
    
    # Using store_true to toggle verbose with default set to True
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output (default)')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false', help='Disable verbose output')

    # Default verbose to True if neither flag is provided
    parser.set_defaults(verbose=True)

    args = parser.parse_args()

    download_data(args.chunk, args.config, window_length=args.window_length, filt=args.filter,
                  max_range=args.max_range, verbose=args.verbose, min_db=args.min_db)