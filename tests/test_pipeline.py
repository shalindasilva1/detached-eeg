import pytest
import numpy as np
import mne
from src.utl.eeg import preprocess_eeg

def test_preprocess_eeg_resampling():
    # Create mock EEG data: 5 seconds, 10 channels, 500 Hz
    sfreq = 500.0
    n_channels = 10
    n_samples = int(5 * sfreq)
    data = np.random.randn(n_channels, n_samples)
    
    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    
    target_sfreq = 128.0
    processed_raw = preprocess_eeg(raw.copy(), target_sfreq=target_sfreq)
    
    # Check if the sampling frequency is updated
    assert processed_raw.info['sfreq'] == target_sfreq
    
    # Check if the number of samples is approximately correct
    # Expected samples = original_samples * (target_sfreq / original_sfreq)
    expected_samples = int(n_samples * (target_sfreq / sfreq))
    assert processed_raw.n_times == expected_samples

def test_preprocess_eeg_no_resampling_needed():
    # Create mock EEG data already at 128 Hz
    sfreq = 128.0
    n_channels = 5
    n_samples = 1280 # 10 seconds
    data = np.random.randn(n_channels, n_samples)
    
    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    
    target_sfreq = 128.0
    processed_raw = preprocess_eeg(raw.copy(), target_sfreq=target_sfreq)
    
    # Should maintain sfreq
    assert processed_raw.info['sfreq'] == target_sfreq
    # Samples should remain same
    assert processed_raw.n_times == n_samples

def test_preprocess_eeg_sine_wave_preservation():
    # Create a 10Hz sine wave at 1000Hz sampling rate
    sfreq = 1000.0
    t = np.arange(0, 1.0, 1.0/sfreq)
    freq = 10.0
    sine = np.sin(2 * np.pi * freq * t)
    
    data = np.array([sine])
    info = mne.create_info(ch_names=['SINE'], sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    
    target_sfreq = 100.0
    processed_raw = preprocess_eeg(raw.copy(), target_sfreq=target_sfreq)
    
    # Check if sfreq is updated
    assert processed_raw.info['sfreq'] == target_sfreq
    
    # Verify signal: find the peak frequency in the resampled signal
    resampled_data = processed_raw.get_data()[0]
    psd = np.abs(np.fft.rfft(resampled_data))
    freqs = np.fft.rfftfreq(len(resampled_data), 1.0/target_sfreq)
    peak_freq = freqs[np.argmax(psd)]
    
    # Peak frequency should still be 10Hz (within some tolerance)
    assert abs(peak_freq - freq) < 1.0
