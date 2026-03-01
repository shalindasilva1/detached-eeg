import mne
import numpy as np
import pandas as pd
from typing import List, Tuple, Any

def preprocess_eeg(raw: mne.io.BaseRaw, target_sfreq: float = 128.0) -> mne.io.BaseRaw:
    """
    Apply basic preprocessing:
    1. Resampling to target frequency (e.g., 128Hz).
    2. (Placeholder) Filtering, artifact rejection, etc.
    """
    if raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq)
    # Add more steps as needed (filtering: raw.filter(1, 40))
    return raw

def load_and_format_data(df: pd.DataFrame, subject_ids: List[str], target_sfreq: float = 128.0, duration: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EEG data for specific subjects and format it for time-series classification.
    
    Returns:
        X: array of shape (n_instances, n_channels, n_timepoints)
        y: array of labels
    """
    X_list = []
    y_list = []
    
    for sub_id in subject_ids:
        sub_df = df[df['participant_id'] == sub_id]
        if sub_df.empty:
            continue
            
        group = sub_df['Group'].iloc[0]
        # Label 0 for Control, 1 for anything else (AD/FTD) in binary mode
        label = 0 if group == 'C' else 1
        
        for _, row in sub_df.iterrows():
            try:
                raw = mne.io.read_raw(row['eeg_file'], preload=True)
                raw = preprocess_eeg(raw, target_sfreq=target_sfreq)
                
                # Get data and crop to duration if needed
                data = raw.get_data()
                n_samples = int(target_sfreq * duration)
                
                if data.shape[1] >= n_samples:
                    X_list.append(data[:, :n_samples])
                    y_list.append(label)
            except Exception as e:
                print(f"Error loading {row['eeg_file']}: {e}")
                
    if not X_list:
        return np.array([]), np.array([])
        
    return np.stack(X_list), np.array(y_list)
