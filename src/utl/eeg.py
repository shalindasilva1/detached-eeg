import mne
import numpy as np
import pandas as pd
import warnings
from typing import List, Tuple, Any

# Suppress MNE boundary warnings
warnings.filterwarnings("ignore", message="The data contains 'boundary' events")


def preprocess_eeg(raw: mne.io.BaseRaw, target_sfreq: float = 128.0) -> mne.io.BaseRaw:
    """
    Apply minimal formatting for the model:
    1. Select 19 scalp electrodes mentioned in the paper.
    2. Resampling to target frequency.
    Note: Filtering and Re-referencing are assumed to be already done in the derivative files.
    """
    # 1. Select 19 scalp electrodes mentioned in the paper
    ch_names_19 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    # Filter only available channels from the list
    available_channels = [ch for ch in ch_names_19 if ch in raw.ch_names]
    if len(available_channels) < len(ch_names_19):
        print(f"Warning: Only {len(available_channels)}/19 target channels found in {raw.filenames[0]}")
    raw.pick(available_channels)

    # 2. Resampling
    if raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)
    
        
    return raw

def load_and_format_data(df: pd.DataFrame, subject_ids: List[str], target_sfreq: float = 128.0, trial_duration: float = 5.0, max_trials_per_subject: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load EEG data and split into trials/epochs.
    
    Returns:
        X: (n_trials, n_channels, n_timepoints)
        y: (n_trials,)
        subject_indices: (n_trials,) mapping each trial back to a subject ID in subject_ids
    """
    X_list = []
    y_list = []
    subject_map = []
    
    for sub_idx, sub_id in enumerate(subject_ids):
        sub_df = df[df['participant_id'] == sub_id]
        if sub_df.empty:
            continue
            
        group = sub_df['Group'].iloc[0]
        # Label 0 for Control, 1 for AD
        label = 0 if group == 'C' else 1
        
        for _, row in sub_df.iterrows():
            try:
                raw = mne.io.read_raw(row['eeg_file'], preload=True, verbose=False)
                raw = preprocess_eeg(raw, target_sfreq=target_sfreq)
                
                # Split continuous data into fix-length trials
                data = raw.get_data()
                n_samples_per_trial = int(target_sfreq * trial_duration)
                n_channels = data.shape[0]
                total_samples = data.shape[1]
                
                n_trials = total_samples // n_samples_per_trial
                
                if n_trials > 0:
                    # Reshape to (n_trials, n_channels, n_samples_per_trial)
                    trials = data[:, :n_trials * n_samples_per_trial].reshape(n_channels, n_trials, n_samples_per_trial)
                    trials = trials.transpose(1, 0, 2) # (n_trials, n_channels, n_samples)
                    
                    if max_trials_per_subject:
                        trials = trials[:max_trials_per_subject]
                        n_trials = trials.shape[0]

                    X_list.append(trials)
                    y_list.extend([label] * n_trials)
                    subject_map.extend([sub_idx] * n_trials)
            except Exception as e:
                print(f"Error loading {row['eeg_file']}: {e}")
                
    if not X_list:
        return np.array([]), np.array([]), np.array([])
        
    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)
    s = np.array(subject_map)
    
    return X, y, s

