import numpy as np
from typing import List, Dict, Any
from sklearn.model_selection import StratifiedKFold

def get_stratified_splits(controls: List[str], ad: List[str], ftd: List[str], n_splits: int = 5, seed: int = 42, binary_mode: bool = False) -> List[Dict[str, Any]]:
    """
    Generate stratified K-fold splits ensuring balanced group representation and no subject overlap.
    
    If binary_mode is True, groups AD and FTD subjects into class 1 (Control remains 0).
    """
    all_subjects = np.concatenate([
        np.array(controls, dtype=object),
        np.array(ad, dtype=object),
        np.array(ftd, dtype=object),
    ])

    # Default: 0 = Control, 1 = AD, 2 = FTD
    all_labels = np.concatenate([
        np.full(len(controls), 0, dtype=int),
        np.full(len(ad), 1, dtype=int),
        np.full(len(ftd), 2, dtype=int),
    ])

    if binary_mode:
        # Group AD (1) and FTD (2) into a single category (1)
        all_labels[all_labels > 0] = 1

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    splits = []
    for train_idx, test_idx in kf.split(all_subjects, all_labels):
        splits.append({
            "train_subjects": all_subjects[train_idx].tolist(),
            "test_subjects": all_subjects[test_idx].tolist(),
            "train_labels": all_labels[train_idx].tolist(),
            "test_labels": all_labels[test_idx].tolist()
        })
    return splits
