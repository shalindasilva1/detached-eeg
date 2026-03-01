import mne
import numpy as np
import json
import os
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, classification_report

from utl.config import load_config
from utl.bids import load_bids_dataset
from utl.splits import get_stratified_splits
from utl.eeg import load_and_format_data

# Import Detach Rocket
try:
    from detach_rocket.detach_classes import DetachEnsemble
except ImportError:
    try:
        from detach_rocket.detach_rocket import DetachEnsemble
    except ImportError:
        print("Warning: detach_rocket not installed correctly. Using placeholder.")
        DetachEnsemble = None

# Make MNE quieter
mne.set_log_level("WARNING")

class TaskEEGPipeline:
    def __init__(self, config_path: str = "config.yml"):
        self.config = load_config(config_path)
        self.resting_path = self.config['data']['resting']
        
        # Load experiment settings
        exp_config = self.config.get('experiment', {})
        self.split_seed = exp_config.get('seed', 42)
        self.n_splits = exp_config.get('n_splits', 5)
        self.binary_mode = exp_config.get('binary_classification', True)
        
        self.df: Any = None
        self.subjects: Dict[str, List[str]] | None = None
        self.splits: List[Dict[str, Any]] | None = None

    def initialize(self):
        """Discover files and prepare splits."""
        print(f"Initializing pipeline with data at: {self.resting_path}")
        self.df, (controls, ad, ftd) = load_bids_dataset(self.resting_path)
        
        if self.df is not None and self.df.empty:
            print("No data found. Check your config.yml paths.")
            return

        n_processed = self.df[self.df['is_derivative'] == True].shape[0] if 'is_derivative' in self.df.columns else 0
        print(f"Discovered {len(self.df)} EEG records ({n_processed} from derivatives).")


        # Filtering for AD vs CN as per Section 5.2 of the paper
        print(f"Total discovered subjects: {len(controls)} Control, {len(ad)} AD, {len(ftd)} FTD.")
        
        # Apply subject limit if specified
        exp_config = self.config.get('experiment', {})
        max_subjects = exp_config.get('max_subjects')
        if max_subjects:
            half = max_subjects // 2
            controls = controls[:half]
            ad = ad[:max_subjects - half]
            print(f"Limiting to {len(controls)} Control and {len(ad)} AD subjects (Total: {len(controls) + len(ad)}).")
        else:
            print("Replicating Section 5.2: Focusing on AD vs CN classification (total 65 subjects).")

        self.subjects = {"Control": controls, "AD": ad}
        
        # LOSO Cross-validation subjects
        loso_subjects = controls + ad
        loso_labels = [0] * len(controls) + [1] * len(ad)
        
        self.splits = []
        for i in range(len(loso_subjects)):
            train_subs = loso_subjects[:i] + loso_subjects[i+1:]
            test_sub = [loso_subjects[i]]
            self.splits.append({
                "train_subjects": train_subs,
                "test_subjects": test_sub,
                "val_idx": i # Index of the subject left out
            })
        
        print(f"Generated {len(self.splits)} LOSO cross-validation folds.")

    def run(self):
        """Execute the pipeline steps."""
        if self.df is None:
            self.initialize()
            
        if self.df is None or self.df.empty or not self.splits:
            return

        if DetachEnsemble is None:
            print("Detach ROCKET is not available. Aborting.")
            return

        print("\nStarting LOSO Cross-Validation Loop...")
        subject_predictions = []
        subject_true_labels = []
        
        results = []

        for i, split in enumerate(self.splits):
            print(f"\n--- Fold {i+1}/{len(self.splits)} (Testing subject: {split['test_subjects'][0]}) ---")
            
            # Load and format data for this split
            max_trials = self.config.get('experiment', {}).get('max_trials_per_subject')
            X_train, y_train, _ = load_and_format_data(self.df, split['train_subjects'], max_trials_per_subject=max_trials)
            X_test, y_test, _ = load_and_format_data(self.df, split['test_subjects'], max_trials_per_subject=max_trials)
            
            if X_train.size == 0 or X_test.size == 0:
                print(f"Skip Fold {i+1}: Insufficient data.")
                continue

            print(f"Train Trials: {X_train.shape[0]}, Test Trials: {X_test.shape[0]}")

            # Initialize and train DetachEnsemble
            model_params = self.config.get('model', {}).get('params', {})
            model = DetachEnsemble(
                num_kernels=model_params.get('num_kernels', 10000),
                num_models=model_params.get('num_models', 10)
            )
            
            model.fit(X_train, y_train)
            
            # Prediction on trials
            y_pred_trials = model.predict(X_test)
            
            # Subject-level prediction via Majority Voting
            # (Since there is only one subject in X_test, we take the mode of y_pred_trials)
            y_pred_subject = 1 if np.mean(y_pred_trials) >= 0.5 else 0
            y_true_subject = y_test[0] # All trials for this subject have same label
            
            subject_predictions.append(y_pred_subject)
            subject_true_labels.append(y_true_subject)
            
            results.append({
                "subject": split['test_subjects'][0],
                "y_true": int(y_true_subject),
                "y_pred": int(y_pred_subject),
                "trial_accuracy": float(accuracy_score(y_test, y_pred_trials))
            })

            print(f"Subject Prediction: {y_pred_subject} (True: {y_true_subject})")

        if subject_predictions:
            final_acc = accuracy_score(subject_true_labels, subject_predictions)
            print(f"\nFinal Subject-Level Accuracy (LOSO): {final_acc*100:.2f}%")
            print("\nClassification Report:")
            print(classification_report(subject_true_labels, subject_predictions, target_names=['Control', 'AD']))
            
            # Save results to file
            os.makedirs("results", exist_ok=True)
            with open("results/loso_results.json", "w") as f:
                json.dump({
                    "subject_level_accuracy": final_acc,
                    "paper_target_accuracy": 0.8615,
                    "folds": results
                }, f, indent=4)
            print("\nDetailed results saved to results/loso_results.json")


if __name__ == "__main__":
    pipeline = TaskEEGPipeline()
    pipeline.run()
