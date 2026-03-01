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

        print(f"Datasets: {len(controls)} Control, {len(ad)} AD, {len(ftd)} FTD subjects.")
        self.subjects = {"Control": controls, "AD": ad, "FTD": ftd}
        
        self.splits = get_stratified_splits(
            controls, ad, ftd, 
            n_splits=self.n_splits, 
            seed=self.split_seed,
            binary_mode=self.binary_mode
        )
        if self.splits:
            print(f"Generated {len(self.splits)} stratified cross-validation splits.")

    def run(self):
        """Execute the pipeline steps."""
        if self.df is None:
            self.initialize()
            
        if self.df is None or self.df.empty or not self.splits:
            return

        if DetachEnsemble is None:
            print("Detach ROCKET is not available. Aborting.")
            return

        print("\nStarting Cross-Validation Loop...")
        all_accuracies = []
        
        results = []

        for i, split in enumerate(self.splits):
            print(f"\n--- Fold {i+1}/{len(self.splits)} ---")
            
            # Load and format data for this split
            print("Loading and formatting training data...")
            X_train, y_train = load_and_format_data(self.df, split['train_subjects'])
            
            print("Loading and formatting test data...")
            X_test, y_test = load_and_format_data(self.df, split['test_subjects'])
            
            if X_train.size == 0 or X_test.size == 0:
                print(f"Skip Fold {i+1}: Insufficient data.")
                continue

            # X should be (n_instances, n_channels, n_timepoints)
            print(f"Train Shape: {X_train.shape}, Labels: {y_train.shape}")
            print(f"Test Shape: {X_test.shape}, Labels: {y_test.shape}")

            # Initialize and train DetachEnsemble
            model_params = self.config.get('model', {}).get('params', {})
            model = DetachEnsemble(
                num_kernels=model_params.get('num_kernels', 10000),
                num_models=model_params.get('num_models', 10)
            )
            
            print("Training DetachEnsemble...")
            model.fit(X_train, y_train)
            
            print("Predicting...")
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            all_accuracies.append(acc)
            
            # Save detailed results for this fold
            results.append({
                "fold": i + 1,
                "y_true": y_test.tolist(),
                "y_pred": y_pred.tolist(),
                "test_subjects": split['test_subjects'],
                "accuracy": acc
            })

            print(f"Fold {i+1} Accuracy: {acc:.4f}")
            print(classification_report(y_test, y_pred, target_names=['Control', 'Dementia']))

        if all_accuracies:
            avg_acc = np.mean(all_accuracies)
            std_acc = np.std(all_accuracies)
            print(f"\nAverage Cross-Validation Accuracy: {avg_acc:.4f} (+/- {std_acc:.4f})")
            
            # Save results to file
            os.makedirs("results", exist_ok=True)
            with open("results/cv_results.json", "w") as f:
                json.dump({
                    "average_accuracy": avg_acc,
                    "std_accuracy": std_acc,
                    "folds": results
                }, f, indent=4)
            print("\nDetailed results saved to results/cv_results.json")

if __name__ == "__main__":
    pipeline = TaskEEGPipeline()
    pipeline.run()
