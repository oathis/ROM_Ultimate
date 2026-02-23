
import numpy as np
from pathlib import Path

class POD:
    def __init__(self, n_modes=None, energy_threshold=None):
        """
        Proper Orthogonal Decomposition (POD) using SVD.
        
        Args:
            n_modes (int): Number of modes to keep.
            energy_threshold (float): Energy threshold to determine n_modes (0.0 to 1.0).
                                      If provided, n_modes is ignored (unless None).
        """
        self.n_modes = n_modes
        self.energy_threshold = energy_threshold
        self.modes = None
        self.singular_values = None
        self.coefficients = None
        self.mean = None

    def fit(self, X):
        """
        Compute POD modes from snapshot matrix X.
        
        Args:
            X (numpy.ndarray): Snapshot matrix of shape (n_features, n_snapshots).
        """
        # 1. Compute mean and subtract
        self.mean = np.mean(X, axis=1, keepdims=True)
        X_centered = X - self.mean

        # 2. SVD
        # U: (n_features, n_snapshots), S: (n_snapshots,), Vh: (n_snapshots, n_snapshots)
        # Using full_matrices=False for efficiency
        U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)
        
        # 3. Determine number of modes
        total_energy = np.sum(S**2)
        cumulative_energy = np.cumsum(S**2) / total_energy
        
        if self.energy_threshold:
            n_modes_energy = np.searchsorted(cumulative_energy, self.energy_threshold) + 1
            if self.n_modes:
                self.n_modes = min(self.n_modes, n_modes_energy)
            else:
                self.n_modes = n_modes_energy
        elif self.n_modes is None:
            self.n_modes = len(S) # Keep all

        print(f"Selecting {self.n_modes} modes.")
        print(f"Energy preserved: {cumulative_energy[self.n_modes-1]:.4f}")

        # 4. Truncate
        self.modes = U[:, :self.n_modes]
        self.singular_values = S[:self.n_modes]
        
        # Coefficients (Projection)
        # alpha = U^T * X_centered
        # Or from SVD: X = U * S * Vh -> U^T * X = S * Vh
        self.coefficients = np.diag(S[:self.n_modes]) @ Vh[:self.n_modes, :]
        
        return self

    def reconstruct(self, coeffs=None):
        """
        Reconstruct data from coefficients.
        
        Args:
            coeffs (numpy.ndarray): Coefficients of shape (n_modes, n_snapshots). 
                                    If None, uses fitted coefficients.
        Returns:
            Reconstructed X (n_features, n_snapshots)
        """
        if coeffs is None:
            coeffs = self.coefficients
            
        if coeffs is None:
            raise ValueError("No coefficients provided and model not fitted.")

        # X ~ Mean + Phi * Alpha
        X_rec = self.mean + self.modes @ coeffs
        return X_rec

    def save(self, output_dir: Path, prefix="pod"):
        """Save POD model (modes, mean, singular values) to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / f"{prefix}_modes.npy", self.modes)
        np.save(output_dir / f"{prefix}_mean.npy", self.mean)
        np.save(output_dir / f"{prefix}_singular_values.npy", self.singular_values)
        np.save(output_dir / f"{prefix}_coefficients.npy", self.coefficients)
        print(f"Saved POD model to {output_dir}")

def run_pod_on_processed_data(data_dir: Path, output_dir: Path, n_modes=10):
    """
    Run POD on all Snapshot files in data_dir.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # List all snapshot files
    snapshot_files = list(data_dir.glob("Snapshot_*.npy"))
    
    if not snapshot_files:
        print(f"No Snapshotfiles found in {data_dir}")
        return

    for f in snapshot_files:
        print(f"Running POD on {f.name}...")
        variable_name = f.stem.replace("Snapshot_", "") # e.g. T, u, v, w
        
        X = np.load(f)
        print(f"Data shape: {X.shape}")
        
        pod = POD(n_modes=n_modes, energy_threshold=0.999) # Create POD object
        pod.fit(X)
        
        # Save results
        model_dir = output_dir / variable_name
        pod.save(model_dir)

if __name__ == "__main__":
    # Default paths
    processed_dir = Path("data/processed")
    models_dir = Path("models/pod")
    
    if processed_dir.exists():
        run_pod_on_processed_data(processed_dir, models_dir)
    else:
        print(f"Processed data directory {processed_dir} not found.")

