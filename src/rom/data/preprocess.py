
import pandas as pd
import numpy as np
from pathlib import Path
import re
import struct
import shutil


def process_transient_data(data_dir: Path, output_dir: Path, selected_filenames=None):
    """
    Reads CSV files from data_dir, extracts data, and saves to output_dir.
    
    Args:
        data_dir: Path to the directory containing xresults-*.csv files (e.g., data/raw/Dataset)
        output_dir: Path to save the processed files (e.g., data/processed)
        selected_filenames: Optional list of raw csv filenames to process.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching for CSV files in {data_dir}...")
    # Find all CSV files matching the pattern
    csv_files = list(data_dir.glob("xresult-*.csv"))
    if selected_filenames is not None:
        selected_names = [Path(name).name for name in selected_filenames]
        selected_set = set(selected_names)
        csv_map = {path.name: path for path in csv_files}
        missing = sorted(selected_set - set(csv_map.keys()))
        if missing:
            missing_preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} files from selected_filenames were not found in {data_dir}. "
                f"Examples: {missing_preview}"
            )
        csv_files = [csv_map[name] for name in selected_names if name in csv_map]
    
    if not csv_files:
        print(f"No files found matching pattern 'xresult-*.csv' in {data_dir}")
        return

    # Extract time from filenames using regex
    # Pattern expects xresult-<time>.csv. 
    # Adjust regex if time format is specific (e.g., scientific notation)
    # Example: xresult-0.005.csv -> 0.005
    files_with_time = []
    pattern = re.compile(r"xresult-(.+)\.csv")
    
    for f in csv_files:
        match = pattern.search(f.name)
        if match:
            try:
                time_val = float(match.group(1))
                files_with_time.append((time_val, f))
            except ValueError:
                print(f"Warning: Could not parse time from filename {f.name}")
    
    # Sort files by time
    files_with_time.sort(key=lambda x: x[0])
    
    if not files_with_time:
        print("No valid CSV files found.")
        return

    times = [t for t, f in files_with_time]
    sorted_files = [f for t, f in files_with_time]
    
    print(f"Found {len(sorted_files)} time steps.")

    # Read first file to get dimensions and coordinates
    print(f"Reading first file: {sorted_files[0].name}")
    first_df = pd.read_csv(sorted_files[0])
    
    # Expected columns: nodenumber, x-coordinate, y-coordinate, z-coordinate, x-velocity, y-velocity, z-velocity, temperature
    # Clean column names (strip whitespace)
    first_df.columns = first_df.columns.str.strip()
    
    # Check if required columns exist
    required_cols = ['x-coordinate', 'y-coordinate', 'z-coordinate', 'x-velocity', 'y-velocity', 'z-velocity', 'temperature']
    if not all(col in first_df.columns for col in required_cols):
        print(f"Error: Missing required columns in CSV. Found: {first_df.columns}")
        return

    n_nodes = len(first_df)
    n_timesteps = len(sorted_files)
    
    print(f"Nodes: {n_nodes}, Time steps: {n_timesteps}")

    # Initialize arrays
    # Shape: (Nodes, TimeSteps)
    snapshot_T = np.zeros((n_nodes, n_timesteps), dtype=np.float64)
    snapshot_u = np.zeros((n_nodes, n_timesteps), dtype=np.float64)
    snapshot_v = np.zeros((n_nodes, n_timesteps), dtype=np.float64)
    snapshot_w = np.zeros((n_nodes, n_timesteps), dtype=np.float64)

    # Save Coordinates (x, y, z) to points.bin
    # Format: Sequence of floats? Or specific structure?
    # Usually: x1, y1, z1, x2, y2, z2... 
    coords = first_df[['x-coordinate', 'y-coordinate', 'z-coordinate']].values.astype(np.float64) # (N, 3)
    
    # Save as flat binary (x1, y1, z1, x2, y2, z2...) or (x1...xn, y1...yn, z1...zn)?
    # "points.bin" usually implies a simple dump. Let's dump the flattened array (C-order: row by row -> x1,y1,z1, x2,y2,z2...)
    points_path = output_dir / "points.bin"
    coords.tofile(points_path)
    print(f"Saved coordinates to {points_path}")

    # Process all files
    for t_idx, file_path in enumerate(sorted_files):
        if t_idx % 10 == 0:
            print(f"Processing time step {t_idx}/{n_timesteps}: {file_path.name}")
        
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        # Ensure node order is consistent (assuming sorted by nodenumber usually, but let's trust file order is consistent)
        # If needed, can sort by nodenumber: df = df.sort_values('nodenumber')
        
        snapshot_T[:, t_idx] = df['temperature'].values
        snapshot_u[:, t_idx] = df['x-velocity'].values
        snapshot_v[:, t_idx] = df['y-velocity'].values
        snapshot_w[:, t_idx] = df['z-velocity'].values

    # Save Snapshots
    print("Saving snapshot matrices...")
    np.save(output_dir / "Snapshot_T.npy", snapshot_T)
    np.save(output_dir / "Snapshot_u.npy", snapshot_u)
    np.save(output_dir / "Snapshot_v.npy", snapshot_v)
    np.save(output_dir / "Snapshot_w.npy", snapshot_w)
    
    # Save Time steps (DOE)
    doe_path = output_dir / "doe.csv"
    doe_df = pd.DataFrame({'time': times})
    doe_df.to_csv(doe_path, index=False)
    print(f"Saved time steps to {doe_path}")
    
    print("Preprocessing completed.")

if __name__ == "__main__":
    # Default paths (relative to project root if run from there)
    # Assumes script is run from project root, or we need to find project root
    # Using relative paths as requested
    
    # Try to locate data dir relative to current working directory
    # If run from src/rom/data, we need to go up. 
    # Best practice: Assume run from Project Root.
    
    raw_data_dir = Path("data/raw/Dataset") 
    processed_data_dir = Path("data/processed")
    
    if not raw_data_dir.exists():
        # Fallback for flexibility
        raw_data_dir_alt = Path("data/raw")
        if (raw_data_dir_alt / "Dataset").exists():
            raw_data_dir = raw_data_dir_alt / "Dataset"
        else:
            print(f"Warning: Default data directory {raw_data_dir} does not exist.")
    
    process_transient_data(raw_data_dir, processed_data_dir)
