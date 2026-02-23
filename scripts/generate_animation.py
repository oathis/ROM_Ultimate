import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT_DIR

from rom.runners.online_prediction import OnlinePredictionRunner

def generate_animation(
    start_time,
    end_time,
    frames,
    output_file,
    models_dir,
    processed_dir,
    variable="u",
    point_size=2.0,
    skip=0,
    mode_name="pod",
    trainer_name="rbf",
):
    """
    Generate an animation of the flow field.
    ...
    """
    print(f"Initializing Animation Generator...")
    # ... (loading runner and points) ...
    runner = OnlinePredictionRunner(models_dir=models_dir, mode_name=mode_name, trainer_name=trainer_name)
    
    # Load points
    points_path = processed_dir / "points.bin"
    if not points_path.exists():
        print(f"Error: Points file not found at {points_path}")
        return

    # Read points directly
    coords = np.fromfile(points_path, dtype=np.float64).reshape(-1, 3)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    
    # ... (time generation and figure setup) ...
    times = np.linspace(start_time, end_time, frames)
    print(f"Generating {frames} frames from T={start_time} to T={end_time}...")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    print("Computing Frame 0...")
    df = runner.step(times[0])
    
    # ... (data extraction) ...
    if variable == 'velocity':
        u = df['u'] if 'u' in df else df.get('x-velocity', np.zeros(len(df)))
        v = df['v'] if 'v' in df else df.get('y-velocity', np.zeros(len(df)))
        w = df['w'] if 'w' in df else df.get('z-velocity', np.zeros(len(df)))
        val = np.sqrt(u**2 + v**2 + w**2)
        label = 'Velocity Magnitude (m/s)'
    else:
        col_name = variable
        if variable == 'T': col_name = 'temperature'
        elif variable == 'u': col_name = 'x-velocity'
        elif variable == 'v': col_name = 'y-velocity'
        elif variable == 'w': col_name = 'z-velocity'
        
        if col_name not in df:
             print(f"Error: Variable {variable} (col: {col_name}) not found in prediction.")
             return
             
        val = df[col_name].values
        label = f'{variable}'

    # Subsampling
    if skip == 0:
        skip = 1
        if len(x) > 10000:
            skip = len(x) // 5000 
            print(f"Auto-subsampling point cloud by factor {skip} for performance. Use --skip 1 to force all points.")
    else:
        print(f"Using manual subsampling factor: {skip}")

    # Color limits
    vmin, vmax = np.min(val), np.max(val)
    vmax = vmax * 1.2 if vmax > 0 else 1.0
    
    scat = ax.scatter(x[::skip], y[::skip], z[::skip], c=val[::skip], cmap='jet', s=point_size, vmin=vmin, vmax=vmax)
    
    # Labels and Colorbar
    cb = fig.colorbar(scat, label=label, shrink=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    title = ax.set_title(f"Time: {times[0]:.6f} s")
    
    def update(frame):
        t_current = times[frame]
        
        # Predict
        df_step = runner.step(t_current)
        
        if variable == 'velocity':
            u = df_step['u'] if 'u' in df_step else df_step.get('x-velocity', np.zeros(len(df_step)))
            v = df_step['v'] if 'v' in df_step else df_step.get('y-velocity', np.zeros(len(df_step)))
            w = df_step['w'] if 'w' in df_step else df_step.get('z-velocity', np.zeros(len(df_step)))
            val_step = np.sqrt(u**2 + v**2 + w**2)
        else:
            col_name = variable
            if variable == 'T': col_name = 'temperature'
            elif variable == 'u': col_name = 'x-velocity'
            elif variable == 'v': col_name = 'y-velocity'
            elif variable == 'w': col_name = 'z-velocity'
            val_step = df_step[col_name].values
            
        scat.set_array(val_step[::skip])
        title.set_text(f"Time: {t_current:.6f} s")
        
        if frame % 10 == 0:
            print(f"Processed frame {frame}/{frames}")
            
        return scat, title
    
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=200, blit=False)
    
    print(f"Saving to {output_file}...")
    try:
        if str(output_file).endswith('.mp4'):
            writer = animation.FFMpegWriter(fps=10, bitrate=1800)
            ani.save(output_file, writer=writer)
        elif str(output_file).endswith('.gif'):
            writer = animation.PillowWriter(fps=10)
            ani.save(output_file, writer=writer)
        else:
            ani.save(output_file, fps=10)
        print("Done!")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Ensure ffmpeg is installed for .mp4, or try .gif")

def main():
    parser = argparse.ArgumentParser(description="Generate Flow Animation from ROM")
    parser.add_argument("--start", type=float, required=True, help="Start time")
    parser.add_argument("--end", type=float, required=True, help="End time")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames")
    parser.add_argument("--output", type=str, default="animation.mp4", help="Output filename")
    parser.add_argument("--variable", type=str, default="u", help="Variable: T/u/v/w/velocity")
    parser.add_argument("--point-size", type=float, default=2.0, help="Point size for scatter plot")
    parser.add_argument("--skip", type=int, default=0, help="Subsampling factor (0 for auto)")
    parser.add_argument("--models-dir", type=str, default=str(ROOT_DIR / "models"), help="Root model directory")
    parser.add_argument("--processed-dir", type=str, default=str(ROOT_DIR / "data/processed"), help="Processed data directory")
    parser.add_argument("--mode", type=str, default="pod", help="Mode artifact namespace")
    parser.add_argument("--trainer", type=str, default="rbf", help="Trainer artifact namespace")
    
    args = parser.parse_args()
    
    generate_animation(
        start_time=args.start,
        end_time=args.end,
        frames=args.frames,
        output_file=args.output,
        models_dir=Path(args.models_dir),
        processed_dir=Path(args.processed_dir),
        variable=args.variable,
        point_size=args.point_size,
        skip=args.skip,
        mode_name=args.mode,
        trainer_name=args.trainer,
    )

if __name__ == "__main__":
    main()
