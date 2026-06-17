import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main():
    """Print basic statistics of an operation trajectory."""
    base = Path("results/retargeting")
    npz_path = base / "mock_operation_trajectory.npz"
    csv_path = base / "mock_operation_trajectory.csv"

    if not npz_path.exists() and not csv_path.exists():
        print(f"Error: no trajectory file found in {base}")
        sys.exit(1)

    if npz_path.exists():
        data = dict(np.load(npz_path, allow_pickle=True))
        n = len(data["step"])
        total_time = float(data["time"][-1] - data["time"][0])
        valid_ratio = float(data["valid"].mean())
        tpos = data["target_pos"]
        gripper = data["gripper_width"]
        source_arr = data.get("source", None)
    else:
        print("CSV fallback: basic stats only.")
        lines = open(csv_path, "r", encoding="utf-8").readlines()
        n = len(lines) - 1
        total_time = 0.0
        valid_ratio = 0.0
        tpos = np.zeros((n, 3))
        gripper = np.zeros(n)
        source_arr = None

    print(f"Samples:          {n}")
    print(f"Total time [s]:   {total_time:.2f}")
    print(f"Valid ratio:      {valid_ratio:.3f}")
    print(f"Target pos x:     {float(tpos[:,0].min()):.4f} .. {float(tpos[:,0].max()):.4f}")
    print(f"Target pos y:     {float(tpos[:,1].min()):.4f} .. {float(tpos[:,1].max()):.4f}")
    print(f"Target pos z:     {float(tpos[:,2].min()):.4f} .. {float(tpos[:,2].max()):.4f}")
    print(f"Gripper width:    {float(gripper.min()):.4f} .. {float(gripper.max()):.4f}")
    if source_arr is not None and len(source_arr) > 0 and isinstance(source_arr[0], str):
        print(f"Sources:          {', '.join(set(source_arr))}")


if __name__ == "__main__":
    main()
