import numpy as np
import math
import os
from futhark_data import dump


def generate_dataset(n_points, filename):
    print(f"Generating {n_points} points for {filename}...")

    # 1. Define Scalars (Must match Futhark types explicitly)
    # W and H are i64 in Futhark
    W = np.int64(800)
    H = np.int64(600)

    # Camera params are f32 in Futhark
    fx = np.float32(800.0)
    fy = np.float32(800.0)
    cx = np.float32(W / 2.0)
    cy = np.float32(H / 2.0)

    # Rotation (Identity) - f32
    cq_w = np.float32(1.0)
    cq_x = np.float32(0.0)
    cq_y = np.float32(0.0)
    cq_z = np.float32(0.0)

    # Translation - f32
    ct_x = np.float32(0.0)
    ct_y = np.float32(0.0)
    ct_z = np.float32(-5.0)

    # 2. Uniform Cube Generation
    side = int(np.ceil(n_points ** (1 / 3)))
    linspace = np.linspace(-1.0, 1.0, side).astype(np.float32)
    x, y, z = np.meshgrid(linspace, linspace, linspace)

    xyz_x = x.flatten()[:n_points]
    xyz_y = y.flatten()[:n_points]
    xyz_z = z.flatten()[:n_points]

    # 3. Attributes (All arrays must be float32)
    opas = np.zeros(n_points, dtype=np.float32)

    s_x = np.full(n_points, math.log(0.01), dtype=np.float32)
    s_y = np.full(n_points, math.log(0.01), dtype=np.float32)
    s_z = np.full(n_points, math.log(0.01), dtype=np.float32)

    rot_w = np.ones(n_points, dtype=np.float32)
    rot_x = np.zeros(n_points, dtype=np.float32)
    rot_y = np.zeros(n_points, dtype=np.float32)
    rot_z = np.zeros(n_points, dtype=np.float32)

    c_r = (xyz_x + 1.0) / 2.0
    c_g = (xyz_y + 1.0) / 2.0
    c_b = (xyz_z + 1.0) / 2.0

    sh_r = np.zeros((n_points, 15), dtype=np.float32)
    sh_g = np.zeros((n_points, 15), dtype=np.float32)
    sh_b = np.zeros((n_points, 15), dtype=np.float32)

    # 4. Write to binary file
    with open(filename, "wb") as f:
        # Pass the numpy scalars directly
        args = [
            W,
            H,
            fx,
            fy,
            cx,
            cy,
            cq_w,
            cq_x,
            cq_y,
            cq_z,
            ct_x,
            ct_y,
            ct_z,
            xyz_x,
            xyz_y,
            xyz_z,
            opas,
            s_x,
            s_y,
            s_z,
            rot_w,
            rot_x,
            rot_y,
            rot_z,
            c_r,
            c_g,
            c_b,
            sh_r,
            sh_g,
            sh_b,
        ]

        for arg in args:
            dump(arg, f)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    generate_dataset(100_000, "data/100k.in")
    generate_dataset(500_000, "data/500k.in")
    generate_dataset(1_000_000, "data/1000k.in")
    generate_dataset(1_500_000, "data/1500k.in")
    generate_dataset(2_000_000, "data/2000k.in")
    print("Data generation complete.")
