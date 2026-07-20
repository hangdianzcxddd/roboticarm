# GraspNet Backend Setup

`checkpoint-rs.tar` is only the trained RealSense checkpoint. The runtime also
needs the official GraspNet baseline source code and its custom operators.

## Expected Layout

Place the official repository here:

```text
models/
  checkpoint-rs.tar
  graspnet-baseline/
    models/graspnet.py
    pointnet2/
    knn/
```

Alternatively, set `GRASPNET_BASELINE_DIR` to the repository path.

## Windows Setup

From the `piper` conda environment:

```powershell
cd D:\git\projects\roboticarm\piper_project\models
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline
python -m pip install -r requirements.txt
cd pointnet2
python setup.py install
cd ..\knn
python setup.py install
```

If compiling `pointnet2` or `knn` fails on Windows, use a CUDA/PyTorch-compatible
Visual Studio Build Tools installation, or build/run this backend in Linux/WSL.

## Run

From `piper_project`:

```powershell
python -m robot_control.windows.client.graspnet_grasp_send_tcp `
  --belt-depth-plane-json models\calibration\belt_depth_plane_1280x720.json `
  --graspnet-checkpoint models\checkpoint-rs.tar `
  --dry-run `
  --show-point-cloud
```

Use `--device cpu` only for import/debug checks. Real inference is expected to be
much faster on CUDA.
