# Boxer Windows Setup Guide

This guide documents the steps to set up and run Boxer on Windows 11 with Python 3.13.

## Prerequisites

- Windows 11
- Python 3.13 (installed via python.org installer)
- Git (for cloning the repo)

## Installation Steps

### 1. Clone the Boxer Repository

```bash
git clone https://github.com/facebookresearch/boxer.git
cd boxer
```

### 2. Install Python Dependencies

```powershell
# Core dependencies
python -m pip install torch numpy opencv-python tqdm dill moderngl moderngl-window imgui-bundle

# Note: projectaria-tools requires Python <=3.11, so we skip Aria data for now
```

### 3. Download Model Checkpoints

Created a Windows PowerShell script `scripts/download_ckpts.ps1` to download checkpoints:

```powershell
.\scripts\download_ckpts.ps1
```

Downloads:
- BoxerNet checkpoint (1.2 GB)
- DinoV3 checkpoint (350 MB)  
- OWLv2 checkpoint (550 MB)

### 4. Download Sample Data

For Aria data (requires Python 3.11):
- Created `scripts/download_aria_data.ps1` for Windows

For CA-1M data (works with Python 3.13):
```powershell
python scripts/download_ca1m_sample.py
```

This downloads a sample sequence to `sample_data/ca1m-val-42898570/`.

### 5. Install FFmpeg for Video Output

```powershell
winget install ffmpeg
```

Added to PATH:
```powershell
$env:PATH += ";C:\Users\[username]\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"
```

### 6. Code Fixes for Windows

Updated `utils/video.py`:
- Fixed `grep` command (Unix-only) to use Python string parsing
- Added Windows ffmpeg search path for winget installations

## Running Boxer

### On CA-1M Data

```powershell
python run_boxer.py --input ca1m-val-42898570 --max_n=10 --track
```

**Results:**
- Processes 10 frames
- Detects 17 2D bounding boxes
- Lifts 12 to 3D oriented bounding boxes
- Saves to CSV files:
  - `boxer_3dbbs.csv` (3D boxes)
  - `owl_2dbbs.csv` (2D boxes)
  - `boxer_3dbbs_tracked.csv` (tracked 3D boxes)

### On Aria Data (Requires Python 3.10/3.11)

```powershell
# Install Python 3.10.11 from https://www.python.org/downloads/release/python-31011/
# Create venv with Python 3.10
py -3.10 -m venv boxer310
.\boxer310\Scripts\Activate.ps1
python -m pip install torch numpy opencv-python tqdm dill moderngl moderngl-window imgui-bundle
python -m pip install projectaria-tools

# Download Aria data
.\scripts\download_aria_data.ps1

# Run
python run_boxer.py --input nym10_gen1 --max_n=90 --track
```

## Issues Encountered & Solutions

1. **projectaria-tools not available for Python 3.13+**
   - Solution: Use Python 3.10 for Aria, or skip to CA-1M/SUN-RGBD

2. **Checkpoint downloads failing**
   - Solution: Added retry logic in download script

3. **FFmpeg not in PATH**
   - Solution: Added winget install path to PATH

4. **Unix commands in Windows**
   - Solution: Replaced `grep` with Python parsing in video.py

5. **Video creation failing**
   - Minor issue: Images saved but glob pattern incorrect (CSVs still work)

## Current Status

✅ Boxer runs successfully on Windows with CA-1M data
✅ 3D object detection and tracking working
✅ CSV outputs generated
⚠️ Video output has minor path issues (non-critical)

## Next Steps

- Fix video output path issue
- Test on SUN-RGBD data
- Test interactive viewer scripts
- Consider contributing Windows fixes back to the repo