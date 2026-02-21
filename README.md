# GPU-Accelerated 2D Heat Conduction Simulation (C++ / CUDA)

A GPU-accelerated simulation of two-dimensional heat conduction written in C++ with CUDA.  
This project uses NVIDIA’s CUDA platform to offload computation of iterative heat diffusion to the GPU for performance.

---

## Overview

This repository contains:

- A 2D grid simulation of heat diffusion over time  
- A parallel CUDA kernel that updates the temperature field across grid cells  
- Host code that manages data transfer to/from the GPU and iteration control  
- Output written to CSV for analysis or visualization  

This project demonstrates basic use of **CUDA** for parallelizing numerical simulations over a grid. :contentReference[oaicite:1]{index=1}

---

## Features

- GPU acceleration using CUDA C/C++  
- Iterative relaxation method for solving heat diffusion  
- Configurable grid size and iteration count  
- Output of final temperature field to CSV  
- Simple, clean command-line interface

---

## Tech Stack

- Language: C++ / CUDA  
- Parallel Acceleration: NVIDIA CUDA  
- Build System: CMake  
- Output: Temperature grid CSV

> CUDA (Compute Unified Device Architecture) enables general-purpose computation on NVIDIA GPUs, allowing thousands of threads to operate in parallel. :contentReference[oaicite:2]{index=2}

---

## Build Instructions

### Prerequisites

- NVIDIA GPU with CUDA support  
- CUDA Toolkit installed  
- C++ compiler compatible with CUDA (e.g., `nvcc`)  
- CMake

### Build

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

---

## Usage

You can then run the simulation directly from the build output directory via the heat-dispersion.exe. Optional arguments
include the number of grid points and number of iterations.

```bash
./heat_conductor --grid 1024 --iterations 10000
```

The result will be a `finalTemperatures.csv` containing a 2D grid of temperatures.

---

## Sample Output

Below is a sample output, plotted in excel from the CSV log.

<img width="465" height="274" alt="image" src="https://github.com/user-attachments/assets/49a4d287-162d-4304-8e9f-eb2c543443dc" />

---

## Author

Jack Newcomb
