# 🔥 Heat Conduction Simulation (C++)

A simple 2D heat conduction / diffusion simulation written in C++. The model iteratively relaxes a temperature field on a square plate with fixed boundary conditions and exports the resulting temperature grid to a CSV file for analysis/plotting.

---

## Overview

This repository contains:

- A CPU-based 2D heat diffusion simulation on a square grid
- A plate model with fixed-temperature edges and an internal heat source region
- An iterative update loop that relaxes interior points using neighbor averaging
- CSV export of the final temperature field (`finalTemperatures.csv`)

---

## Features

- Grid-based plate model with configurable resolution
- Fixed boundary conditions (edges held at a constant temperature)
- Heat source region (filament) with a higher fixed temperature
- Iterative relaxation update for interior cells (4-neighbor average)
- Output to CSV for plotting in Python/Excel/Matlab/etc.

---

## Tech Stack

- **Language:** C++
- **Build:** CMake
- **Output:** CSV (`finalTemperatures.csv`)

---

## Repository Layout

Current structure:

- `lab4/` — simulation source, headers, and CMake project

> If you want to make this repo more portfolio-friendly, consider renaming `lab4/` to something like `src/` or `heat_conductor/`, and updating the CMake project name.

---

## Build Instructions

### Prerequisites
- C++17-capable compiler
- CMake

### Build (out-of-source recommended)

```bash
cd lab4
mkdir build
cd build
cmake ..
cmake --build .
```

--- 

## Usage 

Run the simulation executable from the build output directory. Optionally, you can provide an argument for the number of 
relaxation iterations (default: 10000).

```bash
./main -I 20000
```

After completion, the program writes to finalTemperatures.csv, with each row of the csv corresponding to one row of the plate
grid.

## Sample Output

