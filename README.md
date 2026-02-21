# Heat Conduction Simulation (C++)

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

## Build Instructions

### Prerequisites
- C++17-capable compiler
- CMake

### Build (out-of-source recommended)

```bash
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

---

## Sample Output

Below is a sample output, plotted in excel from the CSV log.

<img width="465" height="274" alt="image" src="https://github.com/user-attachments/assets/49a4d287-162d-4304-8e9f-eb2c543443dc" />

