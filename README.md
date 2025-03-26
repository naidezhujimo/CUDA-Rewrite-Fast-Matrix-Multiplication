# CUDA Rewrite Fast Matrix Multiplication

This repository contains an optimized implementation of matrix multiplication using CUDA. The goal of this project is to provide a high-performance solution for matrix multiplication operations on NVIDIA GPUs.

## Features

- Chunked matrix multiplication
- Thread block memory sharing
- Double buffering
- Vectorised memory access
- Register optimisation
- Loop expands

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- CMake (for building)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/naidezhujimo/CUDA-rewrite-Fast-MatrixMul.git
   ```

2. Create a build directory and compile the project:
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```

### Usage

To perform matrix multiplication, you can use the following command:
```
./matrixmul <matrix_size>
```
Replace `<matrix_size>` with the desired size of the matrices (e.g., 1024 for a 1024x1024 matrix).

## Example

For example, to multiply two 2048x2048 matrices:
```
./matrixmul 2048
```

## Code Structure

- `matrixmul.cu`: Main CUDA implementation file
- `CMakeLists.txt`: Build configuration file
- `README.md`: This documentation file

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for any improvements or bug fixes.

## Acknowledgments

- NVIDIA CUDA documentation
- Various online resources and tutorials on CUDA programming
