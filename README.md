# Neural Network Engine in C++ from Scratch

A lightweight, high-performance Deep Learning inference and training engine written in pure **C++** with zero external library dependencies (no BlaS, no Eigen, no Vectors-of-Vectors). 

This project was engineered from a systems-level perspective, prioritizing contiguous memory layouts and cache-locality—making it structurally optimized for resource-constrained environments, microcontrollers and embedded applications.


## 🚀 Key Architectural Features

*   **Cache-Friendly Contiguous Memory:** Matrices are flattened into a single-dimensional `std::vector<float>` using a Row-Major memory layout (r * cols + c), eliminating the pointer-chasing and heap-fragmentation overhead common in nested structures like `vector<vector<float>>`.

*   **Modular Component Layout:** Mimics industry-standard frameworks (like PyTorch) by abstracting mathematical tensor operations into a custom `Matrix` layer, and execution steps into a discrete `Layer` class.

*   **Deterministic Network State:** Designed to minimize dynamic memory (`malloc`/`new`) execution inside the forward and backward loops.

*   **Calculus-Driven Optimization:** Includes custom hand-rolled implementations of Stochastic Gradient Descent (SGD), Forward Propagation, Mean Squared Error (MSE) Loss Tracking, and Backpropagation using the Chain Rule.

------

## 📐 Mathematical Stack & Layout

The network implements a modular **Dense Connection Layout** (Y = XW + b):

*   **Activation:** Sigmoid Function {S(x) = 1/(1 + e^-x)}
*   **Derivative Optimization:** Inlined derivative computations evaluated straight from cached activation states    (S'(x) = S(x).(1 - S(x))).
*   **Loss Tracking:** Mean Squared Error (MSE) Gradient Evaluation.

------

### Structural Flow Mapping

[Input Layer (1x2)] -->
(Matrix Multiply & Bias Addition) --> [Hidden Dense Layer (1x3)] --> [Sigmoid Activation Map] -->
 (Matrix Multiply & Bias Addition) --> [Output Dense Layer (1x1)] --> [Sigmoid Activation Map] --> [Loss Analysis]

-----

## 🛠️ Build and Execution Guide

```bash
make

./neuralNetworkCpp
```

------

## 📊 Verification Target: Non-Linear XOR Gate

A single-layer perceptron cannot solve non-linearly separable problems like the XOR logic function. This project utilizes a **2-3-1** Network Topology to verify that multi-layer backpropagation successfully forces convergence.

```bash
Training the network

Epoch:0
Output for input:0 0 
is :0.485158 
Output for input:0 1 
is :0.438529 
Output for input:1 0 
is :0.485509 
Output for input:1 1 
is :0.557515 

Epoch:500
Output for input:0 0 
is :0.200364 
Output for input:0 1 
is :0.892078 
Output for input:1 0 
is :0.343648 
Output for input:1 1 
is :0.508838 

Epoch:1000
Output for input:0 0 
is :0.0431387 
Output for input:0 1 
is :0.949023 
Output for input:1 0 
is :0.929422 
Output for input:1 1 
is :0.0760103 

Epoch:1500
Output for input:0 0 
is :0.024685 
Output for input:0 1 
is :0.965694 
Output for input:1 0 
is :0.958209 
Output for input:1 1 
is :0.0475817

```