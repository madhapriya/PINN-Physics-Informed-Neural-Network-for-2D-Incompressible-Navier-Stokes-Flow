# Physics-Informed Neural Network (PINN) for 2D Incompressible Navier–Stokes Flow

## 📌 Project Overview

This project implements a **Physics-Informed Neural Network (PINN)** to model and predict **two-dimensional incompressible fluid flow** governed by the **Navier–Stokes equations**.

Unlike traditional machine learning models that learn purely from data, this approach embeds **physical laws directly into the neural network training process**, ensuring that predictions remain **physically consistent** while still benefiting from the flexibility of deep learning.

The project is developed in **MATLAB** and trained using simulation data of **flow past a cylinder**. The network predicts:

* Velocity components (**u, v**)
* Pressure field (**p**)
* Fluid viscosity (**ν**) as a learnable physical parameter

---

## ❓ Problem Statement

Solving the **Navier–Stokes equations** using traditional numerical techniques such as:

* Finite Difference Method (FDM)
* Finite Element Method (FEM)
* Finite Volume Method (FVM)

is computationally expensive and time-consuming, especially for complex geometries and long-time simulations.

Purely data-driven neural networks, on the other hand:

* Ignore physical constraints
* Require large labeled datasets
* Can produce physically invalid solutions

### 🔴 Challenge

How can we combine **physics knowledge** with **machine learning** to build a model that is:

* Accurate
* Data-efficient
* Physically valid
* Computationally efficient

---

## 💡 Proposed Solution: Physics-Informed Neural Networks (PINNs)

Physics-Informed Neural Networks integrate **governing physical equations** into the neural network loss function.

In this project:

* The **Navier–Stokes equations** and **continuity equation** are enforced during training
* The network learns velocity and pressure fields
* Physical constraints act as a regularizer
* The model generalizes well even with limited data

---

## ⚙️ Methodology

### 1️⃣ Data Preparation

The dataset represents **2D incompressible flow past a cylinder** and contains:

* Spatial coordinates: (x, y)
* Time values: (t)
* Velocity components: (u, v)
* Pressure values: (p)

Steps performed:

* Combine spatial and temporal data into training samples
* Split data into training and testing sets
* Normalize inputs and outputs to improve numerical stability

---

### 2️⃣ Neural Network Architecture

The model uses a **fully connected deep neural network**:

* **Input:** (x, y, t)
* **Output:** (u, v, p)
* Multiple hidden layers with **tanh activation**
* Designed to approximate smooth physical fields

---

### 3️⃣ Physics-Informed Loss Function

The total loss function consists of:

#### 🔹 Data Loss

Ensures predictions match known simulation data.

#### 🔹 Physics Loss

Enforces the governing equations:

* Momentum equations (Navier–Stokes)
* Continuity equation (incompressibility)

#### 🔹 Parameter Learning

Fluid viscosity (**ν**) is treated as a trainable parameter and learned directly from data.

This ensures that the network learns **both the solution and the underlying physics**.

---

### 4️⃣ Training Strategy

Training is performed in **two stages**:

#### Stage 1: Adam Optimizer

* Fast convergence
* Learns coarse solution structure
* Handles large datasets efficiently

#### Stage 2: L-BFGS Optimizer

* Second-order optimization
* Fine-tunes the solution
* Improves physical accuracy and convergence

---

### 5️⃣ Model Evaluation

After training:

* Predictions are generated on unseen test data
* Results are visualized for:
  * Velocity magnitude
  * Velocity vector fields
  * Pressure field
  * Vorticity error
* Model accuracy is evaluated using **R² score**

---

## 📊 Results

* Accurate prediction of velocity and pressure fields
* Physically consistent solutions
* Learned viscosity value close to expected physical value
* High R² scores for u, v, and p
* Stable and smooth flow field reconstruction

---

## 🌍 Applications

* Computational Fluid Dynamics (CFD)
* Aerospace and aerodynamic simulations
* Climate and weather modeling
* Biomedical fluid flow analysis
* Physics-based digital twins
* Scientific Machine Learning (SciML)

---

## 🎓 Learning Outcomes

* Understanding Navier–Stokes equations
* Applying deep learning to physical systems
* Automatic differentiation for PDEs
* Hybrid optimization techniques (Adam + L-BFGS)
* Scientific visualization and validation

---

## ⚖️ Academic Notice

This project is developed **for academic and educational purposes**.
It demonstrates the application of Physics-Informed Neural Networks for solving partial differential equations and is intended for learning, experimentation, and research demonstration.

---

## 👤 Author

**Madha Priya**  
B.Tech – Computer Science & Engineering  
Final Year Project

---

## ⭐ Acknowledgment

Inspired by open-source research and academic work in the field of **Physics-Informed Neural Networks (PINNs)**.

---

## 🚀 Getting Started

### Prerequisites

* MATLAB R2020b or later
* Deep Learning Toolbox
* Optimization Toolbox

### Usage

1. Clone this repository
2. Load the dataset from `data/cylinder_nektar_wake.mat`
3. Run `PINN_Train_NN.m` to train the model
4. Run `PINN_Visualization.m` to visualize results

---

## 📚 References

For more information on Physics-Informed Neural Networks, refer to the foundational research in scientific machine learning and physics-based deep learning.

---

**⭐ If you find this project helpful, please consider giving it a star!**
