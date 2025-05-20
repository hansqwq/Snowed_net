# Fractal-Based Network Topology (SnowedNet)

This project explores the generation, addressing, routing, and analysis of **SnowedNet**, a novel (N-1)-dimensional network topology derived from an N-simplex fractal. It is designed to be scalable and possess a hierarchical structure, making it suitable for various network applications.

The core of the project lies in the Python implementation which simulates the creation of SnowedNet, assigns unique hierarchical labels and binary addresses (Tier Locators) to its nodes, and implements a routing algorithm that leverages this inherent structure.

## About The Project

This research investigates a new method for constructing network topologies using fractal geometry. Traditional network structures can face limitations in scalability, fault tolerance, and routing efficiency, especially in large-scale distributed systems like those envisioned for smart agriculture or expansive IoT deployments. SnowedNet aims to address these challenges by:

* Employing an iterative generation mechanism to build a self-similar, layered network.
* Utilizing a unique node labeling system based on "sub-label pairs" that encodes a node's generation path and hierarchical level.
* Converting these labels into routable binary addresses (Tier Locators).
* Developing a hierarchical routing algorithm that uses the label structure to find paths efficiently, typically by finding a common ancestor between source and destination nodes.

This implementation allows for the practical exploration of SnowedNet's properties and performance characteristics.

## Features

* **Fractal Network Generation:** Iteratively constructs an (N-1)-dimensional SnowedNet based on user-defined iteration count and dimension (related to the initial N-simplex).
* **Unique Node Addressing:**
    * Assigns a unique label (sequence of integer pairs) to each node reflecting its position in the fractal hierarchy.
    * Converts these intuitive labels into fixed-format binary strings (Tier Locators) for potential use in network protocols.
* **Hierarchical Routing:** Implements a routing algorithm that navigates the network by leveraging the common ancestry encoded in node labels. It handles cases with and without direct common ancestors by utilizing initial layer connectivity.
* **Simulation Modes ():**
    * **Interactive Mode:** Build a custom network, inspect node labels and binary addresses (for initial layers), and perform manual routing between specified nodes.
    * **Simulation Mode:** Automatically run routing tests on networks of varying sizes, collecting data on average routing computation time and path lengths. Includes visualization of these metrics using `matplotlib`.
    * **Fault Simulation Mode:** Simulate random node failures in the network and evaluate the routing algorithm's success rate and performance on the degraded network to assess robustness.
* **Performance Visualization:** Generates plots for network scale, average routing time, and average path hops as a function of iteration count.

## Getting Started

### Prerequisites

* Python 3.x
* The following Python libraries (can be installed via pip):
    * `numpy`
    * `matplotlib`

    ```bash
    pip install numpy matplotlib
    ```

### Files

* `main.py`: The main script containing the `Network` class, network generation logic, routing algorithms, and simulation modes.
* `Functions.py`: Contains helper functions, such as `inputTransform` for formatting node address keys.

## Usage

To run the project, execute the `main.py` script from your terminal:

```bash
python main.py
