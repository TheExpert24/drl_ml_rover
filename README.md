# DRL_ML_Rover

DRL_ML_Rover is a project that the Nixon Incorporated Aeronautic Administration has been working on. Inspired by the United States vs. China space race, we have been working on a rover fully controlled by a neural network.

## Technical Info

Coded fully in Python, the neural network currently uses gymnasium for simulation, Stable-Baselines for deep reinforcement learning, PyTorch for the framework, and a proximal policy optimization that trains the neural network over 500,000 steps.

## Performance

Currently, the latest version (v6) has shown a lot of promise, completing 200 evaluated missions with a 100% success rate and a 90% path efficiency rate.

## Simulation

<img width="725" height="665" alt="Screenshot_20261118_174356" src="https://github.com/user-attachments/assets/8f67840f-0a58-4247-8d79-7521b7ec5fa6" />

## Environment

The environment is a 500x500 grid with 600 randomly placed obstacles, a base, a rover, and a goal, where the rover has to reach its goal and return to the base in the most efficient path possible.

## Neural Network

The neural network receives a 15-dimensional observation vector that contains the target direction, front/left/right proximity sensor readings, normalized target distance, battery, base direction and distance, and estimated energy consumption.
