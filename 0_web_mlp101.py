import streamlit as st
import pandas as pd
import numpy as np

# Title of the app
st.title("Interactive Neural Network Visualizer with Data Dimensions")

# Introduction
st.write("""
This app allows you to visualize the structure of a multi-layer perceptron (MLP) and understand how backpropagation works.
You can specify the input dimensions, number of hidden layers, and the number of neurons per layer.
The output layer will use a sigmoid activation for binary classification.
""")

# Input for the dimensions of the data
input_samples = st.number_input("Number of Data Samples", min_value=1, value=2)
input_features = st.number_input("Number of Input Features", min_value=1, value=3)


# Input for the number of hidden layers and their neurons
num_layers = st.number_input("Number of Hidden Layers", min_value=1, max_value=10, value=2)
neurons_per_layer = []

for i in range(num_layers):
    neurons = st.number_input(f"Number of Neurons in Hidden Layer {i+1}", min_value=1, value=4)
    neurons_per_layer.append(neurons)

# Output layer with a fixed sigmoid activation and a single neuron
output_neurons = 1
output_activation = "Sigmoid"

# Calculate the dimensions of weights, biases, and activations
def generate_network_structure(input_features, neurons_per_layer, output_neurons):
    structure = []
    input_size = input_features

    # Add input layer information
    structure.append({
        "Layer": "Input Layer",
        "Weights Shape": "N/A",
        "Biases Shape": "N/A",
        "Activations Shape": f"({input_samples}, {input_features})"
    })

    # Iterate through each layer to create the structure
    for idx, neurons in enumerate(neurons_per_layer):
        weights_shape = (input_size, neurons)
        biases_shape = (1, neurons)
        activations_shape = (input_samples, neurons)
        structure.append({
            "Layer": f"Hidden Layer {idx+1}",
            "Weights Shape": f"{weights_shape}",
            "Biases Shape": f"{biases_shape}",
            "Activations Shape": f"{activations_shape}"
        })
        input_size = neurons

    # Output layer
    weights_shape = (input_size, output_neurons)
    biases_shape = (1, output_neurons)
    activations_shape = (input_samples, output_neurons)
    structure.append({
        "Layer": "Output Layer (Sigmoid)",
        "Weights Shape": f"{weights_shape}",
        "Biases Shape": f"{biases_shape}",
        "Activations Shape": f"{activations_shape}"
    })

    return pd.DataFrame(structure)

# Generate the structure
if st.button("Generate Network Structure"):
    network_structure = generate_network_structure(input_features, neurons_per_layer, output_neurons)
    st.write("### Neural Network Structure")
    st.dataframe(network_structure)
    st.write("""
    Each layer shows the dimensions of the weight matrix, bias vector, and the resulting activations.
    - **Input Layer**: The shape of the input data.
    - **Weights Shape** represents the connections between layers.
    - **Biases Shape** represents the bias term added to each neuron's output.
    - **Activations Shape** represents the output of each layer for the given number of samples.
    """)

# Explanation of Backpropagation
st.header("Understanding Backpropagation")
st.write("""
### How Does Backpropagation Work?
Backpropagation is the process by which a neural network learns from its mistakes. Here’s a simplified explanation:

1. **The Network Makes a Prediction**:
    - **Input**: The input data (e.g., hours of study and hours of sleep) is fed into the network.
    - **Passing Through Layers**: 
        - Each layer in the network consists of **neurons**.
        - The neurons multiply the input by their respective **weights**, add a **bias**, and apply an **activation function**.
        - The output from each layer becomes the input for the next.
        - **Output Layer**: The final layer applies a sigmoid activation to produce a prediction (e.g., a probability).
2. **Comparison with Reality**:
    - The network compares its **prediction** with the **true label** using a **loss function**.
    - **Example**: If the network predicts a 60% chance of a tumor being malignant, but the reality is that it's benign, the loss measures how far this prediction is from the truth.
    - **Formula**: Binary Cross-Entropy Loss.
3. **Identifying Where It Went Wrong (Backpropagation)**:
    - **Calculate each neuron's responsibility**: Backpropagation figures out how much each **neuron**, **weight**, and **bias** contributed to the error.
    - It starts from the **last layer** and moves backward, adjusting weights and biases.
    - **Gradients**: Calculate how the error changes with respect to each weight and bias.
4. **Adjusting Weights and Biases**:
    - Using the gradients, the network adjusts its weights and biases slightly to reduce the error.
    - **Learning Rate**: Controls how big each adjustment is.
    - This process is repeated for many **epochs** (training cycles).

### Mathematical Operations Summary
- **Forward Pass**: 
    - \\( Z = X \\cdot W + b \\)
    - \\( A = \sigma(Z) \\) for the output layer.
- **Loss Calculation**:
    - \\( \text{Loss} = -\\frac{1}{N} \\sum(y \\cdot \\log(\\hat{y}) + (1 - y) \\cdot \\log(1 - \\hat{y})) \\)
- **Backpropagation (Gradient Calculation)**:
    - \\( dW = A^{\text{previous}} \\cdot dZ^T \\)
    - \\( db = dZ \\)
    - Update: \\( W_{new} = W_{old} - \text{learning rate} \\cdot dW \\)
""")

# Footer
st.write("""
This interactive tool helps to understand both the structure of a neural network and how it learns through backpropagation.
Use the input controls to see how different architectures affect the dimensions of the data flowing through the network.
""")