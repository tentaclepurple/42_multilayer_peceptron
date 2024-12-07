import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from utils.MLP import MLPSequential
from utils.preprocess import get_df, set_data_for_model_with_eval

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Project Introduction
st.title("MLP Classifier Interactive Application")

st.image("images/nn.png")

st.write("""
## Multilayer Perceptron Neural Network
This project implements a Multilayer Perceptron (MLP) from scratch, focusing on binary classification. Unlike common approaches that rely on deep learning frameworks like TensorFlow or PyTorch, we've built our neural network by implementing all the fundamental mathematical operations and algorithms manually.

Our implementation includes:
- Complete feedforward propagation, calculating weighted sums and activations through multiple layers
- Backpropagation algorithm for computing gradients of the loss function with respect to weights and biases
- Gradient descent optimization to iteratively adjust network parameters
- Binary cross-entropy loss function for classification tasks
- Various activation functions (ReLU, Sigmoid, Softmax) implemented with their respective derivatives

The network processes input features through multiple layers of neurons, where each neuron computes:

z = Σ(w_i * x_i) + b
         
a = activation(z)

During training, the backpropagation algorithm uses the chain rule to compute partial derivatives and update weights:

w = w - learning_rate * ∂L/∂w
         
b = b - learning_rate * ∂L/∂b

This hands-on approach provides deep insights into neural network operations and the mathematics behind deep learning.
""")

st.image("images/med.png")

st.write("""
## Dataset Overview
This application uses breast cancer diagnostic data from the Wisconsin Breast Cancer Database (WBCD). Each record represents characteristics of cell nuclei obtained from a digitized image of a fine needle aspirate (FNA) of a breast mass.

The dataset includes measurements for malignant and benign breast cancer cases. For computational purposes and binary classification requirements, we have encoded the diagnosis as follows:
- Malignant (M) → 1
- Benign (B) → 0

Each sample contains 30 features computed from the digitized image, measuring various characteristics of the cell nuclei present in the image, such as:
- Radius (mean distance from center to perimeter points)
- Texture (standard deviation of gray-scale values)
- Perimeter
- Area
- Smoothness (local variation in radius lengths)
- Compactness
- Concavity
- Symmetry
- Fractal dimension

For each feature, three different values are calculated:
- Mean
- Standard error
- "Worst" or largest (mean of the three largest values)

This encoding and standardization of the data is crucial for the proper functioning of our neural network, as it allows for binary classification and ensures all features are in appropriate numerical ranges.
""")

data_path = 'data/data.csv'
df = get_df(data_path)
st.dataframe(df)

st.image("images/cell.png")

# Data Analysis Section
st.write("""
## Data Analysis
### Correlation Analysis
[intro text about correlation heatmaps and what insights we can gain from them]
""")

fig, ax = plt.subplots(figsize=(25, 20))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Heatmap')
st.pyplot(fig)
plt.close()

st.write("""
### Feature Visualization
[intro text about scatter plots, how to use this section, and what patterns to look for]
""")

feature_x = st.selectbox("Select X-axis feature:", options=df.columns, index=2)
feature_y = st.selectbox("Select Y-axis feature:", options=df.columns, index=3)

fig_scatter, ax_scatter = plt.subplots()
sns.scatterplot(data=df, x=feature_x, y=feature_y, hue='diagnosis', palette='viridis', ax=ax_scatter)
ax_scatter.set_title(f'Scatter Plot: {feature_x} vs {feature_y}')
st.pyplot(fig_scatter)
plt.close()

# Model Configuration Section
st.write("""
## Evaluation Set Configuration
[intro text about the importance of having a separate evaluation set and how to choose its size]
""")

eval_size = st.slider(
    "How many subjects do you want to keep for final evaluation?", 
    min_value=1, max_value=50, value=10
)

(X_train, X_valid, X_eval, y_train, y_valid, y_eval) = set_data_for_model_with_eval(
    df, random_state=42, eval_size=eval_size
)

st.write("""
## Neural Network Architecture
[intro text about how to configure the neural network, what each parameter means, and best practices]
""")

input_size = X_train.shape[1]
model = MLPSequential(input_size)

# Interactive network configuration
num_layers = st.number_input("Number of hidden layers:", min_value=1, max_value=5, value=2)

for i in range(num_layers):
    col1, col2 = st.columns(2)
    with col1:
        neurons = st.number_input(f"Neurons in layer {i+1}:", min_value=1, max_value=100, value=24)
    with col2:
        activation = st.selectbox(
            f"Activation function for layer {i+1}:",
            options=['relu', 'sigmoid'],
            index=0
        )
    model.Dense(neurons, activation)

# Fixed output layer for binary classification
model.Dense(2, 'softmax')
model.compile()

# Training Configuration
st.write("""
## Model Training and Evaluation
[intro text about training parameters, what they mean, and how they affect the model]
""")

epochs = st.number_input("Enter number of epochs:", min_value=10, max_value=100000, value=1000, step=10)
learning_rate = st.number_input("Enter learning rate:", min_value=0.0001, max_value=0.1, value=0.0349, step=0.0001)
early_stopping_patience = st.slider("Early Stopping Patience:", min_value=1, max_value=50, value=5)

# Initialize session state
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
    st.session_state.model = None
    st.session_state.loss_fig = None
    st.session_state.acc_fig = None

# Training button
training_section = st.container()
with training_section:
    if st.button("Train Model"):
        progress_placeholder = st.empty()
        
        class CustomStdout:
            def __init__(self, placeholder):
                self.placeholder = placeholder
                
            def write(self, text):
                if 'Epoch' in text:
                    self.placeholder.text(text.strip())
                    
            def flush(self):
                pass

        old_stdout = sys.stdout
        sys.stdout = CustomStdout(progress_placeholder)
        
        try:
            with st.spinner('Training in progress...'):
                model.fit(X_train, y_train, epochs=epochs, learning_rate=learning_rate,
                         validation_data=(X_valid, y_valid), early_stopping_patience=early_stopping_patience)
                
                sys.stdout = old_stdout
                
                val_loss, val_accuracy = model.evaluate(X_valid, y_valid)
                
                # Save results and model to session state
                st.session_state.model = model
                st.session_state.training_results = {
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }
                
                # Remove these lines that show immediate results
                # st.success('Training completed!')
                # st.write(f"Final Validation Loss: {val_loss:.4f}")
                # st.write(f"Final Validation Accuracy: {val_accuracy:.4f}")

                # Create and save plots (don't display them here)
                model.plot_loss()
                st.session_state.loss_fig = plt.gcf()
                plt.close()
                
                model.plot_accuracy()
                st.session_state.acc_fig = plt.gcf()
                plt.close()

                model.save_and_plot_history()
                
        except Exception as e:
            sys.stdout = old_stdout
            st.error(f"An error occurred during training: {str(e)}")

# Always display training results if they exist
if st.session_state.training_results:
    st.success('Training completed!')  # Moved here
    st.write("### Training Results:")
    st.write(f"Validation Loss: {st.session_state.training_results['val_loss']:.4f}")
    st.write(f"Validation Accuracy: {st.session_state.training_results['val_accuracy']:.4f}")
    
    if st.session_state.loss_fig and st.session_state.acc_fig:
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(st.session_state.loss_fig)
        with col2:
            st.pyplot(st.session_state.acc_fig)


# Evaluation button
if st.button("Evaluate Model"):
    if st.session_state.model is None:
        st.error("Please train the model first!")
    else:
        y_pred = st.session_state.model.predict(X_eval)
        
        results = []
        for i, (real, pred) in enumerate(zip(y_eval, y_pred)):
            results.append({
                'Subject': i+1,
                'Real Diagnosis': 'Benign' if real == 0 else 'Malignant',
                'Predicted Diagnosis': 'Benign' if pred == 0 else 'Malignant',
                'Result': 'Success' if real == pred else 'Fail'
            })
        
        results_df = pd.DataFrame(results)
        st.write("### Evaluation Results:")
        st.dataframe(results_df)

        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_eval, y_pred)
        
        # Visual confusion matrix
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'],
                    yticklabels=['Benign', 'Malignant'], ax=ax_cm)
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        ax_cm.set_title('Confusion Matrix')
        st.pyplot(fig_cm)
        plt.close()

        # Detailed confusion matrix breakdown
        st.write("### Confusion Matrix Breakdown:")
        tn, fp, fn, tp = cm.ravel()
        st.write(f"""
        - True Negatives (Correctly identified Benign): {tn}
        - False Positives (Incorrectly identified as Malignant): {fp}
        - False Negatives (Incorrectly identified as Benign): {fn}
        - True Positives (Correctly identified Malignant): {tp}
        """)