import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random
import os
import ast

from sklearn.metrics import confusion_matrix


class MLPSequential:

    def __init__(self, input_size, output_size=2):
        """
        MPLSequential class constructor.

        Attributes:
        - input_size (int): Number of features in the input data.
        - output_size (int): Number of classes in the output data.
        - seed (int, optional): Seed for reproducibility

        Initializes the model with the provided input and output sizes.

        The following attributes are also initialized:
        - layers (list): List to store the hidden layers and the output layer.
        - weights (list): List to store the weights of each layer.
        - biases (list): List to store the biases of each layer.
        - is_compiled (bool): Indicates if the model has been compiled.
        - activation_functions (dict): Dictionary with the activation functions.
        - loss_function (function): Loss function for the model.
        - train_loss (list): Training loss history.
        - val_loss (list): Validation loss history.
        - train_accuracy (list): Training accuracy history.
        - val_accuracy (list): Validation accuracy history.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.seed = None
        self.layers = []
        self.weights = []
        self.biases = []
        self.is_compiled = False 
        self.activation_functions = {
            'relu': self.relu,
            'sigmoid': self.sigmoid,
            'softmax': self.softmax
        }
        self.loss_function = self.binary_cross_entropy
        
        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_accuracy = []

        self.history = {}
        self.epochs = 0
        self.timestamp = None
        self.learning_rate = None

    def Dense(self, num_neurons, activation, seed=None):
        """
        Method to add a new layer to the model.

        Parameters:
        - num_neurons (int): Number of neurons in the layer.
        - activation (str): Activation function for the layer ('relu', 'sigmoid', 'softmax')

        Raises:
        - ValueError: If the model has already been compiled.
        - ValueError: If the activation function is not supported.
        """
        if self.is_compiled:
            raise ValueError("Layers cannot be added after compiling the model.")

        if activation not in self.activation_functions:
            raise ValueError(f"Activation function '{activation}' is not supported.")

        if seed is not None:
            self.seed = random.randint(0, 1000)

        self.layers.append({
            'num_neurons': num_neurons,
            'activation': activation
        })

        print(f"Added layer with {num_neurons} neurons and {activation} activation.")

    def compile(self):
        """
        Método para compilar el modelo, inicializar pesos y sesgos.
        Verifica la consistencia de la capa de salida.
        - Si la última capa es 'sigmoid', comprueba que solo tenga una neurona.
        - Si la última capa es 'softmax', comprueba que el número de neuronas sea igual al número de etiquetas.
        - Si no hay una capa de salida, añade una capa con 'softmax' y `output_size` neuronas.
        """
        if len(self.layers) < 2:
            raise ValueError("Neural network must have at least 2 hidden layer and 1 output layer.")

        # Check output layer
        last_layer = self.layers[-1]
        if last_layer['activation'] == 'sigmoid' and last_layer['num_neurons'] != 1:
            raise ValueError("Output layer with 'sigmoid' activation must have 1 "
                            "neuron for binary classification.")
        elif last_layer['activation'] == 'softmax' and last_layer['num_neurons'] != self.output_size:
            raise ValueError(f"Output layer with 'softmax' activation must have {self.output_size} "
                            "neurons for multiclass classification.")
        elif last_layer['activation'] not in ['sigmoid', 'softmax']:
            self.layers.append({
                'num_neurons': self.output_size,
                'activation': 'softmax'
            })
            print(f"Automatically added output layer with {self.output_size} neurons "
                "and 'softmax' activation.")

        # Init weights and biases
        np.random.seed(self.seed)
        layer_input_size = self.input_size

        for layer in self.layers:
            num_neurons = layer['num_neurons']
            # Initialize weights with He initialization
            weights = np.random.randn(layer_input_size, num_neurons) * np.sqrt(2. / (layer_input_size + num_neurons))
            biases = np.zeros((1, num_neurons))
            self.weights.append(weights)
            self.biases.append(biases)
            layer_input_size = num_neurons 

        self.is_compiled = True
        print("Model compiled successfully.")

    def fit(self, X_train, y_train, epochs=100, learning_rate=0.01, validation_data=None, early_stopping_patience=10):
        """
        Method to train the model using the provided data.

        Parameters:
        - X_train (np.array): Training input data.
        - y_train (np.array): Training output labels.
        - epochs (int): Number of training epochs.
        - learning_rate (float): Learning rate for the gradient descent.
        - validation_data (tuple, optional): Tuple (X_valid, y_valid) to monitor performance.
        - early_stopping_patience (int): Number of epochs without improvement for early stopping.
        """        

        # Store history of loss and accuracy
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        best_val_loss = float('inf')
        self.epochs = epochs
        epochs_no_improve = 0
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.learning_rate = learning_rate

        # Unpack validation data if provided
        X_valid, y_valid = validation_data if validation_data is not None else (None, None)

        y_train = np.eye(2)[y_train]
        y_valid = np.eye(2)[y_valid]

        for epoch in range(epochs):
            # Feedforward to get predictions
            y_pred = self.feedforward(X_train)

            # Calculate loss. Binary Cross-Entropy
            loss = self.binary_cross_entropy(y_train, y_pred)
            self.history['loss'].append(loss)

            # Calculate accuracy 
            accuracy = self.compute_accuracy(y_train, y_pred)
            self.history['accuracy'].append(accuracy)

            # Retropropagation to update weights and biases
            self.backpropagation(X_train, y_train, learning_rate=learning_rate)

            # Calculate validation loss and accuracy if validation data is provided
            if X_valid is not None and y_valid is not None:
                y_valid_pred = self.feedforward(X_valid)
                val_loss = self.binary_cross_entropy(y_valid, y_valid_pred)
                val_accuracy = self.compute_accuracy(y_valid, y_valid_pred)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)

                print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f}"
                        f" - accuracy: {accuracy:.4f} - val_accuracy: {val_accuracy:.4f}")

                # Early Stopping: verify if validation loss improves
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f"-> No improvement in validation loss for {epochs_no_improve} epoch(s).")

                # If no improvement for `patience` epochs, stop training
                if epochs_no_improve >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break
            else:
                print(f"Epoch {epoch+1}/{epochs} - loss: {loss:.4f} - accuracy: {accuracy:.4f}")

        print("Training finished.")
        
        if X_valid is not None and y_valid is not None:
            y_valid_pred = self.feedforward(X_valid)
            y_valid_labels = np.argmax(y_valid, axis=1)
            y_pred_labels = np.argmax(y_valid_pred, axis=1)
            cm = confusion_matrix(y_valid_labels, y_pred_labels)
            print("Confusion Matrix:\n", cm)
            TN, FP, FN, TP = cm.ravel()
            print(f"True Positives: {TP}\nTrue Negatives: {TN}\nFalse Positives: {FP}\nFalse Negatives: {FN}")
        

        
    def feedforward(self, X):
        """
        Makes the feedforward through the network to get the predictions.
        Parameters:
        - X (np.array): Input data.
        Returns:
        - np.array: Predictions.
        """
        # At the beginning, the input is the data
        A = X
        
        # Propagate through the layers
        for idx, layer in enumerate(self.layers):
            W = self.weights[idx]
            b = self.biases[idx]
            activation_func = self.activation_functions[layer['activation']]
            
            Z = np.dot(A, W) + b

            A = activation_func(Z)

        return A

    def backpropagation(self, X, y, learning_rate):
        """
        Makes the backpropagation through the network to update the weights and biases.
        Parameters:
        - X (np.array): Input data.
        - y (np.array): True labels.
        - learning_rate (float): Learning rate for the gradient descent.
        Returns:
        - float: Loss value.
        """
        A = X
        activations = [A] 
        Z_values = []

        # Feedforward while storing activations and Z values
        for idx, layer in enumerate(self.layers):
            W = self.weights[idx]
            b = self.biases[idx]
            activation_func = self.activation_functions[layer['activation']]

            Z = np.dot(A, W) + b
            Z_values.append(Z)
            A = activation_func(Z)
            activations.append(A)

        error = activations[-1] - y  # (n_samples, output_size)

        dWs = []
        dbs = []

        # Retropropagation to update weights and biases
        for idx in reversed(range(len(self.layers))):
            Z = Z_values[idx]
            A_prev = activations[idx]
            W = self.weights[idx]
            activation_func = self.activation_functions[self.layers[idx]['activation']]
            
            # calculate the derivative of the activation function
            # using the derivative of the activation function
            if self.layers[idx]['activation'] == 'relu':
                dZ = error * self.relu_derivative(Z)
            elif self.layers[idx]['activation'] == 'sigmoid':
                dZ = error * self.sigmoid_derivative(Z)
            elif self.layers[idx]['activation'] == 'softmax':
                dZ = error

            # Calculate gradients
            dW = np.dot(A_prev.T, dZ) / X.shape[0]
            db = np.sum(dZ, axis=0, keepdims=True) / X.shape[0]

            # Store gradients
            dWs.insert(0, dW)
            dbs.insert(0, db)

            # Propagate the error to the previous layer
            if idx > 0:
                error = np.dot(dZ, W.T)

        # Update weights and biases. Gradient descent step
        for idx in range(len(self.layers)):
            self.weights[idx] -= learning_rate * dWs[idx]
            self.biases[idx] -= learning_rate * dbs[idx]

    def relu(self, z):
        """
        Rectified Linear Unit (ReLU).
        Parameters:
        - z (np.array): Valores before activation.
        Returns:
        - np.array: Activated values.
        """
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def sigmoid(self, z):
        """
        Sigmoid activation function.
        Parameters:
        - z (np.array): Values before activation.
        Returns:
        - np.array: Activated values.
        """
        return 1 / (1 + np.exp(-z))    

    def sigmoid_derivative(self, z):
        sigmoid = self.sigmoid(z)
        return sigmoid * (1 - sigmoid)

    def softmax(self, z):
        """
        Softmax activation function.
        Parameters:
        - z (np.array): Values before activation.
        Returns:
        - np.array: Activated values.
        """
        exps = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def softmax_derivative(self, z):
        return self.softmax(z)

    def binary_cross_entropy(self, y_true, y_pred):
        """
        Binary Cross-Entropy loss function.
        Parameters:
        - y_true (np.array): True labels.
        - y_pred (np.array): Predicted labels.
        Returns:
        - float: Loss value.
        """
        # Add epsilon to prevent log(0)
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def compute_accuracy(self, y_true, y_pred):
        """
        Compute the accuracy of the model.
        Parameters:
        - y_true (np.array): True labels.
        - y_pred (np.array): Predicted labels.
        Returns:
        - float: Accuracy value.
        """
        predictions = np.argmax(y_pred, axis=1)
        targets = np.argmax(y_true, axis=1)
        return np.mean(predictions == targets)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data.

        Parameters:
        - X_test (np.array): Test input data.
        - y_test (np.array): Test output labels.
        
        Returns:
        - test_loss (float): Loss on the test set.
        - test_accuracy (float): Accuracy on the test set.
        """
        y_test = np.eye(2)[y_test]

        y_pred = self.feedforward(X_test)
        test_loss = self.binary_cross_entropy(y_test, y_pred)
        test_accuracy = self.compute_accuracy(y_test, y_pred)
        print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
        return test_loss, test_accuracy

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        - X (np.array): Input data to predict.
        
        Returns:
        - np.array: Predicted classes.
        """
        y_pred = self.feedforward(X)
        # Convert softmax probabilities to class predictions (0 or 1)
        return np.argmax(y_pred, axis=1)

    def save(self, filepath):
        """
        Save the model to a JSON file.
        Parameters:
        - filepath (str): Path to save
        """
        model_data = {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'seed': self.seed,
            'layers': self.layers,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
        print(f"Model saved: {filepath}.")

    @classmethod
    def load(cls, filepath):
        """
        Load a model from a JSON file.
        Parameters:
        - filepath (str): Path to the JSON file.
        Returns:
        - MLPSequential: Model loaded from the file.
        """
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        model = cls(
            input_size=model_data['input_size'],
            output_size=model_data['output_size'],
            seed=model_data['seed']
        )

        model.layers = model_data['layers']
        model.weights = [np.array(w) for w in model_data['weights']]
        model.biases = [np.array(b) for b in model_data['biases']]
        model.is_compiled = True  # Se asume que un modelo cargado ya estaba compilado

        print(f"Model loaded: {filepath}.")
        return model

    def plot_loss(self):
        """
        Plot the loss history if it is not empty.
        """
        if self.history['loss'] and self.history['val_loss']:
            plt.figure(figsize=(7, 6))
            plt.plot(self.history['loss'], label='Loss')
            plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss vs. Validation Loss')
            plt.legend()
            plt.show()
        else:
            print("History is empty. Please train the model before plotting the loss.")

    def plot_accuracy(self):
        """
        Plot the accuracy history if it is not empty.
        """
        if self.history['accuracy'] and self.history['val_accuracy']:
            plt.figure(figsize=(7, 6))
            plt.plot(self.history['accuracy'], label='Accuracy')
            plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs. Validation Accuracy')
            plt.legend()
            plt.show()
        else:
            print("History is empty. Please train the model before plotting the accuracy.")

    def save_and_plot_history(
                            self, history_file='models/historic.csv',
                            plot_file='models/historic.png'
                            ):
        """
        Save the training history to a CSV file and plot the training loss history.
        Parameters:
        - history_file (str): Path to save the CSV file.
        - plot_file (str): Path to save the plot.

        The CSV file will have the following columns:
        - timestamp: Timestamp of the training process.
        - n_layers: Number of layers in the model.
        - n_epochs: Number of epochs in the training process.
        - learning_rate: Learning rate used in the training process.
        - loss_values: List of loss values during training

        The plot will show the training loss history for all the models in the CSV file.
        """
        loss_values = self.history['loss']
        
        n_layers = len(self.layers)
        n_epochs = self.epochs
        learning_rate = self.learning_rate

        new_entry = {
            'timestamp': self.timestamp,
            'n_layers': n_layers if n_layers is not None else 'N/A',
            'n_epochs': n_epochs if n_epochs != 0 else 'N/A',
            'learning_rate': learning_rate if learning_rate is not None else 'N/A',
            'loss_values': str(loss_values)
        }
        
        if os.path.exists(history_file):
            df = pd.read_csv(history_file)
            df['loss_values'] = df['loss_values'].replace('nan', '0')

        else:
            df = pd.DataFrame(columns=['timestamp', 'n_layers', 'n_epochs', 'learning_rate', 'loss_values'])
        
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        
        df.to_csv(history_file, index=False)
        
        plt.figure(figsize=(12, 6))
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(df)))
        
        for idx, row in df.iterrows():
            loss_values = ast.literal_eval(row['loss_values'])
            epochs = range(1, len(loss_values) + 1)
            
            label = f"{row['timestamp']} (Layers:{row['n_layers']}, Epochs:{row['n_epochs']}, LR:{row['learning_rate']})"
            
            plt.plot(epochs, loss_values, color=colors[idx], label=label)
        
        plt.title('Training Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(plot_file, bbox_inches='tight')
        plt.show()
        plt.close()