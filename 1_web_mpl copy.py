import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.MLP import MLPSequential
from utils.preprocess import get_df, set_data_for_model_with_eval

data_path = 'data/data.csv'
df = get_df(data_path)

st.title("MLP Classifier Interactive Application")

st.write("## Dataset Preview")
st.dataframe(df)

st.write("## Heatmap of Feature Correlation")
fig, ax = plt.subplots(figsize=(25, 20))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Heatmap')
st.pyplot(fig)

st.write("## Select Features for Scatter Plot")
feature_x = st.selectbox("Select X-axis feature:", options=df.columns, index=2)
feature_y = st.selectbox("Select Y-axis feature:", options=df.columns, index=3)

st.write(f"### Scatter Plot: {feature_x} vs {feature_y}")
fig_scatter, ax_scatter = plt.subplots()
sns.scatterplot(data=df, x=feature_x, y=feature_y, hue='diagnosis', palette='viridis', ax=ax_scatter)
ax_scatter.set_title(f'Scatter Plot: {feature_x} vs {feature_y}')
st.pyplot(fig_scatter)

st.write("## Evaluation Set Size")
eval_size = st.slider(
    "How many subjects do you want to keep for final evaluation?", 
    min_value=1, max_value=50, value=10
)

st.write("## Data Processing for MLP Model")
(X_train, X_valid, X_eval, y_train, y_valid, y_eval) = set_data_for_model_with_eval(
    df, random_state=42, eval_size=eval_size
)

input_size = X_train.shape[1]

st.write("## Configuring the MLP Model")
model = MLPSequential(input_size)

model.Dense(24, 'relu')
model.Dense(48, 'relu')
model.Dense(2, 'softmax')

model.compile()
st.write("Model compiled successfully with layers:")
st.write("Layer 1: 24 neurons, activation='relu'")
st.write("Layer 2: 48 neurons, activation='relu'")
st.write("Output Layer: 2 neurons, activation='softmax'")

st.write("## Model Training")
epochs = st.number_input("Enter number of epochs:", min_value=10, max_value=100000, value=1000, step=10)
learning_rate = st.number_input("Enter learning rate:", min_value=0.0001, max_value=0.1, value=0.0349, step=0.0001)
early_stopping_patience = st.slider("Early Stopping Patience:", min_value=1, max_value=50, value=5)

if st.button("Train Model"):
    model.fit(X_train, y_train, epochs=epochs, learning_rate=learning_rate,
              validation_data=(X_valid, y_valid), early_stopping_patience=early_stopping_patience)

    val_loss, val_accuracy = model.evaluate(X_valid, y_valid)
    st.write(f"Validation Loss: {val_loss:.4f}")
    st.write(f"Validation Accuracy: {val_accuracy:.4f}")

    model.plot_loss()
    model.plot_accuracy()
    st.pyplot(plt.gcf())

    model.save_and_plot_history()
    st.write("Training history saved and plotted.")

st.write("## Model Evaluation on the Final Test Set")
if st.button("Evaluate Model"):
    y_pred = model.predict(X_eval)
    
    data = {
        'Subject': list(range(1, len(y_eval) + 1)),
        'Real Diagnosis': ['Benign' if real == 0 else 'Malignant' for real in y_eval],
        'Predicted Diagnosis': ['Benign' if pred == 0 else 'Malignant' for pred in y_pred],
        'Result': ['Success' if pred == real else 'Fail' for pred, real in zip(y_pred, y_eval)]
    }
    
    results_df = pd.DataFrame(data)
    
    st.write("### Evaluation Results:")
    st.table(results_df)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_eval, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'], ax=ax_cm)
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    ax_cm.set_title('Confusion Matrix')
    st.pyplot(fig_cm)
