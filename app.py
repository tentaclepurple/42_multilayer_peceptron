import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocess import get_df

# Cargar los datos
data_path = 'data/data.csv'
df = get_df(data_path)

# Título de la aplicación
st.title("Aplicación con Mapa de Calor")

# Mostrar los datos en la app antes de cualquier filtrado
st.write("Datos utilizados para el mapa de calor:")
st.dataframe(df)

# Crear un formulario para que el usuario seleccione columnas
with st.form(key='formulario_mapa_calor'):
    # Selector múltiple de columnas para elegir qué columnas incluir en el mapa de calor
    columnas_seleccionadas = st.multiselect(
        "Selecciona las columnas para el mapa de calor:",
        options=df.columns,
        default=df.columns  # Por defecto, todas las columnas están seleccionadas
    )
    
    # Botón para aplicar los filtros y generar el gráfico
    submit_button = st.form_submit_button(label='Generar Mapa de Calor')

# Verificar si el formulario fue enviado
if submit_button:
    # Filtrar el DataFrame para incluir solo las columnas seleccionadas
    df_filtrado = df[columnas_seleccionadas]

    # Crear la figura del mapa de calor
    fig, ax = plt.subplots(figsize=(20, 15))
    sns.heatmap(df_filtrado.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Heatmap')

    # Mostrar el mapa de calor en Streamlit
    st.write("Mapa de Calor:")
    st.pyplot(fig)
