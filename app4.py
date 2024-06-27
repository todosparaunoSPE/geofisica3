# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:43:24 2024

@author: jperezr
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Título de la aplicación
st.title("Simulación de Propagación de Fracturas en Yacimientos")

# Parámetros de la simulación
st.sidebar.header("Parámetros de Simulación")

# Definir los valores iniciales de los parámetros
default_length = 500
default_width = 50
default_propagation_rate = 1.0
default_time_steps = 50
default_fracture_pressure = 50
default_rock_strength = 50

# Sliders para los parámetros
length = st.sidebar.slider('Longitud de la Fractura (m)', 100, 1000, default_length)
width = st.sidebar.slider('Ancho de la Fractura (m)', 10, 100, default_width)
propagation_rate = st.sidebar.slider('Tasa de Propagación (m/s)', 0.1, 5.0, default_propagation_rate)
time_steps = st.sidebar.slider('Número de Pasos de Tiempo', 1, 100, default_time_steps)
fracture_pressure = st.sidebar.slider('Presión de Fractura (MPa)', 10, 100, default_fracture_pressure)
rock_strength = st.sidebar.slider('Resistencia de la Roca (MPa)', 10, 100, default_rock_strength)

# Función para simular la propagación de fracturas
def simulate_fracture_propagation(length, width, propagation_rate, time_steps, fracture_pressure, rock_strength):
    propagation = np.zeros((time_steps, time_steps))

    for i in range(time_steps):
        for j in range(time_steps):
            propagation[i, j] = (propagation_rate * (i + j) * fracture_pressure / rock_strength) % width

    return propagation

# Función para crear la visualización y el DataFrame
def create_visualization(length, width, propagation_rate, time_steps, fracture_pressure, rock_strength):
    x = np.linspace(0, length, time_steps)
    y = np.linspace(0, width, time_steps)
    
    propagation = simulate_fracture_propagation(length, width, propagation_rate, time_steps, fracture_pressure, rock_strength)
    df = pd.DataFrame(propagation, index=x, columns=y)
    
    # Crear el heatmap inicial
    fig = go.Figure(data=go.Heatmap(
                       z=propagation,
                       x=x,
                       y=y,
                       colorscale='Viridis'))

    fig.update_layout(
        title='Simulación de Propagación de Fracturas',
        xaxis_nticks=36)

    # Crear la visualización 3D
    fig_3d = go.Figure(data=[go.Surface(z=propagation, x=x, y=y, colorscale='Viridis')])

    fig_3d.update_layout(
        title='Visualización 3D de la Propagación de Fracturas',
        scene=dict(
            xaxis_title='Longitud (m)',
            yaxis_title='Ancho (m)',
            zaxis_title='Propagación (m)'
        ),
        width=800,
        height=800
    )
    
    return df, fig, fig_3d

# Definir diferentes configuraciones de parámetros para escenarios
scenarios = {
    'Escenario 1': {
        'length': 500,
        'width': 50,
        'propagation_rate': 1.0,
        'time_steps': 50,
        'fracture_pressure': 50,
        'rock_strength': 50
    },
    'Escenario 2': {
        'length': 700,
        'width': 70,
        'propagation_rate': 1.5,
        'time_steps': 70,
        'fracture_pressure': 60,
        'rock_strength': 60
    },
    'Escenario 3': {
        'length': 800,
        'width': 80,
        'propagation_rate': 2.0,
        'time_steps': 80,
        'fracture_pressure': 70,
        'rock_strength': 70
    }
}

# Mostrar selección de escenarios en la barra lateral
selected_scenario = st.sidebar.selectbox('Selecciona un Escenario', list(scenarios.keys()))

# Obtener parámetros del escenario seleccionado
selected_params = scenarios[selected_scenario]

# Ejecutar la simulación y crear visualizaciones para el escenario seleccionado
df, fig, fig_3d = create_visualization(selected_params['length'], selected_params['width'], 
                                       selected_params['propagation_rate'], selected_params['time_steps'], 
                                       selected_params['fracture_pressure'], selected_params['rock_strength'])

# Mostrar los resultados en pestañas
st.subheader(f'Resultados del {selected_scenario}')

# Pestañas para visualización y datos
tabs = st.expander("Ver Resultados", expanded=True)
with tabs:
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig_3d, use_container_width=True)
    st.subheader('DataFrame de Resultados')
    st.dataframe(df)
    
    # Botón para descargar los resultados
    st.download_button(
        label="Descargar Resultados CSV",
        data=df.to_csv().encode('utf-8'),
        file_name='resultados_simulacion.csv',
        mime='text/csv',
        help="Haz clic para descargar los resultados de la simulación en formato CSV."
    )

# Sección de ayuda
st.header("Ayuda Detallada")
st.info("""
### Parámetros de Simulación
- **Longitud de la Fractura (m)**: La extensión de la fractura en metros.
- **Ancho de la Fractura (m)**: El ancho de la fractura en metros.
- **Tasa de Propagación (m/s)**: La velocidad a la que la fractura se propaga.
- **Número de Pasos de Tiempo**: La cantidad de pasos de tiempo para la simulación.
- **Presión de Fractura (MPa)**: La presión aplicada para fracturar la roca.
- **Resistencia de la Roca (MPa)**: La resistencia de la roca contra la fractura.

### Visualizaciones
- **Heatmap 2D**: Muestra la propagación de fracturas en un plano bidimensional.
- **Visualización 3D**: Representación tridimensional de la propagación de fracturas.

### Descargar Resultados
Puedes descargar los resultados de la simulación en un archivo CSV para análisis adicional.
""")


# Aviso de derechos de autor
st.sidebar.markdown("""
    ---
    © 2024. Todos los derechos reservados.
    Creado por jahoperi.
""")
