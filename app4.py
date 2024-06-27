# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:21:26 2024

@author: jperezr
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Definir x e y globalmente
x = np.linspace(0, length, time_steps)
y = np.linspace(0, width, time_steps)

# Función para simular la propagación de fracturas
def simulate_fracture_propagation(length, width, propagation_rate, time_steps, fracture_pressure, rock_strength):
    propagation = np.zeros((time_steps, time_steps))

    for i in range(time_steps):
        for j in range(time_steps):
            propagation[i, j] = (propagation_rate * (i + j) * fracture_pressure / rock_strength) % width

    return propagation

# Función para crear la visualización y el DataFrame
def create_visualization(length, width, propagation_rate, time_steps, fracture_pressure, rock_strength):
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

# Crear pestañas para diferentes escenarios
with st.sidebar.expander("Configuración Avanzada", expanded=True):
    st.write("Aquí puedes configurar diferentes escenarios para comparar.")

    # Definir diferentes configuraciones de parámetros
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

    scenario_names = list(scenarios.keys())
    selected_scenario = st.selectbox('Selecciona un Escenario', scenario_names)

    # Obtener parámetros del escenario seleccionado
    selected_params = scenarios[selected_scenario]
    length_scenario = selected_params['length']
    width_scenario = selected_params['width']
    propagation_rate_scenario = selected_params['propagation_rate']
    time_steps_scenario = selected_params['time_steps']
    fracture_pressure_scenario = selected_params['fracture_pressure']
    rock_strength_scenario = selected_params['rock_strength']

# Ejecutar la simulación y crear visualizaciones para el escenario seleccionado
df, fig, fig_3d = create_visualization(length_scenario, width_scenario, propagation_rate_scenario,
                                      time_steps_scenario, fracture_pressure_scenario, rock_strength_scenario)

# Mostrar los resultados en pestañas
st.subheader('Resultados del Escenario Seleccionado')

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

# Animación de la propagación de fracturas paso a paso
st.subheader("Animación de la Propagación de Fracturas")

# Crear los frames para la animación
frames = []
for i in range(time_steps):
    propagation_step = simulate_fracture_propagation(length, width, propagation_rate, i+1, fracture_pressure, rock_strength)
    frame = go.Frame(data=go.Heatmap(z=propagation_step, x=x, y=y, colorscale='Viridis'))
    frames.append(frame)

# Crear la figura animada
fig_animation = go.Figure(
    data=go.Heatmap(z=np.zeros((time_steps, time_steps)), x=x, y=y, colorscale='Viridis'),
    frames=frames,
    layout=go.Layout(
        title='Animación de la Propagación de Fracturas',
        xaxis_nticks=36,
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "Pausa",
                    "method": "animate",
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )
)

# Mostrar la animación
st.plotly_chart(fig_animation, use_container_width=True)



st.subheader("")
st.subheader("")

# Sección de ayuda detallada
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

### Análisis Estadístico
- **Media de Propagación**: Promedio de la propagación de fracturas.
- **Mediana de Propagación**: Valor central de la propagación de fracturas.
- **Desviación Estándar**: Variabilidad de la propagación de fracturas.

### Descargar Resultados
Puedes descargar los resultados de la simulación en un archivo CSV para análisis adicional.
""")

# Aviso de derechos de autor
st.sidebar.markdown("""
    ---
    © 2024. Todos los derechos reservados.
    Creado por jahoperi.
""")