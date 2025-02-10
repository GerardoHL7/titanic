import streamlit as st
import joblib
import pandas as pd

# Cargamos nuestro modelo entrenado
model = joblib.load('./titanic_modelo.pkl/')

# Título de la aplicación
st.title('Predicción de Supervivencia en el Titanic')

# Creamos un formulario de entrada
st.sidebar.header('Introduzca los datos del pasajero')

pclass = st.sidebar.selectbox('Clase del pasajero (1 = Primera, 2 = Segunda, 3 = Tercera)', [1, 2, 3])
sex = st.sidebar.radio('Sexo', ['Hombre', 'Mujer'])
age = st.sidebar.number_input('Edad', min_value=0, max_value=100, value=30)
sibsp = st.sidebar.number_input('Número de hermanos/esposos abordo', min_value=0, max_value=10, value=0)
parch = st.sidebar.number_input('Número de padres/hijos abordo', min_value=0, max_value=10, value=0)
fare = st.sidebar.number_input('Tarifa pagada', min_value=0.0, max_value=600.0, value=32.0)
embarked = st.sidebar.selectbox('Puerto de embarque', ['C', 'Q', 'S'])

# Convertimos los valores de texto a valores numéricos para el modelo
sex = 0 if sex == 'female' else 1
# Eliminamos espacios y aseguramos que la primera letra sea mayúscula
embarked = embarked.strip().capitalize()  
full_to_abbr = {'Cherburgo': 'C', 'Queenstown': 'Q', 'Southampton': 'S'}
embarked = full_to_abbr.get(embarked, 'S')  # Asigna 'S' por defecto si no está en el diccionario
embarked_map = {'C': 0, 'Q': 1, 'S': 2}
embarked = embarked_map[embarked]

# Creamos el DataFrame con los datos ingresados
data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]],
                     columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

# Botón de predicción
if st.sidebar.button('Predecir'):
    prediction = model.predict(data)
    resultado = 'sobrevivió' if prediction[0] == 1 else 'no sobrevivió'

    # Utilizamos los metodos (success y error) para que nos muestre los mensajes en color
    if prediction[0]==1:
        st.success(f'La persona {resultado} al hundimiento del RMS Titanic.')
    else:
        st.error(f'La persona {resultado} al hundimiento del RMS Titanic.')

# Mensaje
st.write("\nIntroduzca los datos en el panel lateral y haga clic en 'Predecir' para ver el resultado.")
