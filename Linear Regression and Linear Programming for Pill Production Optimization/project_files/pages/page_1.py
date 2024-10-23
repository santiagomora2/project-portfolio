########################################
# Librerías
import streamlit as st
from scipy.optimize import linprog
import numpy as np
import pandas as pd
#########################################
# Funciones

# Función optimizadora
def optimusprime(t1 = (0.0, 120.0), t2 = (3.5, 350), t3 = (3.5, 100.0), t4 = (0.0, 51.0), t5 = (3.5, 10.7), t6 = (3.5, 10.0),t7 =  (1.8, 3.5), t8 =  (2.8, 4.0)):
    #Los valores defaults de cada tupla/rango corresponden a los máximos encontrados en la base de datos

    #variable: 'd' para max dureza,'f' para min friabilidad 
    #t1 'Velocidad dispositivo de llenado'
    #t2 'Comprimidos por hora'
    #t3 'Fuerza máxima admisible de punzón'
    #t4 'Fuerza de compresión principal: valor medio'
    #t5  'Profundidad llenado'
    #t6  'Fuerza de compresión principal: s-rel',
    #t7  'Altura de alma compresión principal'
    #t8 'Altura de alma precompresión'
    bounds = [t1,t2,t3,t4,t5,t6,t7,t8]
    #valores del modelo para dureza
    beta_d = np.array([0.019900295759609027,
                -0.004789369270233657,
                    0.2241541714885204,
                -0.0009570998006501072,
                    1.1502740433187562,
                -4.811554834600622,
                    0.9823230548587643,
                    3.7266376919341857]) 
    beta_0_d = 7.580292162090917
    
    #valores del modelo para friabilidad
    beta_f = np.array([-4.244456702565667e-06,
                -1.2513994250759695e-06,
                -6.75850763313965e-05,
                2.409559148551634e-05,
                -0.0014939314585098009,
                    0.0024767626760789575,
                    -0.0007726291224251929,
                -0.00038329408932766533]) 
    beta_0_f = 0.0035772706730504717

    #valores óptimos de dureza y resultantes de friabilidad
    opt_d_result = linprog(-beta_d, bounds=bounds, method='highs')
    opt_d_values = [float(value) for value in opt_d_result.x]
    opt_d = float(beta_0_d + sum(b * x for b, x in zip(beta_d, opt_d_values)))
    res_f = float(beta_0_f + sum(b * x for b, x in zip(beta_f, opt_d_values)))
    dureza = [opt_d,opt_d_values,res_f]

    #valores óptimos de friabilidad y resultantes de dureza
    opt_f_result = linprog(beta_f, bounds=bounds, method='highs')
    opt_f_values = [float(value) for value in opt_f_result.x]
    opt_f =float( beta_0_f + sum(b * x for b, x in zip(beta_f, opt_f_values)))
    res_d = float(beta_0_d + sum(b * x for b, x in zip(beta_d, opt_f_values)))
    friabilidad = [opt_f, opt_f_values,res_d]

    return {'d': dureza, 'f': friabilidad}
    #La función regresa un diccionario con la lista de valores para dureza o friabilidad según la key
    #El orden de los valores que hay en cada lista:
    #   0: valor óptimo de la varaible de interés seleccionada
    #   1: valores de las variables regresoras encontradas
    #   2: valor resultante de la otra variable no escogida
    #      Esto se hace introduciendo los valores encontrados para la varaiable seleccionada
    #      en la linea de regresión de la variable no escogida

###################################
# Main
def main():
    # Título
    st.header('Optimización de Parámetros')

    # Descripción
    st.markdown('''¡Bienvenid@! Ajusta los rangos del espacio de búsqueda de cada parámetro del lado izquierdo y obtén los valores óptimos de dureza y friabilidad
    dentro de ese rango en tiempo real, así como los parámetros necesarios para llegar a ese óptimo. Los valores ```Dureza Max``` y ```Friabilidad Min``` son los máximos y mínimos extraídos de la base de datos
    proporcionada para el reto. Las predicciones están basadas en dos modelos de regresión que se construyeron con base en
    los datos proporcionados.''')

    # Sidebar
    with st.sidebar:

        # Hipervínculo a página de exploración de parámetros
        st.page_link("app.py", label="Explorar Parámetros", icon="🏠")

        # Sliders para los datos, regresan una tupla de datos (mínimo, máximo) de un rango
        vel_d_ll = st.slider("Velocidad de Llenado", 4.4, 120.0, (4.4, 120.0))
        comp_p_h = st.slider("Comprimidos por Hora", 0.0025, 400.0, (0.0025, 400.0))
        fmax_ad = st.slider("Fuerza máxima admisible de punzón", 0.0, 100.0, (0.0, 100.0))
        f_compvm = st.slider("Fuerza de compresión principal: valor medio", 0.0, 51.0, (0.0, 51.0))
        prof_ll = st.slider("Profundidad llenado", 3.5, 11.0, (3.5, 11.0))
        fcomp_srel = st.slider("Fuerza de compresión principal: s-rel", 3.5, 10.0, (3.5, 10.0))
        alt_alm_compr = st.slider("Altura de alma compresión principal", 1.8, 4.0, (1.8, 4.0))
        alt_alm_precom = st.slider("Altura de alma precompresión", 2.8, 4.0, (2.8, 4.0))

    # Actualiza el valor en tiempo real según el objeto seleccionado

    # Resultados
    st.header('Resultados', divider='gray')

    # Obtener el diccionario en tiempo real usando los valores de los sliders
    md = optimusprime(vel_d_ll, comp_p_h, fmax_ad, f_compvm, prof_ll, fcomp_srel, alt_alm_compr, alt_alm_precom)

    # Lista de los nombres
    names = ['Velocidad dispositivo de llenado', 'Comprimidos por hora', 'Fuerza máxima admisible de punzón', 'Fuerza de compresión principal: valor medio', 'Profundidad llenado', 'Fuerza de compresión principal: s-rel', 'Altura de alma compresión principal', 'Altura de alma precompresión']

    # Del análisis de maximizar dureza:
        # Extraer valor máximo de dureza
    maxdureza = md['d'][0]
        # Extraer la lista de valores de cada prámetro para ese máximo de dureza
    maxdureza_valores = md['d'][1]
        # Extraer la friabilidad asociada a ese valor de dureza
    maxdureza_fri = md['d'][2] if md['d'][2]>=0 else 0

    # Del análisis de maximizar dureza:
        # Extraer valor mínimo de friabilidad
    maxfri = md['f'][0] if md['f'][0] >=0 else 0
        # Extraer la lista de valores de cada prámetro para ese mínimo de friabilidad
    maxfri_valores = md['f'][1]
        # Extraer la dureza asociada a ese valor de friabilidad
    maxfri_dureza = md['f'][2]

#######################################################################################
# Mostrar valores de dureza

    st.subheader('Valores al maximizar DUREZA')


    # Crear un DataFrame con los valores de dureza y friabilidad para el máximo de dureza
    mdd = pd.DataFrame({
        'Label': ["Dureza", "Dureza Max"],
        'Valor': [maxdureza, 14.200000]
    })

    mdf = pd.DataFrame({
        'Label': ["Friabilidad", "Friabilidad Min"],
        'Valor': [maxdureza_fri, 0.002500]
    })



    # Mostrar valores óptimos

    st.write('Valores óptimos: ')
    for i in range(len(names)):
        st.markdown(f'* {names[i]} -- {maxdureza_valores[i]:.4f}')

    # Crear dos columnas
    col1, col2 = st.columns(2)

    # Mostrar las gráficas en las columnas
    with col1:
        st.write(f'Dureza: {maxdureza:.6f}')
        st.bar_chart(mdd, x='Label', y='Valor', x_label = "")
        
    with col2:
        st.write(f'Friabilidad (Asociada al máximo de dureza): {maxdureza_fri:.6f}')
        st.bar_chart(mdf, x='Label', y='Valor', x_label = "")

#######################################################################################
# Mostrar valores de friabilidad

    st.subheader('Valores al minimizar FRIABILIDAD')


    # Crear un DataFrame con los valores de dureza y friabilidad para el máximo de friabilidad
    mfd = pd.DataFrame({
        'Label': ["Dureza", "Dureza Max"],
        'Valor': [maxfri_dureza, 14.200000]
    })

    mff = pd.DataFrame({
        'Label': ["Friabilidad", "Friabilidad Min"],
        'Valor': [maxfri, 0.002500]
    })

    # Mostrar valores óptimos
    st.write('Valores óptimos: ')
    for i in range(len(names)):
        st.markdown(f'* {names[i]} -- {maxfri_valores[i]:.4f}')

    # Crear dos columnas
    col1, col2 = st.columns(2)

    # Mostrar las gráficas en las columnas
    with col1:
        st.write(f'Dureza (Asociado al mínimo de friabilidad): {maxfri_dureza:.6f}')
        st.bar_chart(mfd, x='Label', y='Valor', x_label = "")
        
    with col2:
        st.write(f'Friabilidad: {maxfri:.6f}')
        st.bar_chart(mff, x='Label', y='Valor', x_label = "")

####################################
if __name__ == "__main__":
    main()