##############################
# Librerías
import streamlit as st
import pandas as pd
##############################
# Funciones
def pred_dureza(vel_d_ll, comp_p_h, fmax_ad, f_compvm, prof_ll, fcomp_srel, alt_alm_compr, alt_alm_precom):
    return 7.580292162090916 + 0.01990029575960765296 * vel_d_ll -0.00478936927023245959 * comp_p_h + 0.22415417148852440077 * fmax_ad  -0.00095709980066172485 * f_compvm + 1.15027404331875926502 * prof_ll -4.81155483460065891421 * fcomp_srel + 0.98232305485875004436 * alt_alm_compr + 3.72663769193418525916 * alt_alm_precom
# Regresa la variable objetivo del modelo de regresión lineal de Dureza

def pred_friabilidad(vel_d_ll, comp_p_h, fmax_ad, f_compvm, prof_ll, fcomp_srel, alt_alm_compr, alt_alm_precom):
    return 0.0035772706730503234 -4.24445670e-06 * vel_d_ll -1.25139943e-06 * comp_p_h -6.75850763e-05 * fmax_ad  + 2.40955915e-05 * f_compvm  - 1.49393146e-03 * prof_ll + 2.47676268e-03 * fcomp_srel -7.72629122e-04 * alt_alm_compr + -3.83294089e-04 * alt_alm_precom
# Regresa la variable objetivo del modelo de regresión lineal de Dureza

##############################
# Main

def main():
    # Título
    st.header('Exploración de Parámetros')

    # Descripción
    st.markdown('''¡Bienvenid@! Ajusta los parámetros del lado izquierdo y mira como cambian los valores de dureza y friabilidad
    en tiempo real. Los valores ```Dureza Max``` y ```Friabilidad Min``` son los máximos y mínimos extraídos de la base de datos
    proporcionada para el reto. Las predicciones están basadas en dos modelos de regresión que se construyeron con base en
    los datos proporcionados.''')

    # Sidebar
    with st.sidebar:

        # Hipervínculo a página de optimización
        st.page_link("pages/page_1.py", label="Optimizar Parámetros", icon="1️⃣")

        # Sliders para cada uno de los valores
        vel_d_ll = st.slider("Velocidad de Llenado", 4.4, 120.0, 93.0655)
        comp_p_h = st.slider("Comprimidos por Hora", 0.0025, 400.0, 238.775956)
        fmax_ad = st.slider("Fuerza máxima admisible de punzón", 0.0, 100.0, 98.284153)
        f_compvm = st.slider("Fuerza de compresión principal: valor medio", 0.0, 51.0, 22.987796)
        prof_ll = st.slider("Profundidad llenado", 3.5, 11.0, 9.266685)
        fcomp_srel = st.slider("Fuerza de compresión principal: s-rel", 3.5, 10.0, 9.905282)
        alt_alm_compr = st.slider("Altura de alma compresión principal", 1.8, 4.0, 2.305510)
        alt_alm_precom = st.slider("Altura de alma precompresión", 2.8, 4.0, 3.379827)

    # Actualiza el valor en tiempo real según el objeto seleccionado

    # Sección de resultados
    st.header('Resultados', divider='gray')

    # Cálculo en tiempo real de dureza y friabilidad usando funciones
    dureza = pred_dureza(vel_d_ll, comp_p_h, fmax_ad, f_compvm, prof_ll, fcomp_srel, alt_alm_compr, alt_alm_precom)
    friabilidad = pred_friabilidad(vel_d_ll, comp_p_h, fmax_ad, f_compvm, prof_ll, fcomp_srel, alt_alm_compr, alt_alm_precom)

    # Regresar 0 si friabilidad es un valor negativo
    friabilidad = friabilidad if friabilidad >= 0 else 0


    # Crear un DataFrame con los valores de dureza y friabilidad
    dfd = pd.DataFrame({
        'Label': ["Dureza", "Dureza Max"],
        'Valor': [dureza, 14.200000]
    })

    dff = pd.DataFrame({
        'Label': ["Friabilidad", "Friabilidad Min"],
        'Valor': [friabilidad, 0.002500]
    })

    # Crear dos columnas
    col1, col2 = st.columns(2)

    # Mostrar las gráficas en las columnas
    with col1:
        st.write(f'Dureza: {dureza:.6f}')
        st.bar_chart(dfd, x='Label', y='Valor', x_label = "")
        
    with col2:
        st.write(f'Friabilidad: {friabilidad:.6f}')
        st.bar_chart(dff, x='Label', y='Valor', x_label = "")
        
#########################################
if __name__ == "__main__":
    main()