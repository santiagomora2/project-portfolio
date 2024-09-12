import streamlit as st

#--------------------------------------------------------------------------------------------------------------
# UTILS
#--------------------------------------------------------------------------------------------------------------

# sobre cliente, gráfica de barras de los materiales que compran

import pandas as pd
import numpy as np
from numpy.linalg import eig, inv
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('db.csv', usecols = ['fecha', 'material', 'id_cliente', 'ventas'])
df['ventas'] = df['ventas'].astype('float')
df['fecha'] = pd.to_datetime(df['fecha'])

similarity_matrix = pd.read_csv('similarity_matrix.csv', index_col=0)

def monthly_trans_mat_mat(material, df = df):

    # Filter dataframe by material
    df_material = df[df['material'] == material]

    # Determine the global date range across all materials
    material_start_date = df_material['fecha'].min().to_period('M')
    global_end_date = df['fecha'].max().to_period('M')

    # Create a full date range for all months from the global start to end date
    all_months = pd.period_range(start=material_start_date, end=global_end_date, freq='M')

    # Group by year and month, and aggregate 'ventas' per month
    
    df_material['year_month'] = df_material['fecha'].dt.to_period('M')
    
    monthly_sales = df_material.groupby('year_month')['ventas'].sum().reset_index()

    # Reindex to include all months in the global date range, filling missing months with 0 ventas
    monthly_sales = monthly_sales.set_index('year_month').reindex(all_months, fill_value=0).reset_index()

    # Initialize activity states: 1 for active, 0 for inactive
    monthly_sales['activity'] = 0

    # Determine activity based on sales
    for i in range(len(monthly_sales)):
        if monthly_sales.loc[i, 'ventas'] > 0:
            monthly_sales.loc[i, 'activity'] = 1
        elif monthly_sales.loc[i, 'ventas'] < 0 and i < len(monthly_sales) - 1:
            # Lookahead: if the next month is positive, mark this month as active
            if monthly_sales.loc[i + 1, 'ventas'] > 0:
                monthly_sales.loc[i, 'activity'] = 1

    # Calculate the transitions
    transitions = monthly_sales['activity'].diff().fillna(0)

    # Initialize the transition matrix
    transition_matrix = np.zeros((2, 2))

    # Count the transitions and fill the transition matrix
    for i in range(1, len(transitions)):
        prev_state = int(monthly_sales['activity'].iloc[i-1])
        current_state = int(monthly_sales['activity'].iloc[i])
        transition_matrix[prev_state, current_state] += 1

    n = transition_matrix[0].sum()
    m = transition_matrix[1].sum()

    transition_matrix[0][0], transition_matrix[0][1] = transition_matrix[0][0]/n, transition_matrix[0][1]/n
    transition_matrix[1][0], transition_matrix[1][1] = transition_matrix[1][0]/m, transition_matrix[1][1]/m

    # Return both the transition matrix and the last active date
    return transition_matrix

def last_date_mat(material, df = df):

    return df[df['material'] == material]['fecha'].max()

def first_date_mat(material, df = df):

    return df[df['material'] == material]['fecha'].min()

def stationary_distr(P):

    p = P[0][1]
    q = P[1][0]

    pi = np.array([q/(p+q), p/(p+q)])

    if P[0][0] == 1 or P[1][1] == 1:
        raise ValueError('Al menos uno de los de la cadena de Markov es absorbente, por lo que la cadena no tiene distribución estacionaria')

    return pi

def t_medio_recurr(pi):

    return 1/pi[1]

def monthly_trans_mat_cli(id_cliente, df = df):

    # Filter dataframe by material
    df_cliente = df[df['id_cliente'] == id_cliente]

    # Determine the global date range across all materials
    client_start_date = df_cliente['fecha'].min().to_period('M')
    global_end_date = df['fecha'].max().to_period('M')

    # Create a full date range for all months from the global start to end date
    all_months = pd.period_range(start=client_start_date, end=global_end_date, freq='M')

    # Group by year and month, and aggregate 'ventas' per month
    
    df_cliente['year_month'] = df_cliente['fecha'].dt.to_period('M')
    
    monthly_sales = df_cliente.groupby('year_month')['ventas'].sum().reset_index()

    # Reindex to include all months in the global date range, filling missing months with 0 ventas
    monthly_sales = monthly_sales.set_index('year_month').reindex(all_months, fill_value=0).reset_index()

    # Initialize activity states: 1 for active, 0 for inactive
    monthly_sales['activity'] = 0

    # Determine activity based on sales
    for i in range(len(monthly_sales)):
        if monthly_sales.loc[i, 'ventas'] > 0:
            monthly_sales.loc[i, 'activity'] = 1
        elif monthly_sales.loc[i, 'ventas'] < 0 and i < len(monthly_sales) - 1:
            # Lookahead: if the next month is positive, mark this month as active
            if monthly_sales.loc[i + 1, 'ventas'] > 0:
                monthly_sales.loc[i, 'activity'] = 1

    # Calculate the transitions
    transitions = monthly_sales['activity'].diff().fillna(0)

    # Initialize the transition matrix
    transition_matrix = np.zeros((2, 2))

    # Count the transitions and fill the transition matrix
    for i in range(1, len(transitions)):
        prev_state = int(monthly_sales['activity'].iloc[i-1])
        current_state = int(monthly_sales['activity'].iloc[i])
        transition_matrix[prev_state, current_state] += 1

    n = transition_matrix[0].sum()
    m = transition_matrix[1].sum()

    transition_matrix[0][0], transition_matrix[0][1] = transition_matrix[0][0]/n, transition_matrix[0][1]/n
    transition_matrix[1][0], transition_matrix[1][1] = transition_matrix[1][0]/m, transition_matrix[1][1]/m

    # Return both the transition matrix and the last active date
    return transition_matrix

def last_date_cli(client_id, df = df):

    return df[df['id_cliente'] == client_id]['fecha'].max()

def first_date_cli(client_id, df = df):

    return df[df['id_cliente'] == client_id]['fecha'].min()

def plot_mat(material, df = df):
    df_material = df[df['material'] == material]
    clientes_mas_compran = df_material.groupby('id_cliente')['ventas'].sum().sort_values(ascending=False).head(10)
    order = clientes_mas_compran.index

    fig, _ = plt.subplots()
    graph  = sns.barplot(x = clientes_mas_compran.index, y = list(clientes_mas_compran.values), order = order, palette = 'plasma')

    plt.xticks(rotation = 90)
    title = 'top 10 clientes que compran el material'
    plt.title(title)
    plt.xlabel('id de cliente')
    plt.ylabel('ventas totales (MXN)')

    return graph

def plot_cli(id_cliente, df = df):
    df_cliente = df[df['id_cliente'] == id_cliente]
    materiales_mas_compran = df_cliente.groupby('material')['ventas'].sum().sort_values(ascending=False).head(10)
    order = materiales_mas_compran.index

    fig, _ = plt.subplots()
    graph  = sns.barplot(x = materiales_mas_compran.index, y = list(materiales_mas_compran.values), order = order, palette = 'plasma')

    plt.xticks(rotation = 90)
    title = 'top 10 materiales que compra el cliente'
    plt.title(title)
    plt.xlabel('material')
    plt.ylabel('ventas totales (MXN)')


    return graph

def proporcion_negativos_cli(id_cliente, db = df):
    db_cliente = db[db['id_cliente'] == id_cliente]
    return np.round(100*db_cliente[db_cliente['ventas'] < 0].shape[0] / db_cliente.shape[0], 2)

def proporcion_negativos_mat(material, db = df):
    db_material = db[db['material'] == material]
    return np.round(100*db_material[db_material['ventas'] < 0].shape[0] / db_material.shape[0], 2)

# --------------  sistema de recomendaciones -------------------------------------

def recomendar_productos(id_cliente, num_recomendaciones=10, df = df):
    df_cliente = df[df['id_cliente'] == id_cliente]

    # Obtener los top 5 materiales más comprados por el cliente
    materiales_mas_compran = df_cliente.groupby('material')['ventas'].sum().sort_values(ascending=False).head(10)
    top5_cliente = materiales_mas_compran.index.tolist()

    recomendaciones = pd.Series(dtype=float)
    for material in top5_cliente:
        recomendaciones = recomendaciones.add(similarity_matrix[material], fill_value=0)
    
    recomendaciones = recomendaciones.groupby(recomendaciones.index).mean().sort_values(ascending=False)
    recomendaciones = recomendaciones[~recomendaciones.index.isin(top5_cliente)]
    return recomendaciones.head(num_recomendaciones)

def recomendar_materiales_similares(material, num_recomendaciones=10):

    # Verificar si el material está en la matriz de similitud
    if material not in similarity_matrix.index:
        raise ValueError(f"El material '{material}' no se encuentra en la matriz de similitud.")
    
    # Obtener las similitudes de ese material con todos los demás
    similitudes = similarity_matrix[material]
    
    # Ordenar los materiales por similitud de forma descendente y excluir el propio material
    recomendaciones = similitudes.sort_values(ascending=False).drop(material)
    
    # Retornar los 10 materiales más similares
    return recomendaciones.head(num_recomendaciones)

def cliente_mas_compra(material, df = df):
    df_material = df[df['material'] == material]
    clientes_mas_compran = df_material.groupby('id_cliente')['ventas'].sum().sort_values(ascending=False).head(2)
    order = clientes_mas_compran.index
    return str(order[0])

def material_mas_compra(cliente, df = df):
    df_material = df[df['id_cliente'] == cliente]
    clientes_mas_compran = df_material.groupby('material')['ventas'].sum().sort_values(ascending=False).head(2)
    order = clientes_mas_compran.index
    return str(order[0])

clientes = df['id_cliente'].unique()
materiales = df['material'].unique()

#--------------------------------------------------------------------------------------------------------------
# App
#--------------------------------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Ventas por Material/Cliente",
)

st.header("Medicom Business Analytics")
with st.sidebar:
    option = st.selectbox(
        "De qué necesitas información?",
        ("Material", "Cliente"))

    if option == 'Cliente':
        id_ = st.selectbox(
        "ID de cliente a buscar",
        clientes)
    elif option == 'Material':
        id_ = st.selectbox(
            "Nombre de material a buscar",
            materiales)


if option and id_:

    if option == 'Cliente':
        id_ = int(id_)

    if st.button('Procesar'): 
        string__ = "Mostrando informacion del " + option + ' "' + str(id_) + '":'
        st.subheader(string__, divider = 'gray')
#---------------------------------------------------------------------------------------------------------------------
# Análisis sobre material
        if option == "Material":
            if id_ not in materiales:
                string_ = str("El material " + str(id_) + " no existe")
                st.error(string_, icon="🚨")
            else:
                graph = plot_mat(id_)
                plt.savefig('graph.jpeg', bbox_inches='tight', dpi = 300)
                st.image('graph.jpeg')

                recomendaciones = recomendar_materiales_similares(id_)

                st.subheader("Los **:gray-background[clientes que compran]** este material, **:gray-background[suelen también comprar]**:", divider = 'gray')
                
                for item in recomendaciones.index.tolist():
                    st.markdown('- ' + str(item))

                st.subheader("Análisis:", divider = 'gray')

                ld = last_date_mat(id_)
                fd = first_date_mat(id_)

                P = monthly_trans_mat_mat(id_)
                P = np.round(P, 3)

                # material se vendió una vez y se ha vendido desde entonces
                if np.array_equal(np.nan_to_num(P[0]), np.array([0, 0])) and np.array_equal(np.nan_to_num(P[1]), np.array([0, 1])):
                    string_ = 'El producto se vendió por primera vez en ' + str(fd) + ' y **:gray-background[desde entonces se ha vendido]**.'
                    st.markdown(string_)
                    st.markdown('Por lo mismo, no hay tasa de reactivación ni de desactivación, pues el producto **:gray-background[siempre ha estado activo]**.')

                    pi = np.array([0, 1])
                    tmr = 1

                    status = 1

                # material se vendió cada mes hasta Agosto, donde (aún) no se ha vendido
                elif np.array_equal(np.nan_to_num(P[0]), np.array([0, 0])):
                    string_ = 'El producto se vendió por primera vez en ' + str(fd) + ' y **:gray-background[desde entonces ha vendido]**, pero en agosto aún no se ha vendido.'
                    st.markdown(string_)
                    st.markdown('¡No te alarmes! La base de datos tiene registros hasta el 7 de agosto de 2024. Lo más probable es que se **:gray-background[venda en agosto también]**')
                    
                    st.markdown('Por lo mismo, no hay tasa de reactivación ni de desactivación, pues el producto **:gray-background[siempre ha estado activo]**.')

                    pi = np.array([0, 1])
                    tmr = 1

                    status = 1

                # material se vendió por última vez en {fecha} y nunca más se vendió
                elif np.array_equal((P[0]), np.array([1, 0])):
                    string_ = 'El material únicamente se vendió del ' + str(fd) +  ' al ' + str(ld) + ' y desde entonces **:gray-background[no se ha vendido]** :('
                    st.markdown(string_)
                    tmr = 0

                    st.markdown('Este material es **:gray-background[devuelto]** por los clientes el ' + str(proporcion_negativos_mat(id_)) + " % de las veces")
                    status = 2

                else:
                    string_ = 'El material se vendió por última vez en ' + str(ld) + '.'
                    st.write(string_)

                    string_ = 'Basado en nuestro análisis de Cadena de Markov, hasta ahora la **:gray-background[tasa de reactivación]** ha sido de ' + str((np.format_float_positional((100*P[0][1]), 2))) + '% y la **:gray-background[tasa de desactivación]** de ' + str((np.format_float_positional((10*P[1][0]), 2))) + '%.'
                    st.markdown(string_)

                    pi = stationary_distr(P)

                    string_ = 'En general, la **:gray-background[probabilidad de reactivación]** es ' + str(np.format_float_positional(pi[1], 2)) + ' y la de **:gray-background[desactivación]** ' + str(np.format_float_positional(pi[0], 2)) + '.'
                    st.markdown(string_)

                    tmr = t_medio_recurr(pi)

                    st.markdown('Este material es **:gray-background[devuelto]** por los clientes el ' + str(proporcion_negativos_mat(id_)) + " % de las veces")


                    status = 3
                
                if tmr:
                    string_ = 'El material **:gray-background[se vende cada ' + str(np.round(tmr, 2)) + ' meses]** (aproximadamente)'
                    st.markdown(string_)
                else:
                    st.write('El sistema no puede calcular el tiempo medio de recurrencia debido a la  falta de información. ¡Intenta con otro material!')

                st.subheader('Estrategias de negocio:', divider = 'gray')                

                if status == 1:
                    st.markdown('Debido a la frecuencia de venta de este material, se debería mantener un **:gray-background[inventario saludable]** y asegurarse de que **:gray-background[siempre esté disponible]**. Además, enviar promociones destacando este producto como :gray-background[**Best Seller**] podría **:gray-background[impulsar sus ventas]** aún más.')

                elif status == 2:
                    st.markdown('Para este material, se debe analizar primero si se quiere :gray-background[**reactivar**] basado en su :gray-background[**costo de producción**] y otros factores. De ser el caso, se podría recomendar a los clientes que compran productos similares. En el caso contrario, se podría considerar :gray-background[**descontinuar el material**] y liquidar el inventario restante para :gray-background[**liberar espacio**] para :gray-background[**nuevos productos**].')

                elif status == 3:
                    if tmr < 3:
                        st.markdown('Dado que este producto tiene ventas intermitentes, se podría asociar su :gray-background[**promoción con eventos**] o agruparlo con otros productos en :gray-background[**promociones de venta cruzada**]. Además, se debería regular el inventario que se tiene basado en su :gray-background[**demanda variable**].')
                    else:
                        st.markdown('Este producto se podría enviar en :gray-background[**ofertas exclusivas**] a clientes que lo han comprado antes junto con los materiales que le podrían :gray-background[**interesar al cliente**]. Además, se podrían :gray-background[**ajustar los precios**] en los periodos que este :gray-background[**producto está inactivo**]')

            with st.sidebar:
                st.subheader('Resumen', divider = 'gray')

                st.markdown(f'Cliente que más lo compra: :gray-background[**{cliente_mas_compra(id_)}**]')
                st.markdown(f'Última venta: :gray-background[**{ld}**]')
                if tmr == 1:
                    st.markdown(f'Se vende :gray-background[**cada 1 mes**]')
                else:
                    st.markdown(f'Se vende cada :gray-background[**{tmr:.2f} meses**]')

                if status == 3:
                    st.markdown(f'Probabilidad de reactivación: :gray-background[**{str(np.format_float_positional(pi[1], 2))}**]')
                    st.markdown(f'Probabilidad de desactivación: :gray-background[**{str(np.format_float_positional(pi[0], 2))}**]')




#-------------------------------------------------------------------------------------------------------------------
# análisis sobre cliente
        if option == "Cliente":
            if id_ not in clientes:
                string_ = str("El cliente " + str(id_) + " no existe")
                st.error(string_, icon="🚨")

            else:
                graph = plot_cli(id_)
                plt.savefig('graph.jpeg', bbox_inches='tight', dpi = 300)
                st.image('graph.jpeg')

                recomendaciones = recomendar_productos(id_)

                st.subheader("Basado en sus :gray-background[hábitos de compra], a tu cliente le :gray-background[podría interesar]:", divider = 'gray')

                for item in recomendaciones.index.tolist():
                    st.markdown('- ' + str(item))

                st.subheader("Análisis:", divider = 'gray')
                
                ld = last_date_cli(id_)
                fd = first_date_cli(id_)

                P = monthly_trans_mat_cli(id_)
                P = np.round(P, 3)


                # cliente compró una vez y ha comprado desde entonces
                if np.array_equal(np.nan_to_num(P[0]), np.array([0, 0])) and np.array_equal(np.nan_to_num(P[1]), np.array([0, 1])):
                    string_ = 'El cliente compró por primera vez en ' + str(fd) + ' y :gray-background[**desde entonces ha comprado**].'
                    st.markdown(string_)
                    st.markdown('Por lo mismo, no hay tasa de reactivación ni de desactivación, pues el cliente :gray-background[**siempre ha estado activo**].')

                    pi = np.array([0, 1])
                    tmr = 1

                    status = 1

                # cliente compró cada mes hasta Agosto, donde (aún) no ha comprado
                elif np.array_equal(np.nan_to_num(P[0]), np.array([0, 0])):
                    string_ = 'El cliente compró por primera vez en ' + str(fd) + ' y :gray-background[**desde entonces ha comprado**], pero en agosto aún no ha comprado.'
                    st.markdown(string_)
                    st.markdown('¡No te alarmes! La base de datos tiene registros hasta el 7 de agosto de 2024. Lo más probable es que :gray-background[**compre en agosto también**].')
                    
                    st.markdown('Por lo mismo, no hay tasa de reactivación ni de desactivación, pues el cliente :gray-background[**siempre ha estado activo**].')

                    pi = np.array([0, 1])
                    tmr = 1

                    status = 1

                # cliente compró por última vez en {fecha} y nunca más compró
                elif np.array_equal((P[0]), np.array([1, 0])):
                    string_ = 'El cliente únicamente compró del ' + str(fd) + ' al ' + str(ld) + ' y desde entonces :gray-background[**no ha comprado**] :('
                    st.write(string_)
                    tmr = False

                    st.markdown('Este cliente :gray-background[**devuelve**] los materiales que compra el ' + str(proporcion_negativos_cli(id_)) + "% de las veces")

                    status = 2

                else:
                    string_ = 'El cliente compró por última vez en ' + str(ld) + '.'
                    st.markdown(string_)

                    string_ = 'Basado en nuestro análisis de Cadena de Markov, hasta ahora la :gray-background[**tasa de reactivación**] ha sido de ' + str((np.format_float_positional(100*P[0][1], precision = 2))) + '% y la :gray-background[**tasa de desactivación**] de ' + str((np.format_float_positional(100*P[1][0], precision = 2))) + '%.'
                    st.markdown(string_)

                    pi = stationary_distr(P)

                    string_ = 'En general, la :gray-background[**probabilidad de reactivación**] es ' + str(np.format_float_positional(pi[1], precision=2)) + ' y la de :gray-background[**desactivación**] ' + str(np.format_float_positional(pi[0], precision=2)) + '.'
                    st.markdown(string_)

                    tmr = t_medio_recurr(pi)

                    st.markdown('Este cliente :gray-background[**devuelve**] los materiales que compra el ' + str(proporcion_negativos_cli(id_)) + "% de las veces")

                    status = 3
                
                if tmr:
                    string_ = 'El cliente compra :gray-background[**cada ' + str(np.round(tmr, 2)) + ' meses**] (aproximadamente)'
                    st.markdown(string_)
                else:
                    st.markdown('El sistema no puede calcular el tiempo medio de recurrencia debido a la falta de información. ¡Intenta con otro cliente!')

                st.subheader('Estrategias de negocio:', divider = 'gray')
                
                if status == 1:
                    st.markdown('Este :gray-background[**cliente es leal**]. Se podría ofrecerle al cliente :gray-background[**programas de fidelización**] (descuentos exclusivos, puntos de recompensa, etc.) para recompensar su lealtad, y probar :gray-background[**nuevos productos**] con estos clientes para obtener retroalimentación directa.')

                    p10 = 0.01

                elif status == 2:
                    st.markdown('Para este cliente, se podría ya sea implementar una :gray-background[**campaña de reactivación**] (descuentos por regreso, recordarles los productos que más compraban), o implementar una :gray-background[**campaña de feedback**], en la que se envíe una encuesta para entender por qué dejó de comprar y :gray-background[**qué se puede mejorar**] en cuanto a producto o servicio.')

                    p10 = 1

                elif status == 3:
                    if tmr < 3:
                        st.markdown('Para este cliente, se podrían implementar :gray-background[**recordatorios automáticos**]: después de ' + str(np.round(tmr, 2)) + ' meses que el cliente compró, enviarle un correo para asegurar que :gray-background[**siga comprando regularmente**]. También se podrían ofrecer :gray-background[**paquetes o suscripciones**] para asegurarse de que en su tiempo de inactividad :gray-background[**no busque alternativas**].')
                    else:
                        st.markdown('Este cliente tiene una frecuencia de compra más espaciada, por lo que en los meses de inactividad se le podrían enviar :gray-background[**promociones especiales**] (podrían ser los productos que le pueden interesar) para intentar que :gray-background[**compre más frecuentemente**].')

                    p10 = pi[0]

                st.subheader('*Customer Lifetime Value* (CLV):', divider = 'gray')

                clv = np.round(1/p10 * df[df['id_cliente'] == id_]['ventas'].mean(), 2)

                st.markdown('El CLV es el :gray-background[**valor que aporta un cliente**] a la empresa mientras que este :gray-background[**está activo**], y está dado por la sigiuente expresión:')

                st.latex(r'CLV = \mu_0 v ')

                st.latex(r'\mu_0: \: \text{Tiempo medio de recurrencia de inactividad}')
                st.latex(r'v: \: \text{Valor promedio de venta}')

                formatted_amount = "${:,.2f}".format(clv)

                st.markdown(f"El CLV del cliente es:  {formatted_amount}")

            with st.sidebar:
                st.subheader('Resumen', divider = 'gray')

                st.markdown(f'CLV: :gray-background[**{formatted_amount}**]')
                st.markdown(f'Material que más compra: :gray-background[**{material_mas_compra(id_)}**]')
                st.markdown(f'Última compra: :gray-background[**{ld}**]')
                if tmr == 1:
                    st.markdown(f'El cliente compra :gray-background[**cada 1 mes**]')
                else:
                    st.markdown(f'El cliente compra cada :gray-background[**{tmr:.2f} meses**]')

                if status == 3:
                    st.markdown(f'Probabilidad de reactivación: :gray-background[**{(np.format_float_positional(pi[1], 2))}**]')
                    st.markdown(f'Probabilidad de desactivación: :gray-background[**{str(np.format_float_positional(pi[0], 2))}**]')
    
