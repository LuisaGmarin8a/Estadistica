import streamlit as st
import pandas as pd
import scipy.stats as stats
from scipy.stats import shapiro, kstest, normaltest
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide")

# Cargar el archivo CSV
df = pd.read_csv("BelleAyrLiquefuctionRuns.csv", delimiter=";")
df.drop(columns=["num"], inplace=True)
df = df.replace({',': '.'}, regex=True).apply(pd.to_numeric, errors='coerce')

page = st.sidebar.selectbox("proceso: ", ["Regresion a lo maldita sea","EDA","Regresion EDA"],)


def evaluate_rmse_mse(y_true, y_pred):
        """Calcula el MSE y el RMSE dados los valores reales y los predichos.
        
        Args:
            y_true: Valores reales (observaciones).
            y_pred: Valores predichos por el modelo.
            
        Returns:
            Tuple: (MSE, RMSE)
        """
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)

        return mse, rmse

# Funciones de EDA
def normality_tests(data, column):
    if pd.api.types.is_numeric_dtype(data[column]):
        col_data = data[column].dropna()
        shapiro_stat, shapiro_p = shapiro(col_data)
        ks_stat, ks_p = kstest(col_data, 'norm')
        dagostino_stat, dagostino_p = normaltest(col_data)
        
        result = {
            'Prueba': ['Shapiro-Wilk', 'Kolmogorov-Smirnov', "D'Agostino"],
            'Estadístico': [shapiro_stat, ks_stat, dagostino_stat],
            'p-valor': [shapiro_p, ks_p, dagostino_p]
        }
        
        return pd.DataFrame(result)
    else:
        return f"La columna '{column}' no es de tipo numérico."

# QQ-Plot con Plotly
def plot_qqplot(data, column):
    if pd.api.types.is_numeric_dtype(data[column]):
        col_data = data[column].dropna()
        (osm, osr), (slope, intercept, r) = stats.probplot(col_data, dist="norm", plot=None)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Datos observados'))
        fig.add_trace(go.Scatter(x=osm, y=slope * osm + intercept, mode='lines', name='Línea de referencia'))

        fig.update_layout(
            title=f'QQ-Plot de {column}',
            xaxis_title='Cuantiles Teóricos',
            yaxis_title='Cuantiles Observados',
            height=400,
            width=600
        )

        st.plotly_chart(fig)
    else:
        st.write(f"La columna '{column}' no es de tipo numérico.")

# Scatter Plot entre dos variables con Plotly
def scatter_plot_between_variables(data, var1, var2):
    if var1 in data.columns and var2 in data.columns:
        fig = px.scatter(data_frame=data, x=var1, y=var2, opacity=0.5, title=f'Scatter Plot: {var1} vs {var2}',
                         labels={var1: var1, var2: var2})
        
        fig.update_layout(
            title=f'Scatter Plot entre {var1} y {var2}',
            xaxis_title=var1,
            yaxis_title=var2,
            #height=600,
            #width=800
        )

        st.plotly_chart(fig)
    else:
        st.write("Las variables proporcionadas no están en el dataframe.")

# Correlaciones entre dos variables
def correlation_between_two(data, var1, var2):
    if pd.api.types.is_numeric_dtype(data[var1]) and pd.api.types.is_numeric_dtype(data[var2]):
        pearson_corr = data[var1].corr(data[var2], method='pearson')
        kendall_corr = data[var1].corr(data[var2], method='kendall')
        spearman_corr = data[var1].corr(data[var2], method='spearman')
        
        result = {
            'Correlación': ['Pearson', 'Kendall', 'Spearman'],
            'Valor': [pearson_corr, kendall_corr, spearman_corr]
        }
        
        return pd.DataFrame(result)
    else:
        return f"Ambas variables '{var1}' y '{var2}' deben ser numéricas."

# Función para graficar la matriz de correlación
def plot_correlation_matrix(data, method='pearson'):
    if method not in ['pearson', 'kendall', 'spearman']:
        raise ValueError("El método debe ser 'pearson', 'kendall' o 'spearman'.")
    
    # Calcular la matriz de correlación en función del método seleccionado
    corr_matrix = data.corr(method=method)
    
    # Crear el heatmap con Plotly
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values, 
        x=corr_matrix.columns, 
        y=corr_matrix.columns, 
        colorscale='RdBu', 
        zmin=-1, 
        zmax=1,
        colorbar=dict(title="Correlación"),
        hoverongaps=False
    ))

    # Añadir los valores numéricos en los recuadros
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            fig.add_annotation(dict(
                x=corr_matrix.columns[i],
                y=corr_matrix.columns[j],
                text=str(round(corr_matrix.iloc[j, i], 2)),
                showarrow=False,
                font=dict(color="black")
            ))

    # Actualizar el layout para mejorar el diseño
    fig.update_layout(
        title=f'Matriz de Correlación ({method.capitalize()})',
        xaxis_nticks=36,
        width=800, 
        height=700,
        yaxis_autorange='reversed'
    )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig)

def evaluate_influential_leverage(model):
    """Evalúa los puntos de leverage e influyentes del modelo.
    
    Args:
        model: El modelo de regresión ajustado.
        
    Returns:
        Tuple: (Lista de índices de puntos de alto leverage, Lista de índices de puntos influyentes)
    """
    # Cálculo de leverage
    leverage = model.get_influence().hat_matrix_diag
    # Cálculo de Cook's Distance
    cooks_d = model.get_influence().cooks_distance[0]

    # Número de observaciones y parámetros
    n = model.nobs
    p = model.params.shape[0]

    # Definir umbrales
    threshold_leverage = 2 * (p / n)
    threshold_cooks = 4 / n

    # Identificar puntos de alto leverage
    high_leverage_points = np.where(leverage > threshold_leverage)[0].tolist()

    # Identificar puntos influyentes
    influential_points = np.where(cooks_d > threshold_cooks)[0].tolist()

    return high_leverage_points, influential_points


def evaluate_leverage_points(X, high_leverage_points):
    """Evalúa los puntos de alto leverage en relación con la media y desviación estándar de cada variable.
    
    Args:
        X: DataFrame de las variables independientes.
        high_leverage_points: Lista de índices de los puntos de alto leverage.
        
    Returns:
        DataFrame: Información sobre la media, desviación estándar y distancia en desviaciones de cada variable.
    """
    results = []

    for idx in high_leverage_points:
        for col in X.columns:
            mean = X[col].mean()
            std_dev = X[col].std()
            point_value = X[col].iloc[idx]
            distance = (point_value - mean) / std_dev
            
            results.append({
                "Var": col,
                "Var_Media": mean,
                "Var_Desviacion": std_dev,
                "Punto": point_value,
                "Distancia(Desviaciones)": distance
            })

    return pd.DataFrame(results)


def plot_leverage_influence(X, model, high_leverage_points, influential_points):
    """Genera un gráfico de dispersión para visualizar los puntos de leverage e influyentes.
    
    Args:
        X: DataFrame de las variables independientes.
        model: El modelo de regresión ajustado.
        high_leverage_points: Lista de índices de los puntos de alto leverage.
        influential_points: Lista de índices de los puntos influyentes.
    """
    # Obtener los valores ajustados y residuos
    fitted_values = model.fittedvalues
    residuals = model.resid

    import plotly.graph_objects as go
    import streamlit as st

    # Supongamos que 'fitted_values', 'residuals', 'high_leverage_points', y 'influential_points' están definidos
    # Estos son arrays o listas con los valores correspondientes

    # Crear el gráfico con Plotly
    fig = go.Figure()

    # Gráfico de dispersión de residuos vs valores ajustados
    fig.add_trace(go.Scatter(
        x=fitted_values,
        y=residuals,
        mode='markers',
        name='Datos',
        marker=dict(color='blue', opacity=0.5),
        showlegend=True
    ))

    # Identificar puntos de alto leverage
    fig.add_trace(go.Scatter(
        x=fitted_values[high_leverage_points],
        y=residuals[high_leverage_points],
        mode='markers',
        name='Puntos de alto leverage',
        marker=dict(color='orange', size=10, line=dict(color='black', width=1)),
        showlegend=True
    ))

    # Identificar puntos influyentes
    fig.add_trace(go.Scatter(
        x=fitted_values[influential_points],
        y=residuals[influential_points],
        mode='markers',
        name='Puntos influyentes',
        marker=dict(color='red', size=10, line=dict(color='black', width=1)),
        showlegend=True
    ))

    # Agregar línea horizontal en y=0
    fig.add_trace(go.Scatter(
        x=[min(fitted_values), max(fitted_values)],
        y=[0, 0],
        mode='lines',
        line=dict(color='black', dash='dash', width=0.8),
        name='Línea 0'
    ))

    # Actualizar el diseño del gráfico
    fig.update_layout(
        title='Residuos vs Valores Ajustados',
        xaxis_title='Valores Ajustados',
        yaxis_title='Residuos',
        legend_title='Leyenda',
    )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig)




# Opciones de análisis
if page == "EDA":

    st.title("Información del Contenido")

    st.subheader("Variables del Modelo")

    # Usar markdown para formatear
    content = """
    - **y:** CO₂: Esta variable representa la cantidad de dióxido de carbono (CO₂) producido durante las corridas de licuefacción en Belle Ayr.
    - **x₁:** Space time (min): El "space time" mide el tiempo promedio que el carbón pasa en el reactor durante el proceso de licuefacción
    - **x₂:** Temperature (°C) : La temperatura en la que se lleva a cabo el proceso de licuefacción
    - **x₃:** Percent solvation: Este porcentaje indica la cantidad de solvente presente en la mezcla durante el proceso de licuefacción
    - **x₄:** Oil yield (g/100 g MAF): El rendimiento de aceite se mide en gramos de aceite producido por cada 100 gramos de materia prima
    - **x₅:** Coal total: Representa la cantidad total de carbón en la mezcla utilizada para la licuefacción
    - **x₆:** Solvent total: Indica la cantidad total de solvente utilizado en el proceso de licuefacción
    - **x₇:** Hydrogen consumption: Se refiere a la cantidad de hidrógeno consumido durante el proceso de licuefacción
    
    La licuefacción es un proceso físico y químico mediante el cual un material, típicamente un gas, se convierte en líquido
    """

    st.markdown(content)


    col1, col2 = st.columns(2)

    with col1:
        st.title('Exploratory Data Analysis (EDA)')
        analysis_type = st.selectbox('Selecciona el tipo de análisis:', ['Univariado', 'Bivariado', 'Multivariado'])

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Análisis Univariado
    if analysis_type == 'Univariado':
        
        with col2:
            st.title('Análisis Univariado')
            
            # Selección de la variable
            selected_column = st.selectbox('Selecciona una columna para el análisis univariado', numeric_columns)
        
        with col1:
                st.subheader(f'Pruebas de Normalidad para {selected_column}')
        normality_result = normality_tests(df, selected_column)

        # Verificamos si el resultado es un DataFrame
            # Estilo de la tabla
        styled_table = (
            normality_result.style
            .format(precision=4)  # Formato de números
            .set_table_attributes('style="width: 100%; background-color: #f9f9f9;"')  # Ajuste de ancho
            .set_properties(**{'background-color': '#f2f2f2', 'color': 'black'})  # Color de fondo
            .set_table_styles([{
                'selector': 'thead th',
                'props': [('background-color', '#007bff'), ('color', 'white')]
            }, {
                'selector': 'tbody tr:nth-child(even)',
                'props': [('background-color', '#e0e0e0')]
            }, {
                'selector': 'tbody tr:nth-child(odd)',
                'props': [('background-color', '#f9f9f9')]
            }])
        )

        # Mostrar la tabla
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(styled_table, use_container_width=True)  # Ajustar a todo el ancho

            # Evaluar los resultados de las pruebas de normalidad
            p_values = normality_result['p-valor']
            normal_count = sum(p > 0.05 for p in p_values)  # Cuenta las pruebas que indican normalidad
            
            # Indicador de normalidad
            if normal_count >= 2:
                st.success("Distribución Normal: ✔️", icon="✅")  # Indicador verde
            else:
                st.error("Distribución No Normal: ❌", icon="🚫")  # Indicador rojo


        with col2:
            plot_qqplot(df, selected_column)


    # Análisis Bivariado
    elif analysis_type == 'Bivariado':
        with col2:
            selected_var1 = st.selectbox('Selecciona la primera variable', numeric_columns)
            selected_var2 = st.selectbox('Selecciona la segunda variable', numeric_columns)
        
        # Mostrar correlaciones en una tabla
        st.subheader(f'Correlaciones entre {selected_var1} y {selected_var2}')
        correlation_result = correlation_between_two(df, selected_var1, selected_var2)

        if isinstance(correlation_result, pd.DataFrame):
            # Estilo de la tabla
            styled_corr_table = (
                correlation_result.style
                .format(precision=4)  # Formato de números
                .set_table_attributes('style="width: 100%; background-color: #f9f9f9;"')  # Ajuste de ancho
                .set_properties(**{'background-color': '#f2f2f2', 'color': 'black'})  # Color de fondo
                .set_table_styles([{
                    'selector': 'thead th',
                    'props': [('background-color', '#007bff'), ('color', 'white')]
                }, {
                    'selector': 'tbody tr:nth-child(even)',
                    'props': [('background-color', '#e0e0e0')]
                }, {
                    'selector': 'tbody tr:nth-child(odd)',
                    'props': [('background-color', '#f9f9f9')]
                }])
            )

            col1,col2 = st.columns(2)
            with col1:
                st.dataframe(styled_corr_table, use_container_width=True)

                # Evaluar si las variables están correlacionadas
                correlation_values = correlation_result['Valor']
                high_corr_count = sum(abs(val) >= 0.8 for val in correlation_values)  # Cuenta los coeficientes altos
                
                # Indicador de correlación
                if high_corr_count >= 2:
                    st.success("Las variables están fuertemente correlacionadas: ✔️", icon="✅")  # Indicador verde
                else:
                    st.warning("Las variables no están fuertemente correlacionadas: ⚠️", icon="⚠️")  # Indicador de advertencia

        else:
            st.write(correlation_result)

        # Mostrar Scatter Plot
        with col2:
            scatter_plot_between_variables(df, selected_var1, selected_var2)


    # Análisis Multivariado
    elif analysis_type == 'Multivariado':
        st.header('Análisis Multivariado')
        
        with col2:
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            method = st.selectbox("Selecciona el método de correlación:", ["pearson", "kendall", "spearman"])
        
        # Mostrar matriz de correlación
        st.subheader(f'Matriz de Correlación ({method.capitalize()})')
        plot_correlation_matrix(df, method)

if page == "Regresion a lo maldita sea":

    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    x = df.drop(columns=["y"])

#    x.drop(index=[13],inplace=True)

    y = df["y"]#.drop(index=[13])

    X = sm.add_constant(x)
    mod = sm.OLS(y, X).fit() ### Minimos cuadrados

    # Supongamos que 'mod' es tu modelo y tienes sus valores ajustados y residuos
    fitted_values = mod.fittedvalues
    residuals = mod.resid

    # Crear el gráfico de dispersión
    fig1 = go.Figure()

    # Agregar los puntos de dispersión
    fig1.add_trace(go.Scatter(
        x=fitted_values,
        y=residuals,
        mode='markers',
        marker=dict(color='darkseagreen'),
        name='Residuos'
    ))

    # Agregar la línea horizontal en y=0
    fig1.add_trace(go.Scatter(
        x=[min(fitted_values), max(fitted_values)],
        y=[0, 0],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Línea y=0'
    ))

    # Actualizar el diseño del gráfico
    fig1.update_layout(
        title='Residuos vs. Valores ajustados',
        xaxis_title='Valores ajustados',
        yaxis_title='Residuos',
    )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig1)

    residuals = mod.resid

    # Calcular los cuantiles teóricos y los cuantiles de residuos
    sm.qqplot(residuals, line='45', ax=plt.gca())
    quantiles = np.linspace(0, 1, len(residuals))
    theoretical_quantiles = np.percentile(np.random.normal(0, 1, 10000), quantiles * 100)
    observed_quantiles = np.sort(residuals)

    # Crear el gráfico Q-Q con Plotly
    fig = go.Figure()

    # Agregar puntos de los cuantiles observados
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=observed_quantiles,
        mode='markers',
        name='Cuantiles observados'
    ))

    # Agregar la línea de referencia
    fig.add_trace(go.Scatter(
        x=[min(theoretical_quantiles), max(theoretical_quantiles)],
        y=[min(theoretical_quantiles), max(theoretical_quantiles)],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Línea de referencia'
    ))

    # Actualizar el diseño del gráfico
    fig.update_layout(
        title='Normalidad: Gráfico Q-Q de los residuos',
        xaxis_title='Cuantiles teóricos',
        yaxis_title='Cuantiles de residuos',
        showlegend=True
    )

    with col2:
        st.plotly_chart(fig)


    import streamlit as st
    import pandas as pd
    import statsmodels.api as sm

    # Supongamos que 'mod' es tu modelo ajustado
    # Convertir el resumen del modelo a un DataFrame
    summary_df = pd.DataFrame(mod.summary().tables[1].data[1:], columns=mod.summary().tables[1].data[0])

    # Convertir a tipo numérico las columnas que deben ser numéricas
    summary_df.iloc[:, 1:] = summary_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    # Mostrar el resumen en una tabla de Streamlit
    st.write("### Resumen del Modelo")

    col1, col2 = st.columns(2)

    with col1:
        st.table(summary_df)

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif = pd.DataFrame()
    vif['Variable'] = ['Constante'] + [f'X{i+1}' for i in range(x.shape[1])]
    vif['VIF'] = [variance_inflation_factor(X,i) for i in range(X.shape[1])]
    
    with col2:
        st.table(vif)

    col1,col2,col3 = st.columns(3)

    from statsmodels.stats.stattools import durbin_watson
    dw = durbin_watson(mod.resid)
    
    with col1:
        st.write(f"Estadístico Durbin-Watson: {dw}, autocorrelacion entre los residuos")
        predictions = mod.predict(X)
        mse, rmse = evaluate_rmse_mse(y, predictions)
        st.write(f'MSE: {mse}')
        st.write(f'RMSE: {rmse}')

    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_test = het_breuschpagan(mod.resid, X)
    lm_statistic = bp_test[0]  # Estadístico LM
    lm_pvalue = bp_test[1]
    with col2:
        st.write(f"LM Statistic: {lm_statistic}")
        st.write(f"LM Statistic: {lm_pvalue}")
        st.write("Errores NO son homocedásticos")

    from scipy import stats

    shapiro_test = stats.shapiro(mod.resid)
    with col3:
        st.write(f"Prueba de Shapiro-Wilk: p-valor={shapiro_test[1]}")
        r2 = mod.rsquared
        r2_ajustado = mod.rsquared_adj
        st.write(f"R2 = {r2}")
        st.write(f"R2 ajustado = {r2_ajustado}")
    
    high_leverage_points, influential_points = evaluate_influential_leverage(mod)
    print("Puntos de alto leverage:", high_leverage_points)
    #print("Puntos influyentes:", influential_points)

    # Evaluar puntos de alto leverage
    leverage_info = evaluate_leverage_points(X, high_leverage_points)
    #print(leverage_info)

    # Graficar los puntos de leverage e influyentes
    plot_leverage_influence(X, mod, high_leverage_points, influential_points)

if page == "Regresion EDA":

    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from sklearn.preprocessing import StandardScaler

    x_eda = df.drop(columns=["y","x3","x1","x5"])

    scaler = StandardScaler()
    x_eda_transform = scaler.fit_transform(x_eda)
    x_eda = pd.DataFrame(x_eda_transform,columns=x_eda.columns)

    y = df["y"]#.drop(index=[13])

    X_eda = sm.add_constant(x_eda)
    mod_eda = sm.OLS(y, X_eda).fit() ### Minimos cuadrados

    # Supongamos que 'mod' es tu modelo y tienes sus valores ajustados y residuos
    fitted_values = mod_eda.fittedvalues
    residuals = mod_eda.resid

    # Crear el gráfico de dispersión
    fig1 = go.Figure()

    # Agregar los puntos de dispersión
    fig1.add_trace(go.Scatter(
        x=fitted_values,
        y=residuals,
        mode='markers',
        marker=dict(color='darkseagreen'),
        name='Residuos'
    ))

    # Agregar la línea horizontal en y=0
    fig1.add_trace(go.Scatter(
        x=[min(fitted_values), max(fitted_values)],
        y=[0, 0],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Línea y=0'
    ))

    # Actualizar el diseño del gráfico
    fig1.update_layout(
        title='Residuos vs. Valores ajustados',
        xaxis_title='Valores ajustados',
        yaxis_title='Residuos',
    )

    col1, col2 = st.columns(2)

    with col1:
    # Mostrar el gráfico
        st.plotly_chart(fig1)

    residuals = mod_eda.resid

    # Calcular los cuantiles teóricos y los cuantiles de residuos
    sm.qqplot(residuals, line='45', ax=plt.gca())
    quantiles = np.linspace(0, 1, len(residuals))
    theoretical_quantiles = np.percentile(np.random.normal(0, 1, 10000), quantiles * 100)
    observed_quantiles = np.sort(residuals)

    # Crear el gráfico Q-Q con Plotly
    fig = go.Figure()

    # Agregar puntos de los cuantiles observados
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=observed_quantiles,
        mode='markers',
        name='Cuantiles observados'
    ))

    # Agregar la línea de referencia
    fig.add_trace(go.Scatter(
        x=[min(theoretical_quantiles), max(theoretical_quantiles)],
        y=[min(theoretical_quantiles), max(theoretical_quantiles)],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Línea de referencia'
    ))

    # Actualizar el diseño del gráfico
    fig.update_layout(
        title='Normalidad: Gráfico Q-Q de los residuos',
        xaxis_title='Cuantiles teóricos',
        yaxis_title='Cuantiles de residuos',
        showlegend=True
    )

    # Mostrar el gráfico en Streamlit
    with col2:
        st.plotly_chart(fig)


    import streamlit as st
    import pandas as pd
    import statsmodels.api as sm

    # Supongamos que 'mod' es tu modelo ajustado
    # Convertir el resumen del modelo a un DataFrame
    summary_df = pd.DataFrame(mod_eda.summary().tables[1].data[1:], columns=mod_eda.summary().tables[1].data[0])

    # Convertir a tipo numérico las columnas que deben ser numéricas
    summary_df.iloc[:, 1:] = summary_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    # Mostrar el resumen en una tabla de Streamlit
    st.write("### Resumen del Modelo")
    col1, col2 = st.columns(2)
    
    with col1:
        st.table(summary_df)

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif = pd.DataFrame()
    vif['Variable'] = ['Constante'] + [i for i in x_eda.columns]
    vif['VIF'] = [variance_inflation_factor(X_eda,i) for i in range(len(X_eda.columns))]
    with col2:
        st.table(vif)

    from statsmodels.stats.stattools import durbin_watson
    
    col1,col2,col3 = st.columns(3)

    dw = durbin_watson(mod_eda.resid)

    with col1:
        st.write(f"Estadístico Durbin-Watson: {dw}, autocorrelacion entre los residuos")
        predictions = mod_eda.predict(X_eda)
        mse, rmse = evaluate_rmse_mse(y, predictions)
        st.write(f'MSE: {mse}')
        st.write(f'RMSE: {rmse}')

    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_test = het_breuschpagan(mod_eda.resid, X_eda)
    lm_statistic = bp_test[0]  # Estadístico LM
    lm_pvalue = bp_test[1]
    with col2:
        st.write(f"LM Statistic: {lm_statistic}")
        st.write(f"LM Statistic: {lm_pvalue}")
        st.write("Errores son homocedásticos")

    from scipy import stats

    shapiro_test = stats.shapiro(mod_eda.resid)
    with col3:  
        st.write(f"Prueba de Shapiro-Wilk: p-valor={shapiro_test[1]}")
        r2 = mod_eda.rsquared
        r2_ajustado = mod_eda.rsquared_adj
        st.write(f"R2 = {r2}")
        st.write(f"R2 ajustado = {r2_ajustado}")


    high_leverage_points, influential_points = evaluate_influential_leverage(mod_eda)
    print("Puntos de alto leverage:", high_leverage_points)
    #print("Puntos influyentes:", influential_points)

    # Evaluar puntos de alto leverage
    leverage_info = evaluate_leverage_points(X_eda, high_leverage_points)
    #print(leverage_info)

    # Graficar los puntos de leverage e influyentes
    plot_leverage_influence(X_eda, mod_eda, high_leverage_points, influential_points)