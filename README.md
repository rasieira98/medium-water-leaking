# Predicción de Consumo de Agua y Deteccion de Alertas por Fuga Utilizando Snowflake y Prophet
El consumo eficiente de agua es un objetivo clave en el desarrollo sostenible. Las técnicas avanzadas de análisis de datos y modelado predictivo pueden desempeñar un papel crucial en la gestión de recursos hídricos. En este artículo, vamos a explorar cómo utilizar Snowflake Cortex ML Functions o la librería Prophet de Python para predecir el consumo de agua. El proceso incluirá la generacion de la prediccion y el calculo de alertas por fuga en el mejor modelo.
## Snowflake Cortex ML Functions
Snowflake es una plataforma de almacenamiento y análisis de datos en la nube. Se destaca por su arquitectura única diseñada específicamente para la nube. Snowflake Cortex ML Functions es una característica de Snowflake que permite a los usuarios implementar modelos de aprendizaje automático directamente dentro de su entorno de Snowflake para realizar análisis predictivos y prescriptivos en los datos almacenados en la plataforma. En este caso nos permite desarrollar un modelo de forecasting simplemente mediante SQL tradicional.
### Paso 1: Agrupar y Agregar Datos
Primero, necesitamos preparar nuestros datos agrupándolos y sumando el flujo de agua por día. Utilizamos el siguiente comando SQL para lograr esto:

```sql
SELECT 
    DATE_TRUNC('DAY', TS_MEASUREMENT), 
    SUM(FLOW) AS FLOW 
FROM 
    TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS
GROUP BY 
    DATE_TRUNC('DAY', TS_MEASUREMENT);

```
### Paso 2: Creación de una Vista Agregada
Para facilitar el manejo de los datos, creamos una vista que agrupa el consumo de agua por día. La vista se ordena por la fecha de medición:

```sql
CREATE OR REPLACE VIEW WATER_CONS_DAILY AS 
SELECT 
    DATE_TRUNC('DAY', TS_MEASUREMENT) AS TS_MEASUREMENT, 
    SUM(FLOW) AS FLOW 
FROM 
    TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS
GROUP BY 
    DATE_TRUNC('DAY', TS_MEASUREMENT) 
ORDER BY 
    DATE_TRUNC('DAY', TS_MEASUREMENT);

```
### Paso 3: Dividir los Datos en Conjuntos de Entrenamiento y Prueba
Contamos el total de registros y calculamos los tamaños de los conjuntos de entrenamiento y prueba, asignando el 80% de los datos al conjunto de entrenamiento y el 20% al conjunto de prueba:

```sql
SELECT 
    COUNT(*) AS TOTAL, 
    ROUND(COUNT(*) * 0.8) AS TRAIN, 
    ROUND(COUNT(*) * 0.2) AS TEST 
FROM 
    WATER_CONS_DAILY;

```

### Paso 4: Crear una Vista para el Conjunto de Entrenamiento
Para el entrenamiento del modelo, creamos una vista con los primeros 397 registros (el 80% del total de datos):

```sql
CREATE OR REPLACE VIEW WATER_CONS_TRAIN AS 
SELECT * 
FROM WATER_CONS_DAILY
ORDER BY TS_MEASUREMENT
LIMIT 397;
```
### Paso 5: Construcción del Modelo de Previsión en Snowflake
Utilizamos el motor de previsión de Snowflake para crear y entrenar un modelo de previsión con los datos de entrenamiento:

```sql
CREATE OR REPLACE SNOWFLAKE.ML.FORECAST model1 (
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'WATER_CONS_TRAIN'),
    TIMESTAMP_COLNAME => 'TS_MEASUREMENT',
    TARGET_COLNAME => 'FLOW'
);

BEGIN
    -- Ejecuta el modelo de previsión para 99 períodos de tiempo
    CALL model1!FORECAST(FORECASTING_PERIODS => 99);
    LET x := SQLID;
    
    -- Crea o reemplaza una tabla con los resultados de la previsión
    CREATE OR REPLACE TABLE TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_FORECAST AS 
    SELECT * 
    FROM TABLE(RESULT_SCAN(:x));
END;
```
## Implementación de Prophet en Python
Además de utilizar las capacidades de Snowflake, implementamos un modelo de previsión con Prophet en Python para mejorar resultados. A continuación, se muestra el código para entrenar un modelo Prophet con los datos de Snowflake:

```python
import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
from prophet import Prophet
import pandas as pd

def main(session: snowpark.Session): 
    # Nombre de la tabla en Snowflake
    source_table_name = 'TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_TRAIN'
    
    # Obtener los datos de la tabla en Snowflake
    dataframe = session.table(source_table_name)

    # Convertir el DataFrame de Snowflake a Pandas
    df = dataframe.toPandas()
    df = df.sort_values(by='TS_MEASUREMENT')
    
    # Creamos un modelo Prophet
    model = Prophet()

    # Renombramos las columnas a 'ds' (TS_MEASUREMENT) y 'y' (FLOW) según los datos
    df = df.rename(columns={'TS_MEASUREMENT': 'ds', 'FLOW': 'y'})
    
    # Ajustamos el modelo a nuestros datos
    model.fit(df)

    # Creamos un DataFrame con fechas futuras para hacer predicciones
    future_dates = model.make_future_dataframe(periods=99, freq='1d', include_history=True)

    # Hacemos predicciones
    forecast = model.predict(future_dates)
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast = forecast.rename(columns={'ds': 'TS', 'yhat': 'FORECAST', 'yhat_lower': 'LOWER_BOUND', 'yhat_upper': 'UPPER_BOUND'})
    
    # Convertir el DataFrame de Pandas de las predicciones a un DataFrame de Snowpark
    forecast_snowpark = session.createDataFrame(forecast)
    
    # Guardar las predicciones en una tabla en Snowflake
    forecast_snowpark.write.mode("overwrite").save_as_table("TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_PROPHET")
    
    return forecast_snowpark
```
## Manejo de Datos Faltantes
Para abordar el problema de los datos faltantes, creamos una vista que introduce huecos significativos en los datos y repetimos el proceso de previsión:
### Snowflake Cortex ML Functions 
```sql
-- Selecciona y agrupa los datos por hora, sumando el flujo de agua
SELECT DATE_TRUNC('DAY', TS_MEASUREMENT), SUM(FLOW) AS FLOW 
FROM TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS
GROUP BY DATE_TRUNC('DAY', TS_MEASUREMENT);

-- Crea o reemplaza una vista que agrupa el consumo de agua por hora
CREATE OR REPLACE VIEW WATER_CONS_DAILY_BIG_GAPS AS 
SELECT 
    DATE_TRUNC('DAY', TS_MEASUREMENT) AS TS_MEASUREMENT, 
    CASE 
        WHEN DATE_TRUNC('DAY', TS_MEASUREMENT) BETWEEN '2021-08-01' AND '2021-08-15' THEN NULL
        WHEN DATE_TRUNC('DAY', TS_MEASUREMENT) BETWEEN '2021-02-15' AND '2021-02-28' THEN NULL
        ELSE SUM(FLOW) 
    END AS FLOW 
FROM TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS
GROUP BY DATE_TRUNC('DAY', TS_MEASUREMENT) 
ORDER BY DATE_TRUNC('DAY', TS_MEASUREMENT);

-- Cuenta el total de registros y calcula los conjuntos de entrenamiento y prueba
SELECT COUNT(*) AS TOTAL, 
       ROUND(COUNT(*) * 0.8) AS TRAIN, 
       ROUND(COUNT(*) * 0.2) AS TEST 
FROM WATER_CONS_DAILY_BIG_GAPS;

-- Crea o reemplaza una vista con los primeros 397 registros para el conjunto de entrenamiento
CREATE OR REPLACE VIEW WATER_CONS_TRAIN_BIG_GAPS AS 
SELECT * 
FROM WATER_CONS_DAILY_BIG_GAPS
ORDER BY TS_MEASUREMENT
LIMIT 397;

-- Crea un modelo de previsión utilizando los datos horarios de entrenamiento de consumo de agua
CREATE OR REPLACE SNOWFLAKE.ML.FORECAST model1(
  INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'WATER_CONS_TRAIN_BIG_GAPS'),
  TIMESTAMP_COLNAME => 'TS_MEASUREMENT',
  TARGET_COLNAME => 'FLOW'
);

BEGIN
    -- Ejecuta el modelo de previsión para 99 períodos de tiempo
    CALL model1!FORECAST(FORECASTING_PERIODS => 99);
    LET x := SQLID;
    
    -- Crea o reemplaza una tabla con los resultados de la previsión
    CREATE OR REPLACE TABLE TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_FORECAST_BIG_GAPS AS 
    SELECT * 
    FROM TABLE(RESULT_SCAN(:x));
END;

-- Selecciona todos los registros de la tabla de previsiones
SELECT * 
FROM TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_FORECAST_BIG_GAPS;
```
### Implementación de Prophet en Python
```python
import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
from prophet import Prophet
import pandas as pd

def main(session: snowpark.Session): 
    # Nombre de la tabla en Snowflake
    source_table_name = 'TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_TRAIN'
    
    # Obtener los datos de la tabla en Snowflake
    dataframe = session.table(source_table_name)

    # Convertir el DataFrame de Snowflake a Pandas
    df = dataframe.toPandas()
    df = df.sort_values(by='TS_MEASUREMENT')
    
    # Creamos un modelo Prophet
    model = Prophet()

    # Renombramos las columnas a 'ds' (TS_MEASUREMENT) y 'y' (FLOW) según los datos
    df = df.rename(columns={'TS_MEASUREMENT': 'ds', 'FLOW': 'y'})
    
    # Ajustamos el modelo a nuestros datos
    model.fit(df)

    # Creamos un DataFrame con fechas futuras para hacer predicciones
    future_dates = model.make_future_dataframe(periods=99, freq='1d', include_history=True)

    # Hacemos predicciones
    forecast = model.predict(future_dates)
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast = forecast.rename(columns={'ds': 'TS', 'yhat': 'FORECAST', 'yhat_lower': 'LOWER_BOUND', 'yhat_upper': 'UPPER_BOUND'})
    
    # Convertir el DataFrame de Pandas de las predicciones a un DataFrame de Snowpark
    forecast_snowpark = session.createDataFrame(forecast)

    # Guardar las predicciones en una tabla en Snowflake
    forecast_snowpark.write.mode("overwrite").save_as_table("TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_PROPHET")
    
    return forecast_snowpark
```
## Generacion de Alertas por Fuga con el forecast de Prophet:
```python
import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
import pandas as pd

def main(session: snowpark.Session): 
    # Nombre de la tabla en Snowflake
    real_table = 'TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_DAILY_BIG_GAPS'
    predicciones_table = 'TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_PROPHET_BIG_GAPS'
    
    # Obtener los datos de la tabla en Snowflake
    real = session.table(real_table)
    predicciones = session.table(predicciones_table)

    # Convertir el DataFrame de Snowflake a Pandas
    real = real.toPandas()
    real = real.sort_values(by='TS_MEASUREMENT')

    predicciones = predicciones.toPandas()
    predicciones = predicciones.rename(columns={'TS': 'TS_MEASUREMENT'})
    predicciones = predicciones.sort_values(by='TS_MEASUREMENT')
    
    # Unir los DataFrames por 'TS_MEASUREMENT'
    df = pd.merge(real, predicciones, on='TS_MEASUREMENT')
    print(df)
    # Añadir columna de alerta
    df['ALERT'] = df['FLOW'] > 1.1*df['FORECAST']
    df=df[['TS_MEASUREMENT','ALERT']]
    # Convertir el DataFrame de Pandas a un DataFrame de Snowpark
    df = session.create_dataframe(df)
    
    # Guardar los datos en una tabla en Snowflake (descomentar si necesario)
    df.write.mode("overwrite").save_as_table("TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_PROPHET_EXP")
    
    return df
```
## Visualizacion de resultados

