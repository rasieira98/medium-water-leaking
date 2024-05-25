# Prediction of Water Consumption and Leak Alert Detection Using Snowflake and Prophet
Efficient water consumption is a key goal in sustainable development. Advanced data analysis techniques and predictive modeling can play a crucial role in water resource management. In this article, we will explore how to use Snowflake Cortex ML Functions or Python's Prophet library to predict water consumption. The process will include generating the prediction and calculating leak alerts in the best model.

### Step 1: Group and Aggregate Data
First, we need to prepare our data by grouping it and summing the water flow per day. We use the following SQL command to achieve this:

```sql
SELECT 
    DATE_TRUNC('DAY', TS_MEASUREMENT), 
    SUM(FLOW) AS FLOW 
FROM 
    TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS
GROUP BY 
    DATE_TRUNC('DAY', TS_MEASUREMENT);

```
### Step 2: Creating an Aggregated View
To simplify data management, we create a view that groups water consumption by day. The view is ordered by the measurement date:

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
### Step 3: Splitting the Data into Training and Testing Sets
We count the total number of records and calculate the sizes of the training and testing sets, assigning 80% of the data to the training set and 20% to the testing set:

```sql
SELECT 
    COUNT(*) AS TOTAL, 
    ROUND(COUNT(*) * 0.8) AS TRAIN, 
    ROUND(COUNT(*) * 0.2) AS TEST 
FROM 
    WATER_CONS_DAILY;

```

### Step 4: Creating a View for the Training Set
For model training, we create a view with the first 397 records (80% of the total data):

```sql
CREATE OR REPLACE VIEW WATER_CONS_TRAIN AS 
SELECT * 
FROM WATER_CONS_DAILY
ORDER BY TS_MEASUREMENT
LIMIT 397;
```
### Step 5: Building the Forecasting Model in Snowflake
We use Snowflake's forecasting engine to create and train a forecasting model with the training data:

```sql
CREATE OR REPLACE SNOWFLAKE.ML.FORECAST model1 (
    INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'WATER_CONS_TRAIN'),
    TIMESTAMP_COLNAME => 'TS_MEASUREMENT',
    TARGET_COLNAME => 'FLOW'
);

BEGIN
    -- Execute the forecasting model for 99 time periods
    CALL model1!FORECAST(FORECASTING_PERIODS => 99);
    LET x := SQLID;
    
    -- Create or replace a table with the forecast results
    CREATE OR REPLACE TABLE TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_FORECAST AS 
    SELECT * 
    FROM TABLE(RESULT_SCAN(:x));
END;
```
![image](https://github.com/rasieira98/medium-water-leaking/assets/116558787/84923ba7-4f49-4aa3-8738-9f9718922e8e)

## Implementation with Prophet and Python
In addition to utilizing Snowflake's capabilities, we implement a forecasting model with Prophet in Python to enhance results. Below is the code for training a Prophet model with the data from Snowflake:

```python
import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
from prophet import Prophet
import pandas as pd

def main(session: snowpark.Session): 
    # Table name in Snowflake
    source_table_name = 'TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_TRAIN'
    
    # Fetch data from Snowflake table
    dataframe = session.table(source_table_name)

    # Convert Snowflake DataFrame to Pandas
    df = dataframe.toPandas()
    df = df.sort_values(by='TS_MEASUREMENT')
    
    # Create a Prophet model
    model = Prophet()

    # Rename columns to 'ds' (TS_MEASUREMENT) and 'y' (FLOW) as per the data
    df = df.rename(columns={'TS_MEASUREMENT': 'ds', 'FLOW': 'y'})
    
    # Fit the model to our data
    model.fit(df)

    # Create a DataFrame with future dates for making predictions
    future_dates = model.make_future_dataframe(periods=99, freq='1d', include_history=True)

    # Make predictions
    forecast = model.predict(future_dates)
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast = forecast.rename(columns={'ds': 'TS', 'yhat': 'FORECAST', 'yhat_lower': 'LOWER_BOUND', 'yhat_upper': 'UPPER_BOUND'})
    
    # Convert the Pandas DataFrame of predictions to a Snowpark DataFrame
    forecast_snowpark = session.createDataFrame(forecast)
    
    # Save predictions to a table in Snowflake
    forecast_snowpark.write.mode("overwrite").save_as_table("TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_PROPHET")
    
    return forecast_snowpark
```
![image](https://github.com/rasieira98/medium-water-leaking/assets/116558787/f76a845e-9263-41d7-a75f-32652bdc68b6)

## Handling Missing Data (GAPs)
To address the issue of missing data, we create a view that introduces significant gaps in the data and repeat the forecasting process:
### Snowflake Cortex ML Functions 
```sql
-- Select and group data by hour, summing the water flow
SELECT DATE_TRUNC('DAY', TS_MEASUREMENT), SUM(FLOW) AS FLOW 
FROM TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS
GROUP BY DATE_TRUNC('DAY', TS_MEASUREMENT);

-- Create or replace a view that groups water consumption per hour
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

-- Count total records and calculate training and testing sets
SELECT COUNT(*) AS TOTAL, 
       ROUND(COUNT(*) * 0.8) AS TRAIN, 
       ROUND(COUNT(*) * 0.2) AS TEST 
FROM WATER_CONS_DAILY_BIG_GAPS;

-- Create or replace a view with the first 397 records for the training set
CREATE OR REPLACE VIEW WATER_CONS_TRAIN_BIG_GAPS AS 
SELECT * 
FROM WATER_CONS_DAILY_BIG_GAPS
ORDER BY TS_MEASUREMENT
LIMIT 397;

-- Create a forecasting model using hourly water consumption training data
CREATE OR REPLACE SNOWFLAKE.ML.FORECAST model1(
  INPUT_DATA => SYSTEM$REFERENCE('VIEW', 'WATER_CONS_TRAIN_BIG_GAPS'),
  TIMESTAMP_COLNAME => 'TS_MEASUREMENT',
  TARGET_COLNAME => 'FLOW'
);

BEGIN
    -- Execute the forecasting model for 99 time periods
    CALL model1!FORECAST(FORECASTING_PERIODS => 99);
    LET x := SQLID;
    
    -- Create or replace a table with the forecast results
    CREATE OR REPLACE TABLE TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_FORECAST_BIG_GAPS AS 
    SELECT * 
    FROM TABLE(RESULT_SCAN(:x));
END;

-- Select all records from the forecast table
SELECT * 
FROM TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_FORECAST_BIG_GAPS;
```
![image](https://github.com/rasieira98/medium-water-leaking/assets/116558787/1f6174ff-62e0-4173-8c1c-ad5976bc4424)

### Implementation with Prophet and Python
```python
import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
from prophet import Prophet
import pandas as pd

def main(session: snowpark.Session): 
    # Table name in Snowflake
    source_table_name = 'TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_TRAIN'
    
    # Fetch data from the table in Snowflake
    dataframe = session.table(source_table_name)

    # Convert Snowflake DataFrame to Pandas
    df = dataframe.toPandas()
    df = df.sort_values(by='TS_MEASUREMENT')
    
    # Create a Prophet model
    model = Prophet()

    # Rename columns to 'ds' (TS_MEASUREMENT) and 'y' (FLOW) as per the data
    df = df.rename(columns={'TS_MEASUREMENT': 'ds', 'FLOW': 'y'})
    
    # Fit the model to our data
    model.fit(df)

    # Create a DataFrame with future dates for making predictions
    future_dates = model.make_future_dataframe(periods=99, freq='1d', include_history=True)

    # Make predictions
    forecast = model.predict(future_dates)
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast = forecast.rename(columns={'ds': 'TS', 'yhat': 'FORECAST', 'yhat_lower': 'LOWER_BOUND', 'yhat_upper': 'UPPER_BOUND'})
    
    # Convert the Pandas DataFrame of predictions to a Snowpark DataFrame
    forecast_snowpark = session.createDataFrame(forecast)

    # Save the predictions to a table in Snowflake
    forecast_snowpark.write.mode("overwrite").save_as_table("TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_PROPHET")
    
    return forecast_snowpark
```
![image](https://github.com/rasieira98/medium-water-leaking/assets/116558787/ea5303be-052d-4953-8e2e-a77c1fb0c85e)

## Generating Leak Alerts with Prophet Forecast
```python
import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
import pandas as pd

def main(session: snowpark.Session): 
    # Table names in Snowflake
    real_table = 'TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_DAILY_BIG_GAPS'
    predictions_table = 'TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_PROPHET_BIG_GAPS'
    
    # Fetch data from Snowflake tables
    real = session.table(real_table)
    predictions = session.table(predictions_table)

    # Convert Snowflake DataFrames to Pandas
    real = real.toPandas()
    real = real.sort_values(by='TS_MEASUREMENT')

    predictions = predictions.toPandas()
    predictions = predictions.rename(columns={'TS': 'TS_MEASUREMENT'})
    predictions = predictions.sort_values(by='TS_MEASUREMENT')
    
    # Merge DataFrames on 'TS_MEASUREMENT'
    df = pd.merge(real, predictions, on='TS_MEASUREMENT')
    
    # Add alert column
    df['ALERT'] = df['FLOW'] > 1.1 * df['FORECAST']
    df = df[['TS_MEASUREMENT', 'ALERT']]
    
    # Convert the Pandas DataFrame to a Snowpark DataFrame
    df = session.create_dataframe(df)
    
    # Save the data to a table in Snowflake (uncomment if needed)
    df.write.mode("overwrite").save_as_table("TIME_SERIES_SDG_EXAMPLES.PUBLIC.WATER_CONS_PROPHET_EXP")
    
    return df
```
![image](https://github.com/rasieira98/medium-water-leaking/assets/116558787/40c03e82-a334-4b17-baae-084886104bcb)


