# Databricks notebook source
import yaml
import os
import pandas as pd
from pyspark.sql.types import *

# COMMAND ----------

# Path to the YAML file
yaml_file_path = "./Archive/config.yaml"

# COMMAND ----------

# Read the YAML file
with open(yaml_file_path, 'r') as file:
    yaml_content = yaml.safe_load(file)

# COMMAND ----------

# Extract the file_paths section
file_paths = yaml_content.get('file_paths', {})
table_names = yaml_content.get('table_names', {})

# Display the file paths and table names dictionaries
print("File Paths:")
print(file_paths)
print("\nTable Names:")
print(table_names)

# COMMAND ----------

# Function to create a table in Databricks for each file
def create_table_from_csv(file_key, file_path, table_name):
    if(file_path!=None):
        # Read the CSV file into a DataFrame
        print(os.path.abspath(file_path))
        file_path = os.path.abspath(file_path)
        file_name = os.path.basename(file_path)
        dbfs_path = f'/FileStore/shared_uploads/{file_name}'
        dbutils.fs.cp("file://" + file_path, dbfs_path)

        df_pd = pd.read_csv(file_path)

        # Map pandas data types to Spark SQL types
        dtype_mapping = {
            'int64': IntegerType(),
            'float64': DoubleType(),
            'object': StringType(),
            'datetime64': TimestampType()
            # Add more mappings as needed for your specific data types
        }

        # Convert pandas data types to Spark SQL types and create StructField objects
        fields = []
        for col_name, col_type in df_pd.dtypes.items():
            spark_type = dtype_mapping.get(str(col_type))
            if spark_type:
                fields.append(StructField(col_name, spark_type, True))
        
        # Create schema from StructType with inferred fields
        schema = StructType(fields)
        

        # Read the CSV file into a Spark DataFrame with inferred schema
        df_spark = spark.read.option("header", "true").csv(dbfs_path, schema=schema)

        # Create a table name based on the file key
        for col_name in df_spark.columns:
            new_col_name = col_name.replace(" ", "_").replace(",", "").replace(";", "").replace("{", "").replace("}", "").replace("(", "").replace(")", "").replace("\n", "").replace("\t", "").replace("=", "")
            df_spark = df_spark.withColumnRenamed(col_name, new_col_name)
        # Create or replace the table in Databricks
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")
        df_spark.write.mode("overwrite").saveAsTable(table_name)
        print(f"Table '{table_name}' created successfully from file '{file_path}'")
        dbutils.fs.rm(dbfs_path)

# COMMAND ----------

# Iterate through the file paths and create tables
for key, path in file_paths.items():
    if key in table_names:
        table_name = table_names[key]
        create_table_from_csv(key, path, table_name)

# COMMAND ----------


