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

        # Read the CSV file into a Spark DataFrame with inferred schema
        df_spark = (spark.read.format("csv")
                         .option("header", True)
                         .option("inferSchema", True)
                         .load(f"file://{file_path}"))

        col_names = [col_name.replace(" ", "_").replace(",", "").replace(";", "").replace("{", "").replace("}", "").replace("(", "").replace(")", "").replace("\n", "").replace("\t", "").replace("=", "") for col_name in df_spark.columns]
        df_spark = df_spark.toDF(*col_names)

        # Create or replace the table in Databricks
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")
        df_spark.write.mode("overwrite").saveAsTable(table_name)
        print(f"Table '{table_name}' created successfully from file '{file_path}'")
        # dbutils.fs.rm(dbfs_path)

# COMMAND ----------

# Iterate through the file paths and create tables
for key, path in file_paths.items():
    if key in table_names:
        table_name = table_names[key]
        create_table_from_csv(key, path, table_name)

# COMMAND ----------


