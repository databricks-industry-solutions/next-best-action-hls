# Databricks notebook source
# MAGIC %md
# MAGIC #This Notebook explains the Model Workflow of the NBA Prediction system

# COMMAND ----------

# MAGIC %md
# MAGIC ### Importing the Necessary libraries

# COMMAND ----------

# Thoughts: 
# 1) Move some of the constants to the config
# 2) Make this more like py files (library types)
# 3) Initial configuration (catalog - NBA calatog - have tables in it, define schema)
# 4) Read directly from volume - RAW data, stagning database, results

# For data manipulation and analysis
import pandas as pd
import numpy as np

# For handling dates and times
from datetime import datetime, timedelta

# For operating system dependent functionality
import os

# For system-specific parameters and functions
import sys

# For time-related functions
import time

# To handle warnings
import warnings

# For regular expressions
import re as re

# For mathematical functions
import math

# Importing again (note: this is redundant since they are already imported above)
import pandas as pd
import numpy as np

# For splitting data into training and testing sets
from sklearn.model_selection import train_test_split

# Importing the Gekko optimization suite
import gekko
from gekko import GEKKO

# For timing the execution of small code snippets
import time

# For creating cartesian products of input iterables
from itertools import product

# For serialization and deserialization of Python objects
import joblib

# For reading and writing YAML configuration files
import yaml

# For creating interactive plots
import plotly.express as px

# For creating static plots
import matplotlib.pyplot as plt

# For creating custom colormaps in matplotlib
from matplotlib.colors import LinearSegmentedColormap

# For logging purposes
import logging

# For processing and evaluating literal Python expressions
import ast


# COMMAND ----------

# MAGIC %md
# MAGIC #### Setting up the logging configuration for the script. It specifies that error messages will be logged to a file named lnba_validation_log.txt. The log entries will include the timestamp, log level, and the actual log message.

# COMMAND ----------

# Configure the logging settings
# - Log messages will be written to 'lnba_validation_log.txt'
# - Only messages with a severity level of ERROR or higher will be logged
# - Each log entry will include the timestamp, log level, and the log message
logging.basicConfig(filename='./Archive/lnba_validation_log.txt', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# COMMAND ----------

# MAGIC %md
# MAGIC #### This dictionary, 'columns_dtypes', defines the expected data types for the columns in various data sources used within the script. Each key in the dictionary represents a different data source, and its value is another dictionary mapping column names to their respective data types. This ensures that when the data is read into pandas DataFrames, the columns have the correct data types, facilitating consistent data processing and analysis.

# COMMAND ----------

# Define the expected data types for the columns in various data sources.
# Each key represents a different data source, with its value being a dictionary
# that maps column names to their respective data types.

columns_dtypes = {
    "customer_data": {
        "HCP_ID": str,             # HCP ID as a string
        "CHANNEL": str,            # Channel name as a string
        "AFFINITY_SCORE": float    # Affinity score as a float
    },
    "mmix_output": {
        "CHANNEL": str,                 # Channel name as a string
        "MMIX_SEGMENT_NAME": str,       # MMIX segment name as a string
        "MMIX_SEGMENT_VALUE": str,      # MMIX segment value as a string
        "PARAMETER1": float,            # Parameter 1 as a float
        "ADJ_FACTOR": float,            # Adjustment factor as a float
        "RESPONSE_COEFFICIENT": float,  # Response coefficient as a float
        "OPTIMIZED_FREQUENCY": float,   # Optimized frequency as a float
        "TRANSFORMATION": str,          # Transformation type as a string
        "GRANULARITY": str              # Granularity as a string
    },
    "asset_availability": {
        "CHANNEL": str,             # Channel name as a string
        "TACTIC_ID": str,           # Tactic ID as a string
        "ASSET_FREQUENCY": float,   # Asset frequency as a float
        "GRANULARITY": str,         # Granularity as a string
        "SEGMENT_NAME": str,        # Segment name as a string
        "SEGMENT_VALUE": str        # Segment value as a string
    },
    "vendor_contract": {
        "CHANNEL": str,         # Channel name as a string
        "MAX_VOLUME": float,    # Maximum volume as a float
        "GRANULARITY": str      # Granularity as a string
    },
    "min_gap": {
        "CHANNEL_1": str,           # First channel name as a string
        "CHANNEL_2": str,           # Second channel name as a string
        "MIN_GAP": float,           # Minimum gap as a float
        "GRANULARITY": str,         # Granularity as a string
        "SEGMENT_NAME": str,        # Segment name as a string
        "SEGMENT_VALUE": str        # Segment value as a string
    },
    "extra_constraint_file": {
        "CHANNEL": str,         # Channel name as a string
        "SEGMENT_NAME": str,    # Segment name as a string
        "SEGMENT_VALUE": str,   # Segment value as a string
        "MIN_TPS": float,       # Minimum TPS as a float
        "MAX_TPS": float,       # Maximum TPS as a float
        "GRANULARITY": str      # Granularity as a string
    },
    "engagement_goal": {
        "CHANNEL": str,             # Channel name as a string
        "TYPE": str,                # Type as a string
        "TARGET": float,            # Target as a float
        "ENGAGEMENT_RATE": float,   # Engagement rate as a float
        "GRANULARITY": str,         # Granularity as a string
        "SEGMENT_NAME": str,        # Segment name as a string
        "SEGMENT_VALUE": str        # Segment value as a string
    },
    "constraint_segment": {
        "CHANNEL": str,                 # Channel name as a string
        "CONSTRAINT_SEGMENT_NAME": str  # Constraint segment name as a string
    },
    "priority": {
        "HCP_ID": str,      # HCP ID as a string
        "CHANNEL": str,     # Channel name as a string
        "PRIORITY": str     # Priority as a string
    },
    "historical_data": {
        "HCP_ID": str,      # HCP ID as a string
        "CHANNEL_ID": str,  # Channel ID as a string
        "EXPOSED_ON": str   # Date exposed on as a string
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC #### The 'granularity_mapping' dictionary maps different time granularities to their respective multipliers. These multipliers represent the number of periods that fit into a year for each granularity type. This mapping is useful for converting and normalizing time-based data.

# COMMAND ----------

# Define a mapping of time granularities to their respective multipliers.
# These multipliers represent the number of periods in a year for each granularity type.
# This mapping is useful for converting and normalizing time-based data.

# Aditya CMT - Put this in config

granularity_mapping = {
    'WEEKLY': 13,       # 13 weeks in a quarter (approximately 52 weeks in a year divided by 4)
    'DAILY': 91,        # 91 days in a quarter (approximately 365 days in a year divided by 4)
    'MONTHLY': 3,       # 3 months in a quarter (12 months in a year divided by 4)
    'QUARTERLY': 1,     # 1 quarter in a quarter
    'SEMESTERLY': 0.5,  # 0.5 semesters in a quarter (2 semesters in a year divided by 4)
    'YEARLY': 0.25      # 0.25 years in a quarter (1 year divided by 4)
}

# COMMAND ----------

# MAGIC %md
# MAGIC #### The 'validate_files' function is designed to validate a set of file paths to ensure that they point to valid CSV files, which can be read successfully and match the expected format. The function takes a dictionary where keys represent file categories (such as 'train', 'test', etc.) and values are the file paths.

# COMMAND ----------

def validate_files(file_paths):
    """
    Validate the provided file paths to ensure they are valid CSV files that can be read and match the expected format.
    
    Parameters:
    file_paths (dict): A dictionary where keys are file categories (e.g., 'train', 'test') and values are the file paths.
    
    Returns:
    bool: True if all files are valid, False otherwise.
    
    This function performs the following checks for each file:
        1. Checks if the file path is provided.
        2. Checks if the file path exists.
        3. Checks if the file is a CSV file.
        4. Checks if the file is readable.
        5. Checks if the file contains the expected columns and data types, and if there are any null values.
    """
    
    all_valid = True  # Track if all files are valid
    
    # Iterate over each file category and its corresponding file path
    for file_category, file_path in file_paths.items():
        
        # Check if the file path is provided
        if not file_path:
            logging.warning(f"File path not provided for {file_category}. Please check if it's required.")
            print(f"File path not provided for {file_category}. Please check if it's required.")
            all_valid = False
            continue

        # Check if the file path exists
        # if not os.path.isfile(file_path):
        #     logging.warning(f"File path for {file_category} does not exist: {file_path}")
        #     print(f"File path for {file_category} does not exist: {file_path}")
        #     all_valid = False
        #     continue

        # Check if the file path is a CSV file
        if not file_path.lower().endswith('.csv'):
            logging.warning(f"File path for {file_category} is not a CSV file: {file_path}")
            print(f"File path for {file_category} is not a CSV file: {file_path}")
            all_valid = False
            continue

        # Check if the file is readable
        try:
            # df = pd.read_csv(file_path)
            df = spark.table(file_path).toPandas()
        except Exception as e:
            logging.warning(f"Unable to read file {file_path} for {file_category}: {e}")
            print(f"Unable to read file {file_path} for {file_category}. Please check the path and file format.")
            all_valid = False
            continue

        # Check if the expected columns and data types match and if there are any null values
        expected_columns_dtypes_category = columns_dtypes.get(file_category, {})
        valid, errors = check_file(file_category, file_path, expected_columns_dtypes_category)
        if not valid:
            all_valid = False
            print(f"Error in {file_category}:")
            for error_message in errors:
                print(f"  - {error_message}")

    # Print the final validation result
    if all_valid:
        print("All files are in the expected format.")
    else:
        print("Validation failed for one or more files. Please check the errors.")

    # TODO: Update the function to return the actual validity flag instead of always returning True
    return True

# COMMAND ----------

# MAGIC %md
# MAGIC #### The 'check_file' function validates a CSV file by ensuring it contains the expected columns, that these columns have the correct data types, and that there are no NULL values present in any of the columns. It is part of a larger process to ensure data integrity before further processing or analysis.

# COMMAND ----------

def check_file(file_category, file_path, expected_columns_dtypes):
    """
    Validate a CSV file by checking for the presence of expected columns, correct data types, 
    and absence of NULL values.

    Parameters:
    file_category (str): Category of the file (e.g., 'train', 'test').
    file_path (str): Path to the CSV file.
    expected_columns_dtypes (dict): Dictionary where keys are expected column names and values are expected data types.

    Returns:
    tuple: A tuple containing:
        - bool: True if no mismatches are found, False otherwise.
        - list: A list of mismatch error messages.
    
    This function performs the following checks on the CSV file:
        1. Ensures all expected columns are present.
        2. Verifies that the data types of columns match the expected types.
        3. Checks for the presence of NULL values in any column.
    """
    
    mismatches = []  # Collect mismatches
    
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Normalize column names by making them uppercase and replacing spaces with underscores
        df.columns = df.columns.str.upper()
        df.columns = df.columns.str.replace(' ', '_')
        
        # Check for missing columns and data type mismatches
        for column, expected_type in expected_columns_dtypes.items():
            if column not in df.columns:
                # Log and record a warning if an expected column is not found
                logging.warning(f"Mismatches in file {file_category}: Expected column '{column}' not found")
                mismatches.append(f"File mismatches {file_category}: Expected column '{column}' not found")
            else:
                # Check if the actual data type matches the expected data type
                actual_type = df[column].dtype
                if actual_type != expected_type:
                    try:
                        # Try converting the column to the expected data type
                        df[column] = df[column].astype(expected_type)
                    except Exception as conversion_error:
                        # Log and record an error if data type conversion fails
                        logging.error(f"Mismatches in file {file_category}: Data type mismatch in column '{column}'. "\
                                      f"Error during conversion: {conversion_error}")
                        mismatches.append(f"File mismatches {file_category}: Data type mismatch in column '{column}'. "
                                          f"Error during conversion: {conversion_error}")

        # Check for presence of NULL values in any column
        null_columns = df.columns[df.isnull().any()].tolist()
        if null_columns:
            logging.warning(f"Mismatches in file {file_category}: Presence of NULL values in columns - {null_columns}")
            mismatches.append(f"File mismatches {file_category}: Presence of NULL values in columns - {null_columns}")

        # Return True if there are no mismatches, along with the list of mismatches
        return not mismatches, mismatches
    
    except Exception as e:
        # Log and return an error if there is an issue reading the file or other exceptions
        logging.error(f"Error due to file {file_path}: {str(e)}", exc_info=True)
        return False, [f"Error in file: {e}"]

# COMMAND ----------

# MAGIC %md
# MAGIC #### The 'calculate_optimized_trx' function computes an optimized transaction value based on given parameters and a specified transformation type. This function is designed to work with data in a pandas DataFrame, taking a row of data and extracting necessary values to perform the calculation. The calculation varies depending on the transformation type, which can be logarithmic, power, linear, or negative exponential.

# COMMAND ----------

# Maybe in the confirg. Have some insight into why someone should be using a particular transformation

def calculate_optimized_trx(row, tps, beta):
    """
    Calculate the optimized transaction value based on the given parameters and transformation type.

    Parameters:
    row (pd.Series): A pandas Series representing a row from a DataFrame, containing necessary parameters for calculation.
    tps (str): The column name in the row representing the number of transactions per second (tps).
    beta (str): The column name in the row representing the beta or response coefficient.

    Returns:
    float: The calculated optimized transaction value.

    This function performs the calculation based on the type of transformation specified in the 'TRANSFORMATION' column.
    Supported transformations include:
        - LOG: Logarithmic transformation
        - POWER: Power transformation
        - LINEAR: Linear transformation
        - NEGEX: Negative Exponential transformation

    Raises:
    ValueError: If an unsupported transformation type is specified.
    """
    
    # Extract relevant values from the row
    freq = row[tps]  # Number of transactions per second (tps)
    k = row["PARAMETER1"]
    a = row["ADJ_FACTOR"]
    coef = row[beta]  # Beta/response coefficient
    transformation = row['TRANSFORMATION']
    
    # Calculate the optimized transaction value based on the transformation type
    if transformation == "LOG":
        # Logarithmic transformation
        return np.log(1 + freq * k) * a * coef
    elif transformation == "POWER":
        # Power transformation
        return (freq ** k) * a * coef
    elif transformation == "LINEAR":
        # Linear transformation
        return freq * coef
    elif transformation == "NEGEX":
        # Negative Exponential transformation
        return (1 - np.exp(-freq * k)) * a * coef
    else:
        # Unsupported transformation type
        error_msg = f"Unsupported transformation: {transformation}. Supported transformations are LOG, POWER, LINEAR, and NEGEX."
        logging.error(error_msg)
        raise ValueError(error_msg)

# COMMAND ----------

# MAGIC %md
# MAGIC #### The 'process_files' function reads, processes, and merges multiple CSV files based on a given configuration to create a final DataFrame for analysis. It handles various data files including customer data, constraint segments, MMIX output, asset availability, vendor contracts, and more. The function includes data cleaning, transformation, and metric calculation to prepare the data for analysis.

# COMMAND ----------

def process_files(config, allow_processing_with_errors=False):
    """
    Process and merge various data files to create a final dataframe for analysis.

    This function reads, processes, and merges multiple CSV files as specified in the configuration.
    The files include customer data, constraint segments, and other datasets such as MMIX output,
    asset availability, vendor contracts, and more. The processing includes cleaning, transforming,
    and calculating various metrics for the final analysis.

    Parameters:
    config (dict): A dictionary containing file paths and configurations for processing.
    allow_processing_with_errors (bool): If True, allows processing to continue despite errors. Default is False.

    Returns:
    pd.DataFrame: The final processed dataframe ready for analysis.
    """
    print("Processing files...")
    
    # Load file paths and expected column data types from the configuration
    customer_data_path = config_file_paths.get('customer_data')
    constraint_segment_path = config_file_paths.get('constraint_segment')
    
    if customer_data_path:
        expected_columns_dtypes_cd = columns_dtypes.get('customer_data', {})
        expected_columns_dtypes_cons = columns_dtypes.get('constraint_segment', {})

        # Read and preprocess customer data
        # customer_data = pd.read_csv(customer_data_path)
        customer_data = spark.table(customer_data_path).toPandas()
        customer_data.columns = customer_data.columns.str.replace(' ', '_').str.upper()
        customer_data = customer_data.apply(lambda x: x if x.dtype != 'O' else x.str.upper())
        customer_data = customer_data.apply(lambda x: x.replace(' ', '_') if x.dtype == 'O' else x)
        customer_data = customer_data.drop_duplicates()
        customer_data['HCP_ID'] = customer_data['HCP_ID'].astype(str)
        customer_data["AFFINITY_SCORE"] = customer_data['AFFINITY_SCORE'].astype('float64')
       
        # Read and preprocess constraint segment data
        # constraint_segment = pd.read_csv(constraint_segment_path)
        constraint_segment = spark.table(constraint_segment_path).toPandas()
        constraint_segment.columns = constraint_segment.columns.str.replace(' ', '_').str.upper()
        constraint_segment = constraint_segment.drop_duplicates()
        constraint_segment = constraint_segment.apply(lambda x: x if x.dtype != 'O' else x.str.upper())
        constraint_segment = constraint_segment.apply(lambda x: x.replace(' ', '_') if x.dtype == 'O' else x)
        
        # Warn if multiple segments exist for a channel
        for channel, group in constraint_segment.groupby('CHANNEL'):
            if len(group['CONSTRAINT_SEGMENT_NAME'].unique()) > 1:
                logging.warning(f"Multiple segments found for channel '{channel}': {', '.join(group['CONSTRAINT_SEGMENT_NAME'].unique())}")
                print(f"Warning: Multiple segments found for channel '{channel}': {', '.join(group['CONSTRAINT_SEGMENT_NAME'].unique())}")

        # Merge customer data with constraint segment data
        constraint_segment = constraint_segment.drop_duplicates(subset=['CHANNEL'])
        customer_data = customer_data.merge(constraint_segment, how='left', on='CHANNEL')
        customer_data['CONSTRAINT_SEGMENT_VALUE'] = customer_data.apply(lambda row: row[row['CONSTRAINT_SEGMENT_NAME']], axis=1)

        # Calculate total HCPs by channel and segment
        total_hcps = customer_data.groupby(['CHANNEL', 'CONSTRAINT_SEGMENT_NAME', 'CONSTRAINT_SEGMENT_VALUE'])['HCP_ID'].nunique().reset_index().rename(columns={'HCP_ID': 'TOTAL_HCPS_SEGMENT'})
        total_hcps['TOTAL_HCPS_CHANNEL'] = total_hcps.groupby('CHANNEL')['TOTAL_HCPS_SEGMENT'].transform('sum')
        constraint_hcp_df = total_hcps.copy()

        # Define list of file categories to process
        file_list = ['mmix_output', 'asset_availability', 'vendor_contract', 'min_gap', 'engagement_goal', 'extra_constraint']
        merge_key_constraint = ['CHANNEL', 'CONSTRAINT_SEGMENT_NAME', 'CONSTRAINT_SEGMENT_VALUE']
        merge_key_datasets = ['CHANNEL', 'SEGMENT_NAME', 'SEGMENT_VALUE']

        # Process each file category
        for file_category in file_list:
            expected_columns_dtypes = columns_dtypes.get(file_category, {})
            file_path = config_file_paths.get(file_category, '')
            try:
                if file_category.lower() != 'customer_data':
                    print(f'Processing {file_category}')
                    # df = pd.read_csv(file_path)
                    df = spark.table(file_path).toPandas()
                    
                    df.columns = df.columns.str.replace(' ', '_').str.upper()
                    df = df.drop_duplicates()
                    df = df.apply(lambda x: x if x.dtype != 'O' else x.str.upper())
                    df = df.apply(lambda x: x.replace(' ', '_') if x.dtype == 'O' else x)
                    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                    
                    if file_category == 'mmix_output':
                        print(f'Creating {file_category} df')
                        mmix_output = df.drop_duplicates(subset=['CHANNEL', 'MMIX_SEGMENT_NAME', 'MMIX_SEGMENT_VALUE'])
                        mmix_output['OPTIMIZED_FREQUENCY'] = mmix_output.apply(lambda row: row['OPTIMIZED_FREQUENCY'] * granularity_mapping[row['GRANULARITY']], axis=1)
                        print(f'Created {file_category} df')
                    
                    elif file_category == 'asset_availability': 
                        print(f'Creating {file_category} df')
                        asset_availability = df.drop_duplicates()
                        asset_availability['ASSET_FREQUENCY'] = asset_availability.apply(lambda row: row['ASSET_FREQUENCY'] * granularity_mapping[row['GRANULARITY']], axis=1)
                        asset_tp = asset_availability.groupby(['CHANNEL', 'SEGMENT_NAME', 'SEGMENT_VALUE'])['ASSET_FREQUENCY'].sum().reset_index(name='ASSET_TP')

                        asset_tp = asset_tp.merge(total_hcps, how='left', left_on=['CHANNEL', 'SEGMENT_NAME', 'SEGMENT_VALUE'], right_on=['CHANNEL', 'CONSTRAINT_SEGMENT_NAME', 'CONSTRAINT_SEGMENT_VALUE'])
                        asset_tp['MAX_TPS_SEGMENT'] = asset_tp['ASSET_TP'] * asset_tp['TOTAL_HCPS_SEGMENT']
                        asset_tp['MAX_TPS_CHANNEL'] = asset_tp.groupby('CHANNEL')['MAX_TPS_SEGMENT'].transform(sum)
                        
                        print(f'Merging {file_category} to constraint_df')
                        constraint_hcp_df = constraint_hcp_df.merge(asset_tp[['CHANNEL', 'MAX_TPS_CHANNEL']].drop_duplicates(), on='CHANNEL')
                        constraint_hcp_df = constraint_hcp_df.merge(asset_tp[['MAX_TPS_SEGMENT'] + merge_key_datasets].drop_duplicates(), left_on=merge_key_constraint, right_on=merge_key_datasets)
                        print(f'Created {file_category} df')

                    elif file_category == 'vendor_contract':
                        print(f'Creating {file_category} df')
                        vendor_contract = df 
                        vendor_contract['MAX_TPS_CHANNEL'] = vendor_contract.apply(lambda row: row['MAX_VOLUME'] * granularity_mapping[row['GRANULARITY']], axis=1)
                        vendor_contract = vendor_contract.merge(total_hcps[['CHANNEL', 'TOTAL_HCPS_CHANNEL']].drop_duplicates(), how='left', on='CHANNEL')
                        vendor_contract = vendor_contract.dropna(subset=['TOTAL_HCPS_CHANNEL'])
                        vendor_contract = vendor_contract[['CHANNEL', 'MAX_TPS_CHANNEL']]
                        print(f'Merging {file_category} to constraint_df')
                        constraint_hcp_df = constraint_hcp_df.merge(vendor_contract[['CHANNEL', 'MAX_TPS_CHANNEL']].drop_duplicates(), on='CHANNEL')
                        print(f'Created {file_category} df')

                    elif file_category == 'min_gap':
                        print(f'Creating {file_category} df')
                        min_gap = df[df['CHANNEL_1'] == df['CHANNEL_2']]
                        min_gap = min_gap[min_gap['MIN_GAP'] != 0]
                        min_gap['CHANNEL'] = min_gap['CHANNEL_1']
                        min_gap['MAX_TPS_SEGMENT'] = min_gap.apply(lambda row: (1 / row['MIN_GAP']) * granularity_mapping[row['GRANULARITY']], axis=1)
                        min_gap['MAX_TPS_CHANNEL'] = min_gap.groupby('CHANNEL')['MAX_TPS_SEGMENT'].transform(sum)
                        min_gap = min_gap.merge(total_hcps, how='left', left_on=['CHANNEL', 'SEGMENT_NAME', 'SEGMENT_VALUE'], right_on=['CHANNEL', 'CONSTRAINT_SEGMENT_NAME', 'CONSTRAINT_SEGMENT_VALUE'])
                        min_gap['MAX_TPS_SEGMENT'] = min_gap['MAX_TPS_SEGMENT'] * min_gap['TOTAL_HCPS_SEGMENT']
                        min_gap['MAX_TPS_CHANNEL'] = min_gap.groupby('CHANNEL')['MAX_TPS_SEGMENT'].transform(sum)
                        min_gap = min_gap[['CHANNEL', 'SEGMENT_NAME', 'SEGMENT_VALUE', 'MAX_TPS_SEGMENT', 'MAX_TPS_CHANNEL']]
                        
                        print(f'Merging {file_category} to constraint_df')
                        constraint_hcp_df = constraint_hcp_df.merge(min_gap[['CHANNEL', 'MAX_TPS_CHANNEL']].drop_duplicates(), on='CHANNEL')
                        constraint_hcp_df = constraint_hcp_df.merge(min_gap[['MAX_TPS_SEGMENT'] + merge_key_datasets].drop_duplicates(), left_on=merge_key_constraint, right_on=merge_key_datasets)
                        print(f'Created {file_category} df')

                    elif file_category == 'engagement_goal':
                        print(f'Creating {file_category} df')
                        engagement_goal = df
                        engagement_goal['TARGET'] = engagement_goal.apply(lambda row: row['TARGET'] * granularity_mapping[row['GRANULARITY']], axis=1)
                        engagement_goal = engagement_goal.drop_duplicates(subset=['CHANNEL', 'SEGMENT_NAME', 'SEGMENT_VALUE'])
                        print(f'Merging {file_category} to constraint_df')
                        constraint_hcp_df = constraint_hcp_df.merge(engagement_goal[['CHANNEL', 'SEGMENT_NAME', 'SEGMENT_VALUE', 'TARGET']].drop_duplicates(), left_on=merge_key_constraint, right_on=merge_key_datasets)
                        print(f'Created {file_category} df')

                    elif file_category == 'extra_constraint':
                        print(f'Creating {file_category} df')
                        extra_constraint = df
                        extra_constraint['EXTRA_CONSTRAINT'] = extra_constraint.apply(lambda row: row['EXTRA_CONSTRAINT'] * granularity_mapping[row['GRANULARITY']], axis=1)
                        extra_constraint = extra_constraint.drop_duplicates(subset=['CHANNEL', 'SEGMENT_NAME', 'SEGMENT_VALUE'])
                        print(f'Merging {file_category} to constraint_df')
                        constraint_hcp_df = constraint_hcp_df.merge(extra_constraint[['CHANNEL', 'SEGMENT_NAME', 'SEGMENT_VALUE', 'EXTRA_CONSTRAINT']].drop_duplicates(), left_on=merge_key_constraint, right_on=merge_key_datasets)
                        print(f'Created {file_category} df')

            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                if not allow_processing_with_errors:
                    print(f"Processing aborted due to errors. Please check the logs for details.")
                    sys.exit(1)
        
        # Merging MMix Output with Master Data
        customer_data = customer_data.merge(mmix_output[['CHANNEL','MMIX_SEGMENT_NAME']].drop_duplicates(),how='left',on=['CHANNEL'])
        customer_data['MMIX_SEGMENT_VALUE']  = customer_data.apply(lambda row: row[row['MMIX_SEGMENT_NAME']], axis=1)
        
        customer_data  = customer_data.merge(mmix_output,how='left',on=['CHANNEL','MMIX_SEGMENT_NAME','MMIX_SEGMENT_VALUE'])
        
        # Creating Constraint df
        max_tps_segment_cols = constraint_hcp_df.filter(like='MAX_TPS_SEGMENT').columns.tolist()
        max_tps_channel_cols = constraint_hcp_df.filter(like='MAX_TPS_CHANNEL').columns.tolist()
        min_tps_segment_cols = constraint_hcp_df.filter(like='MIN_TPS_SEGMENT').columns.tolist()
        min_tps_channel_cols = constraint_hcp_df.filter(like='MIN_TPS_CHANNEL').columns.tolist()
        
        # Min of all max tps constraints
        constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_CHANNEL']  = constraint_hcp_df[max_tps_channel_cols].min(axis=1)
        constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SEGMENT']  = constraint_hcp_df[max_tps_segment_cols].min(axis=1)
        
        constraint_hcp_df['FINAL_MIN_TOUCHPOINTS_SEGMENT']  = constraint_hcp_df[min_tps_segment_cols].min(axis=1)
        constraint_hcp_df['FINAL_MIN_TOUCHPOINTS_SEGMENT']  = constraint_hcp_df['FINAL_MIN_TOUCHPOINTS_SEGMENT'].fillna(0.0)
        
        # Handle missing values in FINAL_MAX_TOUCHPOINTS  
        remaining_tps = constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_CHANNEL'] - constraint_hcp_df.groupby('CHANNEL')['FINAL_MAX_TOUCHPOINTS_SEGMENT'].transform('sum')

        nan_rows = constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SEGMENT'].isna()

        total_hcps_nan_channel = constraint_hcp_df.loc[nan_rows].groupby('CHANNEL')['TOTAL_HCPS_SEGMENT'].transform('sum')

        ratio_fill = constraint_hcp_df['TOTAL_HCPS_SEGMENT'] / total_hcps_nan_channel

        constraint_hcp_df.loc[nan_rows, 'FINAL_MAX_TOUCHPOINTS_SEGMENT'] = remaining_tps[nan_rows] * ratio_fill[nan_rows]
       
        constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SEGMENT'] = np.where(constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SEGMENT']<1,1,constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SEGMENT'])

       
    # Checking if channel level max tps are not exceeded and if they are breached decreasing segment level tps in the req ratio

        constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SUM'] = constraint_hcp_df.groupby('CHANNEL')['FINAL_MAX_TOUCHPOINTS_SEGMENT'].transform('sum')
       
        decrease_ratio  = constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_CHANNEL']/constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SUM']
        
        constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SEGMENT']  = np.where(constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SUM']>constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_CHANNEL'], np.floor(decrease_ratio*constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SEGMENT']),constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SEGMENT'])
       
       # Setting min equal to max if min tps exceed max tps
       
        violation_1 = constraint_hcp_df['FINAL_MIN_TOUCHPOINTS_SEGMENT'] > constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SEGMENT']
       
        for index, row in constraint_hcp_df[violation_1].iterrows():
                 print(f"For channel {row['CHANNEL']} - {row['CONSTRAINT_SEGMENT_NAME']} ({row['CONSTRAINT_SEGMENT_VALUE']}), max tp is less than min tp.")
                 logging.warning(f"For channel {row['CHANNEL']} - {row['CONSTRAINT_SEGMENT_NAME']} ({row['CONSTRAINT_SEGMENT_VALUE']}), max tp is less than min tp.")
       
        constraint_hcp_df['FINAL_MIN_TOUCHPOINTS_SEGMENT'] = np.where(constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SEGMENT'] < constraint_hcp_df['FINAL_MIN_TOUCHPOINTS_SEGMENT'], constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SEGMENT'], constraint_hcp_df['FINAL_MIN_TOUCHPOINTS_SEGMENT'])
        
        constraint_hcp_df['FINAL_MIN_TOUCHPOINTS_SEGMENT'] = np.ceil(constraint_hcp_df['FINAL_MIN_TOUCHPOINTS_SEGMENT']/constraint_hcp_df['TOTAL_HCPS_SEGMENT'])
        constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SEGMENT'] = np.floor(constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SEGMENT']/constraint_hcp_df['TOTAL_HCPS_SEGMENT'])
        
        constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SEGMENT'] = np.where(constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SEGMENT']<1,1,constraint_hcp_df['FINAL_MAX_TOUCHPOINTS_SEGMENT'])
        
        required_constraint_cols = ['CHANNEL','CONSTRAINT_SEGMENT_NAME','CONSTRAINT_SEGMENT_VALUE','FINAL_MIN_TOUCHPOINTS_SEGMENT','FINAL_MAX_TOUCHPOINTS_SEGMENT','TOTAL_HCPS_SEGMENT','TOTAL_HCPS_CHANNEL']

        final_df = customer_data.merge(constraint_hcp_df[required_constraint_cols].drop_duplicates(),how='left',on = ['CHANNEL','CONSTRAINT_SEGMENT_NAME','CONSTRAINT_SEGMENT_VALUE'])
        
        # Creating Affinty Segments for visualisations and classifications
        final_df["AFFINITY_SCORE"] = final_df["AFFINITY_SCORE"].round(2)

        bins = [0, 0.001, 0.2, 0.4, 0.6, 0.8, 1.01]
        labels = ['Z', 'VL', 'L', 'M', 'H','VH']

        final_df['AFFINITY_SEGMENT'] = pd.cut(final_df['AFFINITY_SCORE'], bins=bins, labels=labels, right=False, include_lowest=True)
        
        # Calculate OPTIMIZED_TRX using Optimized frequnecy and resp coeff from mmix

        final_df["OPTIMIZED_TRX"] = final_df.apply(lambda row: calculate_optimized_trx(row, tps='OPTIMIZED_FREQUENCY', beta='RESPONSE_COEFFICIENT'), axis=1)
    
    # Creating only for Check no other use 
        calc2 = final_df.groupby(["CHANNEL","CONSTRAINT_SEGMENT_NAME", "CONSTRAINT_SEGMENT_VALUE", "TRANSFORMATION", "PARAMETER1", "ADJ_FACTOR", "FINAL_MIN_TOUCHPOINTS_SEGMENT", "FINAL_MAX_TOUCHPOINTS_SEGMENT"]).agg({'OPTIMIZED_TRX':'sum', 'OPTIMIZED_FREQUENCY':'mean', 'HCP_ID':'count'}).reset_index()

    # Aggregating OPTIMZED TRX on Channel Constraint Segment Level
        calc_segment = final_df.groupby(["CHANNEL", "CONSTRAINT_SEGMENT_NAME","CONSTRAINT_SEGMENT_VALUE", "TRANSFORMATION", "PARAMETER1", "ADJ_FACTOR"]).agg({
                            'OPTIMIZED_TRX': 'sum',\
                            'OPTIMIZED_FREQUENCY': 'mean',\
                            'HCP_ID': 'count'}).reset_index()
    
    # back calculating the response coeff at Channel Constrain Segment level which is required level for Optimization
        calc_segment["RESPONSE COEFFICIENT"] = 0

        a = calc_segment['ADJ_FACTOR']
        k = calc_segment['PARAMETER1']

        calc_segment.loc[calc_segment['TRANSFORMATION'] == "LOG", ['RESPONSE COEFFICIENT']] = calc_segment["OPTIMIZED_TRX"]/(
               calc_segment["HCP_ID"] * np.log(1 + calc_segment["OPTIMIZED_FREQUENCY"]*k)*a)
        calc_segment.loc[calc_segment['TRANSFORMATION'] == "POWER", ['RESPONSE COEFFICIENT']] = calc_segment["OPTIMIZED_TRX"]/(
               calc_segment["HCP_ID"] * (calc_segment["OPTIMIZED_FREQUENCY"] ** k)*a)
        calc_segment.loc[calc_segment['TRANSFORMATION'] == "LINEAR", ['RESPONSE COEFFICIENT']] = calc_segment["OPTIMIZED_TRX"]/(
               calc_segment["HCP_ID"] * calc_segment["OPTIMIZED_FREQUENCY"])
        calc_segment.loc[calc_segment['TRANSFORMATION'] == "NEGEX", ['RESPONSE COEFFICIENT']] = calc_segment["OPTIMIZED_TRX"]/(
               calc_segment["HCP_ID"] * (1 - np.exp(-1 * calc_segment["OPTIMIZED_FREQUENCY"]*k))*a)

    # Getting Calculated Response Coeff at Constraint Segment Level
        df_final = pd.merge(final_df, calc_segment[["CHANNEL", "CONSTRAINT_SEGMENT_NAME","CONSTRAINT_SEGMENT_VALUE", "RESPONSE COEFFICIENT"]],
                        how='left', left_on=['CHANNEL', 'CONSTRAINT_SEGMENT_NAME','CONSTRAINT_SEGMENT_VALUE'], right_on=['CHANNEL', 'CONSTRAINT_SEGMENT_NAME','CONSTRAINT_SEGMENT_VALUE'], suffixes=['_OLD', ''])
    
        violation = calc2[calc2["FINAL_MIN_TOUCHPOINTS_SEGMENT"]>calc2["OPTIMIZED_FREQUENCY"]]
        violation = violation[["CHANNEL","CONSTRAINT_SEGMENT_NAME","CONSTRAINT_SEGMENT_VALUE","FINAL_MIN_TOUCHPOINTS_SEGMENT","FINAL_MAX_TOUCHPOINTS_SEGMENT","OPTIMIZED_FREQUENCY"]]
     
        if violation.shape[0]>0:
            for index, row in violation.iterrows():
                logging.error(f"Violation: Min Touchpoints for Channel '{row['CHANNEL']}' with Constraint Segment Value '{row['CONSTRAINT_SEGMENT_VALUE']}' are higher than the OPTIMIZED_FREQUENCY")
            sys.exit(1)

        df_final.drop(df_final.loc[:, df_final.columns.str.endswith("Old")],axis = 1,inplace = True)

        df_final.drop(columns = {'MMIX_SEGMENT_VALUE',"OPTIMIZED_TRX"},inplace = True)

        df_final = df_final.sort_values(["CHANNEL","CONSTRAINT_SEGMENT_VALUE","AFFINITY_SCORE"], ascending = [True,True,False]).reset_index(drop = True)
    
    #Creating new beta for calculating trx for optimization module to maximize
       
        df_final['NEW_BETA'] = (df_final['AFFINITY_SCORE'] * df_final['RESPONSE_COEFFICIENT'] /df_final.groupby(\
           ['CHANNEL', 'CONSTRAINT_SEGMENT_VALUE'])['AFFINITY_SCORE'].transform('mean')).replace({np.inf: 0, -np.inf: 0}).fillna(0) 
        df_final.to_csv(f'{output_folder}/df_final_before_model_run.csv',index=False)
    
    return df_final

# COMMAND ----------

# MAGIC %md
# MAGIC #### The 'affinity_segment_charts' function generates classification charts for different customer segments based on their affinity scores. It creates bar charts illustrating the distribution of customers across various segments and affinity levels for each channel.

# COMMAND ----------

def affinity_segment_charts(final_df):
    cps = ["VH", "H", "M", "L", "VL", "Z", "NR"]

    customer_segment = pd.DataFrame(columns=["Segment", "Channel", "#HCPs"])
    segment_column_list = final_df['CONSTRAINT_SEGMENT_NAME'].unique().tolist()
    channel_column = 'CHANNEL'
    affinity_column = 'AFFINITY_SEGMENT'
    
    for i in segment_column_list:
        for j in cps:
                reqdata = final_df[['HCP_ID', i, channel_column, 'AFFINITY_SCORE', affinity_column]]
                if (j in reqdata[affinity_column].unique()):
                    reqdata2 = reqdata[reqdata[affinity_column] == j]
                    la = pd.DataFrame(pd.crosstab(index=reqdata2[i], columns=reqdata2[channel_column],
                                                  values=reqdata2['HCP_ID'], aggfunc=pd.Series.nunique).reset_index())

                    lab = la.set_index(i)
                    labb = lab.stack().reset_index()
                    labb.rename(columns={0: '#HCPs'}, inplace=True)

                    colors = ['#ED7D31', '#EF995F', '#FFD68A', '#FFA500', '#FFB52E', '#FFC55C', '#FF6347', '#FF4500',
                              '#FF7F50']
                    abcd = f"Classification of customers for '{i}' and affinity segment '{j}'"

                    fig = px.bar(labb, x=i, y='#HCPs', color=labb[channel_column], barmode='stack',
                                 text=labb['#HCPs'].astype(str), color_discrete_sequence=colors)

                    ylabel = f"n size ({len(np.unique(reqdata2['HCP_ID'])):,} HCPs)"

                    fig.update_layout(title_text=abcd, title_x=0.5, title_font_color='#ED7D31')
                    fig.update_layout(xaxis_title=ylabel)
                    fig.update_layout(yaxis_title="Number of Customers")
                    fig.update_layout({
                        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                    })

                    fig.update_traces(texttemplate='%{text:,}')
                    fig.update_traces(textfont_size=12)
                    fig.update_traces(texttemplate='%{text:,}')
                    fig.update_yaxes(tickformat=",d")
                     
                    output_path = os.path.join(output_folder, 'Customer_Segmentation_Charts', f"Classification_of_customers_for_{i}_and_affinity_segment_{j}.png")
                    fig.write_image(file=output_path,engine="kaleido")

                    labb["Segment Name"] = i
                    labb["Affinity Segment"] = j
                    labb.rename(columns={i: "Segment"}, inplace=True)
                    customer_segment = pd.concat([customer_segment, labb]).reset_index(drop=True)

    return None

# COMMAND ----------

# config_path = "./Archive/config.yaml"
# output_path = output_path.replace('//', '/')
# output_path = os.path.join(*output_path.split('/'))
# output_folder = os.path.join(output_path, run_id)

# with open(config_path, 'r') as config_file:
#     # The yaml.safe_load function parses the YAML data into a Python dictionary named config_data
#     config_data = yaml.safe_load(config_file)
#     output = config_data.get('output', {})
#     run_id = output.get('run_id', '')
#     output_path = output.get('output_folder_path', '')
#     a=os.path.join(output_folder,"a")



# COMMAND ----------

# MAGIC %md
# MAGIC #### The 'load_config' function reads configuration data from a YAML file, parses it into a Python dictionary, and sets up the necessary file paths and directories. It extracts file paths, control parameters, run periods, and output configurations from the YAML file. The output folder path is constructed using the provided run ID, and directories for storing results are created accordingly.

# COMMAND ----------

def load_config(config_path):
    """
    Load configuration data from a YAML file and set up the necessary file paths and directories.

    This function reads a configuration file in YAML format from the specified path, parses it into
    a Python dictionary, and extracts various configuration parameters. It then constructs the output
    folder path using the provided run_id and creates necessary directories for storing results.

    Parameters:
    config_path (str): The path to the YAML configuration file.

    Returns:
    bool: Returns True if the configuration is successfully loaded and directories are created.
    
    Raises:
    SystemExit: Exits the program if an error occurs while loading the configuration.
    """
    # Reading Configuration Data
    try:
        with open(config_path, 'r') as config_file:
            print('config file loaded')
            # The yaml.safe_load function parses the YAML data into a Python dictionary named config_data
            config_data = yaml.safe_load(config_file)

        # Define global variables for storing configuration data
        global config_file_paths, config_controls, config_run_period, output_folder
        
        # Extract file paths, control parameters, and run period from the configuration data
        # config_file_paths = config_data.get('file_paths', {})
        config_file_paths = config_data.get('table_names', {})
        config_controls = config_data.get('control_parameters', {})
        config_run_period = config_data.get('run_period', {})
        print('file paths set')
        # Extract output configuration
        output = config_data.get('output', {})
        run_id = output.get('run_id', '')
        output_path = output.get('output_folder_path', '')
        
        # Normalize the output path and construct the full output folder path
        output_path = output_path.replace('//', '/')
        output_path = os.path.join(*output_path.split('/'))
        output_folder = os.path.join(output_path, run_id)

        # Create the necessary directories for storing results
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'Final_Plan'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'Customer_Segmentation_Charts'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'Uplift'), exist_ok=True)
    
    except Exception as e:
        # Log the error and print a message to the console
        logging.error(f"Error loading configuration from {config_path}: {e}")
        print(f"Error loading configuration from {config_path}. Please check the logs for details.")
        # Exit the program with an error code
        sys.exit(1)
    
    # Return True if the configuration is successfully loaded and directories are created
    return True

# COMMAND ----------

# MAGIC %md
# MAGIC #### The 'select_option' function facilitates the selection of an option from a dictionary based on user preference. It iterates through the dictionary of options, checking for the first option with a selection value of 'Y' (case-insensitive). If no such option is found, it returns the first option in the dictionary. If the dictionary is empty, it returns None.

# COMMAND ----------

def select_option(options):
    """
    Select an option from a dictionary based on user preference.

    This function iterates through a dictionary of options, checking for the first option
    with a selection value of 'Y' (case-insensitive). If no such option is found, it returns
    the first option in the dictionary. If the dictionary is empty, it returns None.

    Parameters:
    options (dict): A dictionary where keys are options and values are selection indicators ('Y' or 'N').

    Returns:
    str or None: The selected option if a 'Y' is found, the first option if no 'Y' is found,
                 or None if the dictionary is empty.
    """
    # Iterate through the dictionary items
    for option, selection in options.items():
        # Check if the selection is 'Y' (case-insensitive)
        if selection.upper() == 'Y':
            # Return the option if 'Y' is found
            return option
    
    # Use the first option if no 'Y' is found
    return next(iter(options), None)

# COMMAND ----------

# MAGIC %md
# MAGIC #### The 'tp_allocation' function allocates touchpoints to segments based on optimization criteria defined by the columns 'OPTIMIZED_FREQUENCY', 'GEKKO_MAX_TP', and 'FINAL_MAX_TOUCHPOINTS_SEGMENT'. It takes a DataFrame containing input data and returns a DataFrame containing the recommended touchpoint allocations with specific columns renamed.

# COMMAND ----------

def tp_allocation(df_final_gekko):
    """
    Allocate touchpoints to segments based on optimization criteria.

    This function takes a DataFrame containing input data and allocates touchpoints
    to different segments based on optimization criteria defined by the columns
    'OPTIMIZED_FREQUENCY', 'GEKKO_MAX_TP', and 'FINAL_MAX_TOUCHPOINTS_SEGMENT'.

    Parameters:
    df_final_gekko (pd.DataFrame): Input DataFrame containing the necessary data for touchpoint allocation.

    Returns:
    pd.DataFrame: A DataFrame containing the recommended touchpoint allocations with specific columns renamed.
    """

    # DataFrame Initialization
    columns = df_final_gekko.columns.tolist()
    final_df = pd.DataFrame(columns=columns)

    # Extract unique values for 'CHANNEL' and 'CONSTRAINT_SEGMENT_VALUE'
    chan_list = df_final_gekko['CHANNEL'].unique().tolist()
    seg_list = df_final_gekko['CONSTRAINT_SEGMENT_VALUE'].unique().tolist()

    # Iterate over each unique combination of 'CHANNEL' and 'CONSTRAINT_SEGMENT_VALUE'
    for i in chan_list:
        for j in seg_list:
           
            # Create a subset DataFrame based on 'CHANNEL' and 'CONSTRAINT_SEGMENT_VALUE' values
            df_rest = df_final_gekko[(df_final_gekko['CHANNEL'] == i) & (df_final_gekko['CONSTRAINT_SEGMENT_VALUE'] == j)].reset_index(drop=True)

            count = df_rest["OPTIMIZED_FREQUENCY"].sum()
            max_count = 0
            Gekko_sum = df_rest["GEKKO_MAX_TP"].sum()
            Max_sum = df_rest["FINAL_MAX_TOUCHPOINTS_SEGMENT"].sum()
            
            # Apply touchpoint allocation logic based on optimization criteria
            while max_count < count and max_count < Gekko_sum:
                for z in range(len(df_rest.index)):
                    if df_rest.at[z, 'FINAL_TOUCHPOINT'] < df_rest.at[z, 'GEKKO_MAX_TP']:
                        df_rest.at[z, 'FINAL_TOUCHPOINT'] += 1
                        max_count += 1
                    if max_count >= count or max_count >= Gekko_sum:
                        break

                if max_count >= Gekko_sum:
                    break

            while max_count < count and max_count < Max_sum:
                for z in range(len(df_rest.index)):
                    if df_rest.at[z, 'FINAL_TOUCHPOINT'] < df_rest.at[z, 'FINAL_MAX_TOUCHPOINTS_SEGMENT']:
                        df_rest.at[z, 'FINAL_TOUCHPOINT'] += 1
                        max_count += 1
                    if max_count >= count or max_count >= Max_sum:
                        break

                if max_count >= Max_sum:
                    break
            
            # Concatenate the resulting DataFrame df_rest with final_df after each iteration
            final_df = pd.concat([final_df, df_rest]).reset_index(drop=True)
    
    # DataFrame manipulation: rename columns as per the expected output format
    recommendations = final_df[['HCP_ID','CHANNEL','CONSTRAINT_SEGMENT_NAME','CONSTRAINT_SEGMENT_VALUE','FINAL_TOUCHPOINT','OPTIMIZED_FREQUENCY','RESPONSE_COEFFICIENT','NEW_BETA','PARAMETER1','ADJ_FACTOR','TRANSFORMATION','AFFINITY_SCORE','AFFINITY_SEGMENT']]
    recommendations.rename(columns={'CONSTRAINT_SEGMENT_NAME':'SEGMENT_NAME','CONSTRAINT_SEGMENT_VALUE':'SEGMENT_VALUE','FINAL_TOUCHPOINT':'FINAL_QTRLY_TOUCHPOINT'}, inplace=True)

    return recommendations

# COMMAND ----------

# MAGIC %md
# MAGIC #### The 'uplift_calc' function calculates uplift metrics based on the provided DataFrame and generates related visualizations. It adds new columns to the input DataFrame for optimized transactions, calculates overall uplift percentages, and saves the results to CSV files. Additionally, it generates and saves a bar chart visualization of the uplift metrics.

# COMMAND ----------

def uplift_calc(final_df):
    """
    Calculate uplift metrics based on the provided DataFrame and generate related visualizations.

    This function adds new columns to the input DataFrame for optimized transactions,
    calculates overall uplift percentages, and saves the results to CSV files.
    Additionally, it generates and saves a bar chart visualization of the uplift metrics.

    Parameters:
    final_df (pd.DataFrame): Input DataFrame containing the necessary data for uplift calculation.

    Returns:
    None
    """

    # Adding two new columns (MMO_TRX and OPTIMIZED_TRX) to the DataFrame using the apply method and lambda functions
    final_df["MMO_TRX"] = final_df.apply(lambda row: calculate_optimized_trx(row, tps='OPTIMIZED_FREQUENCY', beta='RESPONSE_COEFFICIENT'), axis=1)
    final_df["OPTIMIZED_TRX"] = final_df.apply(lambda row: calculate_optimized_trx(row, tps='FINAL_QTRLY_TOUCHPOINT', beta='NEW_BETA'), axis=1)
    
    # Calculate the total transactions for optimized and MMO touchpoints where affinity score is greater than 0
    Optimized_trx = final_df[final_df['AFFINITY_SCORE'] > 0]['OPTIMIZED_TRX'].sum()
    MMO_Trx = final_df[final_df['AFFINITY_SCORE'] > 0]['MMO_TRX'].sum()
    
    # Calculate the overall uplift percentage
    d = (Optimized_trx / MMO_Trx) - 1
    d = d * 100
    overall_lift = d
    
    # Calculate uplift percentages assuming different execution rates (60% and 80%)
    overall_lift3 = overall_lift * 0.6
    overall_lift4 = overall_lift * 0.8
    
    # Store uplift metrics in a DataFrame and round values to four decimal places
    uplift_data = np.array([[overall_lift / 100, overall_lift * 0.8 / 100, overall_lift * 0.6 / 100]])
    ipp_uplift = pd.DataFrame(uplift_data, columns=["Total Uplift", "Assuming 80% Execution", "Assuming 60% Execution"])
    ipp_uplift = ipp_uplift.round(decimals=4)
    
    # Save the uplift metrics and the DataFrame with calculated metrics to CSV files
    ipp_uplift.to_csv(f"{output_folder}/IPP_Uplift.csv", index=False)
    final_df.to_csv(f'{output_folder}/final_output_with_opti_trx.csv', index=False)

    # Print the overall uplift percentages
    print(f"Overall_Uplift : {round(overall_lift, 2)}%\n",
          f"Uplift 80% execution : {round(overall_lift4, 2)}%\n",
          f"Uplift 60% execution: {round(overall_lift3, 2)}%\n")
    
    # Generate a bar chart visualization of the uplift metrics

    plt.rcParams.update({'font.size': 22})
    colors = ['#EF995F', '#ccffcc', '#FFD68A', '#FFA500', '#FFB52E', '#FFC55C', '#C5E0B4']
    cmap1 = LinearSegmentedColormap.from_list("my_colormap", colors)

    df = pd.DataFrame({'data': [overall_lift, overall_lift3],
                       'data2': [np.nan, overall_lift4 - overall_lift3]}, index=['Potential Uplift', 'Expected Realizable Uplift'])

    ax = df.plot.bar(figsize=(9, 7), rot=0, legend=False, align='center', width=0.4, stacked=True, colormap=cmap1)

    texts = ['Estimated opportunity with the analysis', 'Assuming 60% of the execution']
    a = "~" + str(round(overall_lift, 1)) + "%"
    b = "~" + str(round(overall_lift3, 1)) + "%"
    c = "~" + str(round(overall_lift4, 1)) + "%"
    text = [a, b]
    text2 = ["", c]

    for i, v in enumerate(df['data']):
        ax.plot([i + 0.2, ax.get_xlim()[-1]], [v, v],
                ls='--', c='k')
        ax.text(ax.get_xlim()[-1] + 0.1, v, texts[i], size=20)
        ax.text(i, v + 0.1, text[i], size=20, horizontalalignment='center')

    texts = ["", 'Assuming 80% of the execution']

    for i, v in enumerate(df['data2']):
        ax.plot([i + 0.1, ax.get_xlim()[-1]], [v + overall_lift3, v + overall_lift3],
                ls='--', c='k')
        ax.text(ax.get_xlim()[-1] + .1, v + overall_lift3, texts[i], size=20)
        ax.text(i, v + overall_lift3 + 0.05, text2[i], size=20, horizontalalignment='center')

    ax.axes.yaxis.set_visible(True)
    ax.tick_params(axis='x', which='major', pad=15)
    ax.axes.yaxis.set_ticklabels([])

    for spine in ax.spines:
        ax.spines[spine].set_visible(False)

    ax.set_facecolor('xkcd:white')

    # Save the plot as an image
    uplift_output_path = os.path.join(output_folder, 'Uplift', 'Uplift.png')
    plt.savefig(uplift_output_path, bbox_inches="tight")

    return None

# COMMAND ----------

# MAGIC %md
# MAGIC #### The 'quarterly_to_monthly' function converts quarterly touchpoint allocations to monthly allocations based on configuration settings and historical data. It processes a DataFrame of touchpoint allocation recommendations, adjusts the allocations based on historical data and predefined rules, and returns a DataFrame with the final monthly touchpoint allocations.

# COMMAND ----------

def quarterly_to_monthly(recommendations):
    """
    Convert quarterly touchpoint allocations to monthly allocations based on configuration settings and historical data.

    This function processes a DataFrame of touchpoint allocation recommendations, adjusts the allocations based on historical
    data and predefined rules, and returns a DataFrame with the final monthly touchpoint allocations.

    Parameters:
    recommendations (pd.DataFrame): DataFrame containing touchpoint allocation recommendations with columns like 
                                    'HCP_ID', 'CHANNEL', 'FINAL_QTRLY_TOUCHPOINT', etc.

    Returns:
    pd.DataFrame: DataFrame with monthly touchpoint allocations including columns 'HCP_ID', 'CHANNEL', 'SEGMENT_NAME', 
                  'SEGMENT_VALUE', 'FINAL_MONTHLY_TOUCHPOINTS', and 'FINAL_QTRLY_TOUCHPOINT'.
    """
    
    # Extract configuration parameters for distribution rules and splits
    dist_rule_monthly = select_option(config_controls.get('dist_rule_monthly', {}))
    monthly_dist_split = config_controls.get('monthly_dist_split', '')

    # Get the refresh date from the configuration and calculate the last two months
    refresh_date = config_run_period.get('refresh_date')
    refresh_date = pd.to_datetime(refresh_date)
    last_2_months = refresh_date - timedelta(days=60)
    last_2_months_str = last_2_months.strftime('%Y-%m-%d')
    
    # Load historical data from the CSV file specified in the configuration
    historical_data_path = config_file_paths.get('historical_data', '')
    global historical_data
    # historical_data = pd.read_csv(historical_data_path)
    historical_data = spark.table(historical_data_path).toPandas()
    historical_data = historical_data.drop_duplicates()
    historical_data.columns = historical_data.columns.str.upper()
    historical_data.rename(columns={'CHANNEL_ID':'CHANNEL'}, inplace=True)

    # Standardize and filter historical data
    historical_data['EXPOSED_ON'] = pd.to_datetime(historical_data['EXPOSED_ON'])
    historical_data = historical_data.apply(lambda x: x.str.upper() if x.dtype == 'O' else x)
    historical_data = historical_data.apply(lambda x: x.str.replace(' ', '_') if x.dtype == 'O' else x)

    historical_data['EXPOSED_ON'] = historical_data['EXPOSED_ON'].dt.strftime("%Y-%m-%d")
    historical_data['HCP_ID'] = historical_data['HCP_ID'].astype(str)
    last_2_mth_exp = historical_data[historical_data['EXPOSED_ON'] >= last_2_months_str]
    last_2_mth_tps = last_2_mth_exp.groupby(['HCP_ID', 'CHANNEL']).size().reset_index(name='last_2_mth_tps')
    
    # Merge historical data with recommendations and fill missing values
    recommendations = recommendations.merge(last_2_mth_tps.drop_duplicates(), how='left', on=['HCP_ID', 'CHANNEL'])
    recommendations['last_2_mth_tps'] = recommendations['last_2_mth_tps'].fillna(0)
    
    # Calculate monthly touchpoints based on equal distribution and difference from last two months
    recommendations['Monthly_tps_eql_dist'] = np.floor(recommendations['FINAL_QTRLY_TOUCHPOINT'] / 3)
    recommendations['Monthly_tps_diff_lst_2_mth'] = recommendations['FINAL_QTRLY_TOUCHPOINT'] - recommendations['last_2_mth_tps']
    recommendations['Monthly_tps_diff_lst_2_mth'] = np.where(recommendations['Monthly_tps_diff_lst_2_mth'] < 0, 0, recommendations['Monthly_tps_diff_lst_2_mth'])
    
    # Adjust touchpoint allocations based on the selected distribution rule
    if dist_rule_monthly == 'EQUAL':
        recommendations['FINAL_MONTHLY_TOUCHPOINTS'] = recommendations['Monthly_tps_eql_dist']
    elif dist_rule_monthly == 'PREDEFINED_SPLIT':
        if isinstance(monthly_dist_split, float) and 0 < monthly_dist_split <= 1:
            recommendations['FINAL_MONTHLY_TOUCHPOINTS'] = recommendations['FINAL_QTRLY_TOUCHPOINT'] * monthly_dist_split
        else:
            print('Monthly split not in expected format. Please provide a split between 0 to 1')
            logging.error('Monthly Split not in required format.')
            raise TypeError('Monthly split out of range')
    elif dist_rule_monthly == 'HISTORICAL':
        recommendations['FINAL_MONTHLY_TOUCHPOINTS'] = recommendations['Monthly_tps_diff_lst_2_mth']
    
    # Select relevant columns for the final output
    recommendations = recommendations[['HCP_ID', 'CHANNEL', 'SEGMENT_NAME', 'SEGMENT_VALUE', 'FINAL_MONTHLY_TOUCHPOINTS', 'FINAL_QTRLY_TOUCHPOINT']]
    
    return recommendations

# COMMAND ----------

# MAGIC %md
# MAGIC #### The 'weekly_tps' function distributes monthly touchpoint allocations into weekly allocations for each HCP (Health Care Professional). It processes a DataFrame of touchpoint allocation recommendations and distributes the monthly touchpoints into weekly touchpoints based on predefined rules.

# COMMAND ----------

def weekly_tps(recommendations, tps_column=None, num_weeks=None):
    """
    Distribute monthly touchpoint allocations into weekly allocations for each HCP (Health Care Professional).

    This function processes a DataFrame of touchpoint allocation recommendations and distributes the monthly touchpoints
    into weekly touchpoints based on predefined rules.

    Parameters:
    recommendations (pd.DataFrame): DataFrame containing touchpoint allocation recommendations with columns like 
                                    'HCP_ID', 'FINAL_MONTHLY_TOUCHPOINTS', etc.
    tps_column (str, optional): The column name in the recommendations DataFrame that contains the monthly touchpoints. 
                                Default is 'FINAL_MONTHLY_TOUCHPOINTS'.
    num_weeks (int, optional): The number of weeks in a month for distribution. Default is 4.

    Returns:
    pd.DataFrame: DataFrame with weekly touchpoint allocations including columns for each week and the original columns.
    """
    
    # Extract the distribution rule for weekly planning from the configuration
    dist_rule_weekly = select_option(config_controls.get('dist_rule_weekly', {}))
    
    # Set default values for tps_column and num_weeks if not provided
    num_weeks = num_weeks if num_weeks is not None else 4
    tps_column = tps_column if tps_column is not None else 'FINAL_MONTHLY_TOUCHPOINTS'

    # Create column names for weekly touchpoints
    weekly_tps_cols = [f'WEEK_{i+1}_TOTAL_TPS' for i in range(num_weeks)]
    
    # Summarize the monthly touchpoints per HCP and initialize weekly touchpoint columns to 0
    recommendations_hcp = recommendations.groupby('HCP_ID')[tps_column].sum().reset_index(name='FINAL_MONTHLY_TOUCHPOINTS_TOTAL')
    recommendations_hcp[weekly_tps_cols] = 0

    # Handle 'AGGRESSIVE' distribution rule: attempt to retrieve the weekly maximum touchpoints from configuration
    if dist_rule_weekly == 'AGGRESSIVE':
        try:
            weekly_max_tp = config_controls.get('weekly_max_tp_aggresive')
            weekly_max_tp = int(weekly_max_tp)
        except (TypeError, ValueError):
            print('Weekly Max TP not provided in correct format for aggressive approach. Considering all monthly tps available to be sent in one week')
            logging.warning('Weekly Max TP not provided in correct format for aggressive approach')
            weekly_max_tp = float('inf')  # Use all monthly tps available to be sent in one week
    
    # Loop through each unique monthly touchpoint count and distribute it into weeks
    for tp in recommendations_hcp['FINAL_MONTHLY_TOUCHPOINTS_TOTAL'].unique():
        tp = int(tp)
        
        # 'EQUAL' distribution rule
        if dist_rule_weekly == 'EQUAL':
            if tp == num_weeks:
                weekly_tps = [1] * num_weeks
            elif tp < num_weeks:
                weekly_tps = [1] * tp + [0] * (num_weeks - tp)  # Distribute equally
            elif tp > num_weeks:
                week_tps = tp // num_weeks
                remaining_tps = tp % num_weeks
                weekly_tps = [week_tps] * num_weeks

                for i in range(0, num_weeks, 2):
                    if remaining_tps > 0:
                        weekly_tps[i] += 1
                        remaining_tps -= 1
                for i in range(1, num_weeks, 2):
                    if remaining_tps > 0:
                        weekly_tps[i] += 1
                        remaining_tps -= 1
        
        # 'AGGRESSIVE' distribution rule
        elif dist_rule_weekly == 'AGGRESSIVE':
            weekly_tps = [min(tp, weekly_max_tp)] * num_weeks

        # Assign the calculated weekly touchpoints to the DataFrame
        recommendations_hcp.loc[recommendations_hcp['FINAL_MONTHLY_TOUCHPOINTS_TOTAL'] == tp, weekly_tps_cols] = weekly_tps

    # Merge the weekly touchpoint allocations back into the original recommendations DataFrame
    recommendations_hcp = recommendations.merge(recommendations_hcp, how='left', on='HCP_ID')
    
    return recommendations_hcp

# COMMAND ----------

# MAGIC %md
# MAGIC #### This function allocates weekly touchpoints to Health Care Professionals (HCPs) based on various criteria including historical data, distribution rules, priority, and minimum gap constraints. It reads input files, processes the data, applies business rules, and returns the final allocation of touchpoints for each HCP.

# COMMAND ----------

def weekly_seq(recommendations_hcp, historical_data):
    """
    Allocates weekly touchpoints to Health Care Professionals (HCPs).

    Parameters:
    recommendations_hcp (pd.DataFrame): DataFrame containing recommended touchpoints for HCPs.
    historical_data (pd.DataFrame): DataFrame containing historical touchpoint data.

    Returns:
    pd.DataFrame: DataFrame with weekly touchpoint allocations for each HCP.
    """
    
    # Retrieve distribution rule for the weekly allocation from the config
    dist_rule_weekly = select_option(config_controls.get('dist_rule_weekly', {}))
    
    # Get the path for the priority file from the config
    priority_file_path = config_file_paths.get('priority', '')
    
    if priority_file_path is None:
        try:
            # Generate the priority file using the model input if not provided
            priority_file = model_input(config_file_paths)
        except Exception as e:
            logging.error(f"Error creating priority file using model: {e}")
            sys.exit(1)
    else:
        try:
            # Load the priority file
            # priority_file = pd.read_csv(priority_file_path)
            priority_file = spark.table(priority_file_path).toPandas()
            priority_file.columns = priority_file.columns.str.upper()
            priority_file['HCP_ID'] = priority_file['HCP_ID'].astype(str)
            priority_file = priority_file.apply(lambda x: x.str.upper() if x.dtype=='O' else x)
            priority_file = priority_file.apply(lambda x: x.str.replace(' ', '_') if x.dtype=='O' else x)
        except Exception as e:
            logging.error(f"Error loading priority file, Please check the path : {e}")
            sys.exit(1)
    
    # Retrieve start and refresh dates from the config
    start_date = config_run_period.get('start_date', '')
    refresh_date = config_run_period.get('refresh_date', '')

    start_date = pd.to_datetime(start_date)
    refresh_date = pd.to_datetime(refresh_date)

    # Merge priority file with recommendations data
    recommendations_hcp = recommendations_hcp.merge(priority_file, how='left', on=['HCP_ID', 'CHANNEL'])
    recommendations_hcp = recommendations_hcp.sort_values(['HCP_ID', 'PRIORITY', 'CHANNEL'])
    
    # Assign priority numbers within each HCP group
    recommendations_hcp['PRIORITY'] = recommendations_hcp.groupby('HCP_ID').cumcount() + 1
    
    # Initialize columns
    recommendations_hcp['NOT_AVAILABLE'] = 0
    recommendations_hcp['FINAL_MONTHLY_TOUCHPOINTS_REMAINING'] = recommendations_hcp['FINAL_MONTHLY_TOUCHPOINTS'].copy()
    
    # Load minimum gap constraints
    min_gap_path = config_file_paths.get('min_gap', '')
    # min_gap = pd.read_csv(min_gap_path)
    min_gap = spark.table(min_gap_path).toPandas()
    min_gap.columns = min_gap.columns.str.upper()
    min_gap = min_gap.apply(lambda x: x.str.upper() if x.dtype=='O' else x)
    min_gap = min_gap.apply(lambda x: x.str.replace(' ', '_') if x.dtype=='O' else x)
    
    # Filter out invalid minimum gaps
    min_gap = min_gap[(min_gap['CHANNEL_1'] == min_gap['CHANNEL_2']) or (min_gap['MIN_GAP'] >= 1)]
    min_gap = min_gap[min_gap['MIN_GAP'] > 0]
    
    historical_data['EXPOSED_ON'] = pd.to_datetime(historical_data['EXPOSED_ON'])
    week_list = ['WEEK_1', 'WEEK_2', 'WEEK_3', 'WEEK_4']

    # Adjust week list based on the difference between start and refresh dates
    if start_date != refresh_date:
        day_difference = (refresh_date - start_date).days
        week = min(4, int((day_difference // 7.1)) + 2)
        week_list = [f'WEEK_{i}' for i in range(week, 5)]
    
    weekly_channel_tps_cols = [week + '_CHANNEL_TPS' for week in week_list]
    recommendations_hcp[weekly_channel_tps_cols] = 0

    for grp_name, grp in recommendations_hcp.groupby('HCP_ID'):
        historical_exp = historical_data[historical_data['HCP_ID'] == grp_name]
        total_tps = 0
        
        if dist_rule_weekly == 'AGGRESSIVE':
            total_tps = grp['WEEK_1_TOTAL_TPS'].values[0]
        
        for week in week_list:
            if dist_rule_weekly == 'EQUAL':
                total_tps = grp[week + '_TOTAL_TPS'].values[0] + total_tps 
            channels = set(grp['CHANNEL'].unique())

            not_available_channels = set()
            break_loop = False
            
            while total_tps > 0 and not channels.issubset(not_available_channels) and not break_loop:
                for idx, row_hcp in grp.iterrows():
                    if total_tps <= 0 or channels.issubset(not_available_channels):
                        break_loop = True
                        break
                    channel = recommendations_hcp.at[idx, 'CHANNEL'] 
                    channel_monthly_tps = recommendations_hcp.at[idx, 'FINAL_MONTHLY_TOUCHPOINTS_REMAINING']
                    available_channels = set()

                    if channel in not_available_channels or channel_monthly_tps <= 0:
                        recommendations_hcp.loc[idx, 'NOT_AVAILABLE'] = 1
                        not_available_channels.add(channel)
                        continue
                
                    # Select minimum gap constraints applicable to the HCP based on their segment
                    min_gap_subset = min_gap[(min_gap['CHANNEL_1'] == channel) or (min_gap['CHANNEL_2'] == channel)]
                    if not min_gap_subset.empty:
                        hcp_segment = row_hcp['SEGMENT_VALUE']
                        min_gap_req = min_gap_subset[min_gap_subset['SEGMENT_VALUE'] == hcp_segment]
                        if not min_gap_req.empty:
                            for i, row_min_gap in min_gap_req.iterrows():
                                channel1 = row_min_gap['CHANNEL_1'] 
                                channel2 = row_min_gap['CHANNEL_2']
                                min_gap_value = row_min_gap['MIN_GAP'] 

                                if min_gap_value >= 1:
                                    lookback_period = refresh_date + timedelta(week_list.index(week) * 7) - timedelta(days=(min_gap_value - 1) * 7)  # Define the lookback period based on refresh date and min gap
                                    future_period = refresh_date + timedelta(week_list.index(week) * 7) + timedelta(days=(min_gap_value) * 7)
                                if min_gap_value < 1 and min_gap_value > 0:
                                    lookback_period = refresh_date + timedelta(week_list.index(week) * 7)
                                    future_period = refresh_date + timedelta(week_list.index(week) * 7) + timedelta(days=7)
                                
                                channel1_data = historical_exp[(historical_exp['CHANNEL'] == channel1) & (historical_exp['EXPOSED_ON'] >= lookback_period) & (historical_exp['EXPOSED_ON'] <= future_period)]
                                channel2_data = historical_exp[(historical_exp['CHANNEL'] == channel2) & (historical_exp['EXPOSED_ON'] >= lookback_period) & (historical_exp['EXPOSED_ON'] <= future_period)]
                                
                                if channel1 == channel2:
                                   if len(channel1_data) >= 1/min_gap_value:
                                       not_available_channels.add(channel1)
                                       
                                if channel1 != channel2:     
                                    if not channel1_data.empty:  # If channel 1 data is not empty add channel2 to not_available_channel
                                        not_available_channels.add(channel2)
                                        available_channels.add(channel2)
                                    if not channel2_data.empty:  # If channel 2 data is not empty, add channel1 to not_available_channels
                                        not_available_channels.add(channel1)
                                        available_channels.add(channel1)
                                if channel in not_available_channels: 
                                    recommendations_hcp.loc[idx, 'NOT_AVAILABLE'] = 1
                                    not_available_channels = [x for x in not_available_channels if x not in available_channels]
                                    not_available_channels = set(not_available_channels)
                                    break

                    if row_hcp['NOT_AVAILABLE'] == 0:
                        recommendations_hcp.loc[idx, week + '_CHANNEL_TPS'] += 1
                        not_same_week_channels = min_gap_req[min_gap_req['MIN_GAP'] >= 1][['CHANNEL_1', 'CHANNEL_2']].values.flatten().tolist()
                        not_available_channels.update(not_same_week_channels)

                        total_tps -= 1
                        recommendations_hcp.loc[idx, 'FINAL_MONTHLY_TOUCHPOINTS_REMAINING'] -= 1
                        channel_monthly_tps -= 1
                        new_tp = {'HCP_ID': grp_name, 'CHANNEL': channel, 'EXPOSED_ON': refresh_date + timedelta(week_list.index(week) * 7)}
                        historical_exp = pd.concat([historical_exp, pd.DataFrame([new_tp])], ignore_index=True)
    
    req_col = ['HCP_ID', 'CHANNEL', 'SEGMENT_NAME', 'SEGMENT_VALUE', 'FINAL_MONTHLY_TOUCHPOINTS', 'FINAL_QTRLY_TOUCHPOINT', 'PRIORITY'] + recommendations_hcp.filter(like='CHANNEL_TPS').columns.tolist()               
    recommendations_final = recommendations_hcp[req_col]
    return recommendations_final

# COMMAND ----------

# MAGIC %md
# MAGIC #### The 'model_input' function generates priority scores for Health Care Professionals (HCPs) and channels based on a trained machine learning model. It leverages paths to the model, model features, and input data files provided in a configuration dictionary.

# COMMAND ----------

# Generating priority scores for Health Care Professionals (HCPs) and channels based on a trained machine learning model
def model_input(config_file_paths):
    """
    Generates priority scores for Health Care Professionals (HCPs) and channels based on a trained machine learning model.

    Args:
    - config_file_paths (dict): A dictionary containing paths to the model, model features, and input data files.

    Returns:
    - hcp_priority (DataFrame): A DataFrame containing priority scores for HCPs and channels, sorted by HCP ID and priority.
    """
    print("In model function")
    model_path = config_file_paths.get('model','')
    model_features_path = config_file_paths.get('model_cols','')
    input_data_path = config_file_paths.get('model_input_data','')
    
    model  =  joblib.load(model_path)
    model_cols = joblib.load(model_features_path)
    
    model_data = pd.read_csv(input_data_path)
     
    missing_cols = [x.upper() for x in model_cols if x.upper() not in model_data.columns.str.upper()]
    print('Following features are missing from the input data:',missing_cols)    
    
    predictions = model.predict_proba(model_data[model_cols])
    
    model_data['ENG_PROB'] = predictions[:, 1] # considering a binary classifier
    
    hcp_priority = model_data.groupby(['HCP_ID', 'CHANNEL'])['ENG_PROB'].mean().reset_index() # taking avg prob
    hcp_priority = model_data[['HCP_ID','CHANNEL','ENG_PROB']].drop_duplicates()
    hcp_priority.sort_values(by=['HCP_ID', 'ENG_PROB'], ascending=[True, False], inplace=True)
    hcp_priority['PRIORITY'] = hcp_priority.groupby('HCP_ID').cumcount() + 1

    return hcp_priority


# COMMAND ----------

# MAGIC %md
# MAGIC #### The 'create_subsets' function creates subsets of the input DataFrame based on unique combinations of channels and constraint segment values. This function handles the creation and processing of these subsets with specific rules to ensure a balanced and meaningful distribution of data.

# COMMAND ----------

def create_subsets(df_final):
    """
    Creates subsets of the input DataFrame based on unique combinations of channels and constraint segment values.

    Args:
    - df_final (DataFrame): The DataFrame containing the final data.

    Returns:
    - train_df (DataFrame): The DataFrame containing subsets of data based on unique combinations of channels and constraint segment values.
    """
    columns = df_final.columns.tolist()
    train_df = pd.DataFrame(columns=columns)

    chan_list = df_final['CHANNEL'].unique().tolist()
    seg_list = df_final['CONSTRAINT_SEGMENT_VALUE'].unique().tolist()

    # Iterating over unique combinations of channels and constraint segment values
    for i in chan_list:
        for j in seg_list:
            df_temp = df_final[(df_final['CHANNEL'] == i) & (df_final['CONSTRAINT_SEGMENT_VALUE'] == j)].reset_index(drop=True)
            
            # Skipping empty subsets
            if df_temp['HCP_ID'].count() == 0:
                continue
            
            # Creating subsets based on affinity score
            if df_temp['HCP_ID'].count() <= 500:
                train = df_temp
            else:
                single_value = df_temp[~df_temp['AFFINITY_SCORE'].duplicated(keep=False)]
                df_temp = df_temp[df_temp['AFFINITY_SCORE'].duplicated(keep=False)]
                n = 500 / df_temp['HCP_ID'].count()
                train, test = train_test_split(df_temp, test_size=1 - n, random_state=0, stratify=df_temp[['AFFINITY_SCORE']])
            
            # Sorting the subset based on affinity score
            train = train.reset_index(drop=True)
            train = train.sort_values(["AFFINITY_SCORE"], ascending=[False])

            train_df = pd.concat([train_df, train, single_value]).reset_index(drop=True)

    return train_df


# COMMAND ----------

# MAGIC %md
# MAGIC #### The 'run_gekko_optimization' function performs optimization using the GEKKO optimization library on a given DataFrame, train_df. The function iterates over unique combinations of channels and constraint segment values, applying optimization techniques to determine the optimal touchpoints for each subset of data.

# COMMAND ----------

def run_gekko_optimization(train_df):
    """
    Performs optimization using the GEKKO optimization library on a DataFrame train_df.

    Args:
    - train_df (DataFrame): The DataFrame containing the data for optimization.

    Returns:
    - df_for_algo (DataFrame): The DataFrame containing the optimized results.
    """

    # Extracting column names from the DataFrame
    columns = train_df.columns.tolist()

    # Initializing an empty DataFrame to store optimized results
    df_for_algo = pd.DataFrame(columns=columns)

    # Extracting unique channel and constraint segment values
    chan_list = train_df['CHANNEL'].unique().tolist()
    seg_list = train_df['CONSTRAINT_SEGMENT_VALUE'].unique().tolist()

    # Iterating over unique combinations of channels and constraint segment values
    for i in chan_list:
        for j in seg_list:
            # Filtering DataFrame based on current channel and constraint segment value
            gekko_db = train_df[(train_df['CHANNEL'] == i) & (train_df['CONSTRAINT_SEGMENT_VALUE'] == j)].reset_index(drop=True)

            # Skipping empty subsets
            if gekko_db['HCP_ID'].count() == 0:
                continue

            # Getting maximum optimized frequency and number of rows
            max_of = gekko_db["OPTIMIZED_FREQUENCY"].sum()
            n = len(gekko_db)

            # initialize model
            m = GEKKO(remote=False)

            # initialize variable
            tp = np.array([m.Var(lb=gekko_db.at[i, "FINAL_MIN_TOUCHPOINTS_SEGMENT"], ub=gekko_db.at[i, "FINAL_MAX_TOUCHPOINTS_SEGMENT"]) for i in range(n)])

            # constraints
            m.Equation(m.sum(list(tp)) <= max_of)

            # objective
            New_Beta = gekko_db['NEW_BETA'].values
            a = gekko_db.at[0, "ADJ_FACTOR"]
            k = gekko_db.at[0, "PARAMETER1"]

            # objective
            if gekko_db.at[0, "TRANSFORMATION"] == "LOG":
                [m.Maximize(m.log(i) * a * j) for i, j in zip((1 + tp * k), New_Beta)]
                m.solve(disp=False)

            if gekko_db.at[0, "TRANSFORMATION"] == "POWER":
                [m.Maximize((i ** k) * j * a) for i, j in zip(tp, New_Beta)]
                m.solve(disp=False)

            if gekko_db.at[0, "TRANSFORMATION"] == "LINEAR":
                [m.Maximize(i * j) for i, j in zip(tp, New_Beta)]
                m.solve(disp=False)

            if gekko_db.at[0, "TRANSFORMATION"] == "NEGEX":
                [m.Maximize((1 - m.exp(-i)) * a * j) for i, j in zip(tp * k, New_Beta)]
                m.solve(disp=False)

            # Extracting optimized results
            x = [tp[i][0] for i in range(n)]
            x = np.array(x)
            gekko_db['GEKKO_MAX_TP'] = x

            # Rounding the optimized touchpoints to the nearest integer
            gekko_db["GEKKO_MAX_TP"] = gekko_db["GEKKO_MAX_TP"].round(decimals=0)

            # Concatenating current optimized results with the main DataFrame
            df_for_algo = pd.concat([gekko_db, df_for_algo]).reset_index(drop=True)

    return df_for_algo

# COMMAND ----------

# MAGIC %md
# MAGIC #### The 'final_merge' function merges an original DataFrame (df_final) with another DataFrame (algo_df) containing optimized maximum touchpoints calculated using the GEKKO optimization. This function ensures that the resulting DataFrame (df_final_gekko) includes the optimized touchpoints, properly sorted and initialized for further processing.

# COMMAND ----------

def final_merge(df_final, algo_df):
    """
    Merges the original DataFrame df_final with another DataFrame algo_df, 
    which contains the optimized maximum touchpoints calculated using GEKKO optimization.

    Args:
    - df_final (DataFrame): The original DataFrame containing the final data.
    - algo_df (DataFrame): The DataFrame containing the optimized maximum touchpoints calculated using GEKKO optimization.

    Returns:
    - df_final_gekko (DataFrame): The merged DataFrame containing the final data along with the optimized maximum touchpoints.
    """

    # Grouping algo_df by channel, constraint segment value, and affinity score and getting the maximum optimized touchpoints
    algo_df = algo_df.groupby(['CHANNEL', 'CONSTRAINT_SEGMENT_VALUE', 'AFFINITY_SCORE'])["GEKKO_MAX_TP"].max().reset_index()

    # Resetting the index of algo_df
    algo_df = algo_df.reset_index(drop=True)

    # Merging df_final with algo_df based on channel, constraint segment value, and affinity score
    df_final_gekko = pd.merge(df_final, algo_df, how='right', on=['CHANNEL', 'CONSTRAINT_SEGMENT_VALUE', 'AFFINITY_SCORE'])

    # Sorting df_final_gekko based on channel, constraint segment value, and affinity score
    df_final_gekko = df_final_gekko.sort_values(['CHANNEL', 'CONSTRAINT_SEGMENT_VALUE', 'AFFINITY_SCORE'], ascending=[True, True, False]).reset_index(drop=True)

    # Adding a new column 'FINAL_TOUCHPOINT' and initializing it with 0
    df_final_gekko["FINAL_TOUCHPOINT"] = 0
    
    return df_final_gekko

# COMMAND ----------

# MAGIC %md
# MAGIC #### The 'run_constraint_module' function orchestrates the execution of various modules related to constraint optimization, leveraging configuration settings from a specified file.

# COMMAND ----------

def run_constraint_module(config_path):
    """
    Orchestrates the execution of various modules related to constraint optimization.

    Args:
    - config_path (str): The path to the configuration file.

    Returns:
    - None
    """

    # Load configuration from the provided config file
    load_config(config_path) 

    # Validate files
    validation_result = validate_files(config_file_paths)

    # Check if the script is running in Databricks environment
    if 'databricks' in sys.modules:
        if validation_result:
            print('Processing files for optimization module')
            df_final = process_files(config_file_paths, allow_processing_with_errors=True)
            if df_final is not None:
                print('Running Optimization Model in Databricks environment')
                # Run optimization model
            else:
                print('Errors in Processing Files. Check Logs for more details')
        else:
            print("Validation failed. Please check the logs for details.")
            sys.exit(1)
    # else:
        # User to choose whether to proceed with processing even if there are validation errors
    proceed_with_errors = 'y'
    if not validation_result:
        proceed_with_errors = input("Do you want to proceed with processing despite validation errors? (y/n): ")
    if proceed_with_errors.lower() == 'y':
        df_final = process_files(config_file_paths, allow_processing_with_errors=True)
        affinity_segment_charts(df_final)
        if df_final is not None:
            print('Running Optimization Model')
            train_df = create_subsets(df_final)
            df_for_algo = run_gekko_optimization(train_df)
            quarterly_gekko_df = final_merge(df_final, df_for_algo)

            print('Optimization model run completed')
            print('Allocating_final_quarterly_tps')

            final_quarterly_plan = tp_allocation(quarterly_gekko_df)
            print('Calculating Uplift')
            uplift_calc(final_quarterly_plan)
            final_output_level = select_option(config_controls.get('final_output_level', {}))
            if final_output_level == 'QUARTERLY':
                output_path = os.path.join(output_folder,'Final_Plan','HCP_Quarterly_Plan.csv')
                final_quarterly_plan.to_csv(output_path,index=False)
                print('Quarterly HCP Promotional Plan Created')
                # sys.exit()
                quit()
            else:
                monthly_plan  = quarterly_to_monthly(final_quarterly_plan) 
                if final_output_level == 'MONTHLY':
                    print('Monthly HCP Promotional Plan Created')
                    output_path = os.path.join(output_folder,'Final_Plan','HCP_Monthly_Plan.csv')
                    monthly_plan.to_csv(output_path,index=False)
                    # sys.exit()
                    quit()
                else:
                    print('Creating Weekly HCP Promotional Plan')
                    weekly_plan = weekly_tps(monthly_plan)
                    final_weekly_seq  = weekly_seq(weekly_plan,historical_data)
                    output_path = os.path.join(output_folder,'Final_Plan','HCP_Weekly_Plan.csv')
                    final_weekly_seq.to_csv(output_path,index=False)
                    # sys.exit()
                    quit()    
        else:
            print('Errors in Processing Files. Check Logs for more details')
            sys.exit(1)
    else:
        print("Processing aborted.Validation failed.Please check the logs for details.")
        sys.exit(1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### The 'model_start' function serves as the entry point for initiating the optimization model. It sets the configuration path and triggers the execution of the constraint optimization module.

# COMMAND ----------

def model_start():
    """
    Entry point for starting the optimization model.

    This function sets the configuration path and initiates the execution of the constraint optimization module.
    """

    # Setting the configuration path
    # config_path = sys.argv[1]
    config_path = "./Archive/config.yaml"
    # config_path = "config.yaml"

    # Running the constraint optimization module with the specified configuration path
    run_constraint_module(config_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Calling the function to start model run

# COMMAND ----------

model_start()

# COMMAND ----------


