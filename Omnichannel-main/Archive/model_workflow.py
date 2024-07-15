# Databricks notebook source
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import os
import sys
import time
import warnings

import re as re

import math

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import gekko
from gekko import GEKKO
import time
from itertools import product
import joblib
import yaml
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import logging
import ast

# COMMAND ----------

logging.basicConfig(filename='lnba_validation_log.txt', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# COMMAND ----------

# Expected file, columns and their dtpyes 
columns_dtypes = {
    "customer_data": {
        "HCP_ID": str,
        "CHANNEL": str,
        "AFFINITY_SCORE": float
    },
    "mmix_output": {
        "CHANNEL": str,
        "MMIX_SEGMENT_NAME": str,
        "MMIX_SEGMENT_VALUE": str,
        "PARAMETER1": float,
        "ADJ_FACTOR": float,
        "RESPONSE_COEFFICIENT": float,
        "OPTIMIZED_FREQUENCY": float,
        "TRANSFORMATION": str,
        "GRANULARITY": str
    },
    "asset_availability": {
        "CHANNEL": str,
        "TACTIC_ID": str,
        "ASSET_FREQUENCY": float,
        "GRANULARITY": str,
        "SEGMENT_NAME": str,
        "SEGMENT_VALUE": str
    },
    "vendor_contract": {
        "CHANNEL": str,
        "MAX_VOLUME": float,
        "GRANULARITY": str,
    },
    "min_gap": {
        "CHANNEL_1": str,
        "CHANNEL_2": str,
        "MIN_GAP": float,
        "GRANULARITY": str,
        "SEGMENT_NAME": str,
        "SEGMENT_VALUE": str
    },
    "extra_constraint_file": {
        "CHANNEL": str,
        "SEGMENT_NAME": str,
        "SEGMENT_VALUE": str,
        "MIN_TPS": float,
        "MAX_TPS": float,
        "GRANULARITY": str
    },
    "engagement_goal": {
        "CHANNEL": str,
        "TYPE": str,
        "TARGET": float,
        "ENGAGEMENT_RATE": float,
        "GRANULARITY": str,
        "SEGMENT_NAME": str,
        "SEGMENT_VALUE": str
    },
    "constraint_segment": {
        "CHANNEL": str,
        "CONSTRAINT_SEGMENT_NAME": str
    },
    "priority": {
        "HCP_ID": str,
        "CHANNEL": str,
        "PRIORITY": str
    },
    "historical_data": {
        "HCP_ID": str,
        "CHANNEL_ID": str,
        "EXPOSED_ON": str
    },
}

# COMMAND ----------

granularity_mapping = {
        'WEEKLY': 13,  
        'DAILY': 91,
        'MONTHLY': 3,
        'QUARTERLY': 1,
        'SEMESTERLY': 0.5,
        'YEARLY': 0.25
    }

# COMMAND ----------

def validate_files(file_paths):
    all_valid = True  # track if all files are valid
    for file_category, file_path in file_paths.items():

        # Check if the file path is provided
        if not file_path:
            logging.warning(f"File path not provided for {file_category}. Please check if it's required.")
            print(f"File path not provided for {file_category}. Please check if it's required.")
            all_valid = False
            continue

        # Check if the file path exists
        if not os.path.isfile(file_path):
            logging.warning(f"File path for {file_category} does not exist: {file_path}")
            print(f"File path for {file_category} does not exist: {file_path}")
            all_valid = False
            continue

        # Check if the file path is a CSV file
        if not file_path.lower().endswith('.csv'):
            logging.warning(f"File path for {file_category} is not a CSV file: {file_path}")
            print(f"File path for {file_category} is not a CSV file: {file_path}")
            all_valid = False
            continue

        # Check if the file is readable
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logging.warning(f"Unable to read file {file_path} for {file_category}: {e}")
            print(f"Unable to read file {file_path} for {file_category}. Please check the path and file format.")
            all_valid = False
            continue

        # Check if the expected columns and data types match and if their are any null values
        expected_columns_dtypes_category = columns_dtypes.get(file_category, {})
        valid, errors = check_file(file_category,file_path, expected_columns_dtypes_category)
        if not valid:
            all_valid = False
            print(f"Error in {file_category}:")
            for error_message in errors:
                print(f"  - {error_message}")

    if all_valid:
        print("All files are in the expected format.")
    else:
        print("Validation failed for one or more files. Please check the errors.")

    #todo passing true instead of flag - update later
    # return all_valid
    return True

# COMMAND ----------

def check_file(file_category,file_path, expected_columns_dtypes):
    mismatches = []  # Collect mismatches
    try:
        # reading the datasets
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.upper()
        df.columns = df.columns.str.replace(' ', '_')
        
        #checking for missing columns
        for column, expected_type in expected_columns_dtypes.items():
            if column not in df.columns:
                logging.warning(f"Mismatches in file {file_category}: Expected column '{column}' not found")
                mismatches.append(f"File mismatches {file_category}: Expected column '{column}' not found,")
            else:
                actual_type = df[column].dtype
                if actual_type != expected_type:
                    try:
                        # Try converting the column to the expected data type
                        df[column] = df[column].astype(expected_type)
                    except Exception as conversion_error:
                        logging.error(f"Mismatches in file {file_category}: Data type mismatch in column '{column}'. "\
                                      f"Error during conversion: {conversion_error}")
                        mismatches.append(f"File mismatches {file_category}: Data type mismatch in column '{column}'. "
                                          f"Error during conversion: {conversion_error}")

        # Check presence of NULL values
        null_columns = df.columns[df.isnull().any()].tolist()
        if null_columns:
            logging.warning(f"Mismatches in file {file_category}: Presence of NULL values in columns - {null_columns}")
            mismatches.append(f"File mismatches {file_category}: Presence of NULL values in columns - {null_columns}")

        return not mismatches, mismatches  # Return True if no mismatches, along with the list of mismatches
    
    except Exception as e:
        logging.error(f"Error due to file {file_path}: {str(e)}",exc_info=True)
        return False, [f"Error in file: {e}"]

# COMMAND ----------

#Funtion to calculate trx
def calculate_optimized_trx(row,tps,beta):

    freq = row[tps]  # No of tps
    k =  row["PARAMETER1"] 
    a =  row["ADJ_FACTOR"]
    coef = row[beta] # Beta/response coefficient
    transformation = row['TRANSFORMATION'] 

    if transformation == "LOG":
        return np.log(1 + freq * k) * a * coef
    elif transformation == "POWER":
        return (freq ** k) * a * coef
    elif transformation == "LINEAR":
        return freq * coef
    elif transformation == "NEGEX":  # Negative Exponential
        return (1 - np.exp(-freq * k)) * a * coef
    else:
        # Transformation not supported
        error_msg = f"Unsupported transformation: {transformation}. Supported transformations are LOG, POWER, LINEAR, and NEGEX."
        logging.error(error_msg)
        raise ValueError(error_msg)

# COMMAND ----------

def process_files(config, allow_processing_with_errors=False):
    print("Processing files...")
    
    customer_data_path = config_file_paths.get('customer_data')
    constraint_segment_path = config_file_paths.get('constraint_segment')
    
    if customer_data_path:
       
        expected_columns_dtypes_cd = columns_dtypes.get('customer_data',{})
        expected_columns_dtypes_cons = columns_dtypes.get('constraint_segment',{})

        customer_data = pd.read_csv(customer_data_path)
        constraint_segment = pd.read_csv(constraint_segment_path)
 
        customer_data.columns = customer_data.columns.str.replace(' ','_')
        customer_data.columns = customer_data.columns.str.upper()
        
        customer_data = customer_data.apply(lambda x: x if x.dtype != 'O' else x.str.upper())
        customer_data = customer_data.apply(lambda x: x.replace(' ','_') if x.dtype == 'O' else x)
        customer_data = customer_data.drop_duplicates()
        
        customer_data['HCP_ID'] = customer_data['HCP_ID'].astype(str)
        customer_data["AFFINITY_SCORE"] = customer_data['AFFINITY_SCORE'].astype('float64')
       
        expected_columns_cons = list(expected_columns_dtypes_cons.keys())
        constraint_segment.columns = constraint_segment.columns.str.replace(' ','_')
        constraint_segment.columns = constraint_segment.columns.str.upper()
        constraint_segment = constraint_segment[expected_columns_cons]
        
        constraint_segment = constraint_segment.drop_duplicates()
        constraint_segment = constraint_segment.apply(lambda x: x if x.dtype != 'O' else x.str.upper())
        constraint_segment = constraint_segment.apply(lambda x: x.replace(' ','_') if x.dtype == 'O' else x)
      
       # to get only one segment for one channel
        for channel, group in constraint_segment.groupby('CHANNEL'):
            if len(group['CONSTRAINT_SEGMENT_NAME'].unique()) > 1:
                logging.warning(f"Multiple segments found for channel '{channel}': {', '.join(group['CONSTRAINT_SEGMENT_NAME'].unique())}")
                print(f"Warning: Multiple segments found for channel '{channel}': {', '.join(group['CONSTRAINT_SEGMENT_NAME'].unique())}")
        
        constraint_segment = constraint_segment.drop_duplicates(subset = ['CHANNEL']) 
        customer_data = customer_data.merge(constraint_segment,how='left',on='CHANNEL')
        customer_data['CONSTRAINT_SEGMENT_VALUE']  = customer_data.apply(lambda row: row[row['CONSTRAINT_SEGMENT_NAME']], axis=1)

        total_hcps = customer_data.groupby(['CHANNEL','CONSTRAINT_SEGMENT_NAME','CONSTRAINT_SEGMENT_VALUE'])['HCP_ID'].nunique().reset_index().rename(columns={'HCP_ID':'TOTAL_HCPS_SEGMENT'})
        total_hcps['TOTAL_HCPS_CHANNEL']  = total_hcps.groupby('CHANNEL')['TOTAL_HCPS_SEGMENT'].transform('sum')
        constraint_hcp_df = total_hcps.copy()

        file_list = ['mmix_output','asset_availability','vendor_contract','min_gap','engagement_goal','extra_constraint']
        
        merge_key_constraint = ['CHANNEL','CONSTRAINT_SEGMENT_NAME','CONSTRAINT_SEGMENT_VALUE']
        merge_key_datasets = ['CHANNEL','SEGMENT_NAME','SEGMENT_VALUE']

        for file_category in file_list:
            expected_columns_dtypes = columns_dtypes.get(file_category,{})
            file_path  = config_file_paths.get(file_category,'')
            try:
                if file_category.lower() != 'customer_data':
                    print(f'Processing {file_category}')
                    df = pd.read_csv(file_path)
                    
                    df.columns = df.columns.str.replace(' ', '_')
                    df.columns = df.columns.str.upper()
                    
                    df = df.drop_duplicates()
                    df = df.apply(lambda x: x if x.dtype != 'O' else x.str.upper()) #Converting all string elements to uppercase
                    df = df.apply(lambda x: x.replace(' ','_') if x.dtype == 'O' else x)
                    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                    
                    if file_category == 'mmix_output':
                        print(f'Creating {file_category} df')
                        mmix_output = df
                        mmix_output = mmix_output.drop_duplicates(subset=['CHANNEL','MMIX_SEGMENT_NAME','MMIX_SEGMENT_VALUE'])
                        mmix_output['OPTIMIZED_FREQUENCY'] = mmix_output.apply(lambda row: row['OPTIMIZED_FREQUENCY'] * granularity_mapping[row['GRANULARITY']], axis=1)
                        print(f'Created {file_category} df')
                    
                    elif file_category == 'asset_availability': 
                        print(f'Creating {file_category} df')
                        asset_availability = df
                        asset_availability  =  asset_availability.drop_duplicates()
                        asset_availability['ASSET_FREQUENCY'] = asset_availability.apply(lambda row: row['ASSET_FREQUENCY'] * granularity_mapping[row['GRANULARITY']], axis=1)
                        asset_tp = asset_availability.groupby(['CHANNEL','SEGMENT_NAME','SEGMENT_VALUE'])['ASSET_FREQUENCY'].sum().reset_index(name='ASSET_TP')

                        asset_tp = asset_tp.merge(total_hcps,how='left',left_on=['CHANNEL','SEGMENT_NAME','SEGMENT_VALUE'],right_on = ['CHANNEL','CONSTRAINT_SEGMENT_NAME','CONSTRAINT_SEGMENT_VALUE'])
                        asset_tp['MAX_TPS_SEGMENT'] = asset_tp['ASSET_TP']*asset_tp['TOTAL_HCPS_SEGMENT']
                        asset_tp['MAX_TPS_CHANNEL'] = asset_tp.groupby('CHANNEL')['MAX_TPS_SEGMENT'].transform(sum)
                        
                        print(f'Merging {file_category} to constraint_df')
                        constraint_hcp_df  = constraint_hcp_df.merge(asset_tp[['CHANNEL','MAX_TPS_CHANNEL']].drop_duplicates(),on='CHANNEL')
                        constraint_hcp_df = constraint_hcp_df.merge(asset_tp[['MAX_TPS_SEGMENT']+merge_key_datasets].drop_duplicates(),left_on = merge_key_constraint, right_on=merge_key_datasets)

                        print(f'Created {file_category} df')

                    elif file_category == 'vendor_contract': # take confirmation with rahul
                        print(f'Creating {file_category} df')
                        vendor_contract = df 
                        vendor_contract['MAX_TPS_CHANNEL']  = vendor_contract.apply(lambda row: row['MAX_VOLUME'] * granularity_mapping[row['GRANULARITY']], axis=1)
                        vendor_contract = vendor_contract.merge(total_hcps[['CHANNEL','TOTAL_HCPS_CHANNEL']].drop_duplicates(),how='left',on='CHANNEL')
                        vendor_contract  = vendor_contract.dropna(subset=['TOTAL_HCPS_CHANNEL'])
                        vendor_contract = vendor_contract[['CHANNEL','MAX_TPS_CHANNEL']]
                        print(f'Merging {file_category} to constraint_df')
                        constraint_hcp_df  = constraint_hcp_df.merge(vendor_contract[['CHANNEL','MAX_TPS_CHANNEL']].drop_duplicates(),on='CHANNEL')
                        
                        print(f'Created {file_category} df')
                        
                    elif file_category == 'min_gap':
                        print(f'Creating {file_category} df')
                        min_gap = df
                        min_gap  = min_gap[min_gap['CHANNEL_1']==min_gap['CHANNEL_2']]
                        min_gap = min_gap[min_gap['MIN_GAP']!=0]
                        min_gap['CHANNEL'] = min_gap['CHANNEL_1']
                        min_gap['MAX_TPS_SEGMENT'] = min_gap.apply(lambda row: (1/row['MIN_GAP']) * granularity_mapping[row['GRANULARITY']], axis=1)
                        min_gap['MAX_TPS_CHANNEL'] = min_gap.groupby('CHANNEL')['MAX_TPS_SEGMENT'].transform(sum)
                        min_gap = min_gap.merge(total_hcps,how='left',left_on=['CHANNEL','SEGMENT_NAME','SEGMENT_VALUE'],right_on = ['CHANNEL','CONSTRAINT_SEGMENT_NAME','CONSTRAINT_SEGMENT_VALUE'])
                        min_gap['MAX_TPS_SEGMENT'] = min_gap['MAX_TPS_SEGMENT']*min_gap['TOTAL_HCPS_SEGMENT']
                        min_gap['MAX_TPS_CHANNEL'] = min_gap.groupby('CHANNEL')['MAX_TPS_SEGMENT'].transform(sum)
                        min_gap = min_gap[['CHANNEL','SEGMENT_NAME','SEGMENT_VALUE','MAX_TPS_SEGMENT','MAX_TPS_CHANNEL']]
                        
                        print(f'Merging {file_category} to constraint_df')
                        constraint_hcp_df  = constraint_hcp_df.merge(min_gap[['CHANNEL','MAX_TPS_CHANNEL']].drop_duplicates(),on='CHANNEL')
                        constraint_hcp_df = constraint_hcp_df.merge(min_gap[['MAX_TPS_SEGMENT']+merge_key_datasets].drop_duplicates(),left_on = merge_key_constraint, right_on=merge_key_datasets)
                        
                        
                        print(f'Created {file_category} df')

                    
                    elif file_category == 'engagement_goal':
                        print(f'Creating {file_category} df')
                        engagement_goal = df
                        engagement_goal['TARGET'] = engagement_goal.apply(lambda row: row['TARGET'] * granularity_mapping[row['GRANULARITY']], axis=1)
                        
                        invalid_types = engagement_goal[~engagement_goal['TYPE'].isin(['ENGAGEMENT', 'EXPOSURE'])]
                        if not invalid_types.empty:
                            for _, row in invalid_types.iterrows():
                                channel = row['CHANNEL']
                                segment = row['SEGMENT_VALUE']
                                type_value = row['TYPE']
                                print(f"Warning: Unexpected TYPE '{type_value}' for CHANNEL '{channel}' and SEGMENT '{segment}'.")
                                logging.warning(f"Unexpected TYPE '{type_value}' for CHANNEL '{channel}' and SEGMENT '{segment}'.")

                        engagement_goal['MIN_TPS_SEGMENT'] = np.where(engagement_goal['TYPE']=='ENGAGEMENT', engagement_goal['TARGET']/engagement_goal['ENGAGEMENT_RATE'],engagement_goal['TARGET']) 
                        engagement_goal  = engagement_goal.merge(total_hcps,how='left',left_on=['CHANNEL','SEGMENT_NAME','SEGMENT_VALUE'],right_on = ['CHANNEL','CONSTRAINT_SEGMENT_NAME','CONSTRAINT_SEGMENT_VALUE'])
                        engagement_goal['MIN_TPS_CHANNEL'] = engagement_goal.groupby('CHANNEL')['MIN_TPS_SEGMENT'].transform(sum)
                        #subset on req cols
                        eng_goal_cols = ['CHANNEL','SEGMENT_NAME','SEGMENT_VALUE','MIN_TPS_SEGMENT','MIN_TPS_CHANNEL']
                        engagement_goal = engagement_goal[eng_goal_cols]
                        
                        print(f'Merging {file_category} to constraint_df')
                        constraint_hcp_df  = constraint_hcp_df.merge(engagement_goal[['CHANNEL','MIN_TPS_CHANNEL']].drop_duplicates(),on='CHANNEL')
                        constraint_hcp_df = constraint_hcp_df.merge(engagement_goal[['MIN_TPS_SEGMENT']+merge_key_datasets].drop_duplicates(),left_on = merge_key_constraint, right_on=merge_key_datasets)

                        print(f'Created {file_category} df')
                    
                    elif file_category == 'extra_constraint':
                         extra_constraint = df
                         extra_constraint = extra_constraint.drop_duplicates(subset=['CHANNEL','SEGMENT_NAME','SEGMENT_VALUE'])
                         extra_constraint['MIN_TPS'] = extra_constraint.apply(lambda row: row['MIN_TPS'] * granularity_mapping[row['GRANULARITY']], axis=1)
                         extra_constraint['MAX_TPS'] = extra_constraint.apply(lambda row: row['MAX_TPS'] * granularity_mapping[row['GRANULARITY']], axis=1)
                         extra_constraint = extra_constraint.merge(total_hcps,how='left',left_on=['CHANNEL','SEGMENT_NAME','SEGMENT_VALUE'],right_on = ['CHANNEL','CONSTRAINT_SEGMENT_NAME','CONSTRAINT_SEGMENT_VALUE'])
                         extra_constraint['MIN_TPS_SEGMENT'] = extra_constraint['MIN_TPS_SEGMENT']*extra_constraint['TOTAL_HCPS_SEGMENT']
                         extra_constraint['MIN_TPS_CHANNEL'] = extra_constraint.groupby('CHANNEL')['MIN_TPS_SEGMENT'].transform(sum)
                         extra_constraint['MAX_TPS_SEGMENT'] = extra_constraint['MAX_TPS_SEGMENT']*extra_constraint['TOTAL_HCPS_SEGMENT']
                         extra_constraint['MAX_TPS_CHANNEL'] = extra_constraint.groupby('CHANNEL')['MAX_TPS_SEGMENT'].transform(sum)
                         print(f'Merging {file_category} to constraint_df')
                         constraint_hcp_df  = constraint_hcp_df.merge(extra_constraint[['CHANNEL','MIN_TPS_CHANNEL','MAX_TPS_CHANNEL']].drop_duplicates(),on='CHANNEL')
                         constraint_hcp_df = constraint_hcp_df.merge(engagement_goal[['MIN_TPS_SEGMENT','MAX_TPS_SEGMENT']+merge_key_datasets].drop_duplicates(),left_on = merge_key_constraint, right_on=merge_key_datasets)
            
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

def load_config(config_path):
    try:
        with open(config_path, 'r') as config_file:
            config_data = yaml.safe_load(config_file)

        global config_file_paths, config_controls, config_run_period, output_folder
        
        config_file_paths = config_data.get('file_paths', {})
        config_controls = config_data.get('control_parameters', {})
        config_run_period = config_data.get('run_period', {})
        
        output = config_data.get('output',{})
        run_id = output.get('run_id','')
        output_path = output.get('output_folder_path', '')
        output_path = output_path.replace('//', '/')
        output_path = os.path.join(*output_path.split('/'))
        output_folder = os.path.join(output_path, run_id)
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder,'Final_Plan'), exist_ok=True)
        os.makedirs(os.path.join(output_folder,'Customer_Segmentation_Charts'), exist_ok=True)
        os.makedirs(os.path.join(output_folder,'Uplift'), exist_ok=True)
    
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        print(f"Error loading configuration from {config_path}. Please check the logs for details.")
        sys.exit(1)
    
    return True

# COMMAND ----------

def select_option(options):
    for option, selection in options.items():
        if selection.upper() == 'Y':
            return option
    # Use the first option if no 'Y' is found
    return next(iter(options), None)

# COMMAND ----------

def tp_allocation(df_final_gekko):
    
    columns = df_final_gekko.columns.tolist()
    final_df = pd.DataFrame(columns=columns)
    
    chan_list = df_final_gekko['CHANNEL'].unique().tolist()
    seg_list = df_final_gekko['CONSTRAINT_SEGMENT_VALUE'].unique().tolist()

    for i in chan_list:
        for j in seg_list:
            df_rest = df_final_gekko[(df_final_gekko['CHANNEL'] == i) & (df_final_gekko['CONSTRAINT_SEGMENT_VALUE'] == j)].reset_index(drop=True)

            count = df_rest["OPTIMIZED_FREQUENCY"].sum()
            max_count = 0
            Gekko_sum = df_rest["GEKKO_MAX_TP"].sum()
            Max_sum = df_rest["FINAL_MAX_TOUCHPOINTS_SEGMENT"].sum()

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

            final_df = pd.concat([final_df, df_rest]).reset_index(drop=True)
    

    recommendations = final_df[['HCP_ID','CHANNEL','CONSTRAINT_SEGMENT_NAME','CONSTRAINT_SEGMENT_VALUE','FINAL_TOUCHPOINT','OPTIMIZED_FREQUENCY','RESPONSE_COEFFICIENT','NEW_BETA','PARAMETER1','ADJ_FACTOR','TRANSFORMATION','AFFINITY_SCORE','AFFINITY_SEGMENT']]
    recommendations.rename(columns={'CONSTRAINT_SEGMENT_NAME':'SEGMENT_NAME','CONSTRAINT_SEGMENT_VALUE':'SEGMENT_VALUE','FINAL_TOUCHPOINT':'FINAL_QTRLY_TOUCHPOINT'},inplace=True)

    return recommendations

# COMMAND ----------

def uplift_calc(final_df):
    
    final_df["MMO_TRX"] = final_df.apply(lambda row: calculate_optimized_trx(row, tps = 'OPTIMIZED_FREQUENCY', beta = 'RESPONSE_COEFFICIENT'), axis=1)
    final_df["OPTIMIZED_TRX"] = final_df.apply(lambda row: calculate_optimized_trx(row, tps = 'FINAL_QTRLY_TOUCHPOINT', beta = 'NEW_BETA'), axis=1)
    
    Optimized_trx  = final_df[final_df['AFFINITY_SCORE']>0]['OPTIMIZED_TRX'].sum()
    MMO_Trx = final_df[final_df['AFFINITY_SCORE']>0]['MMO_TRX'].sum()
    d = (Optimized_trx/MMO_Trx) - 1
    d = d * 100
    overall_lift = d
    
    overall_lift3 = overall_lift * 0.6
    overall_lift4 = overall_lift * 0.8

    uplift_data = np.array([[overall_lift / 100, overall_lift * 0.8 / 100, overall_lift * 0.6 / 100]])
    ipp_uplift = pd.DataFrame(uplift_data, columns=["Total Uplift", "Assuming 80% Execution", "Assuming 60% Execution"])
    ipp_uplift = ipp_uplift.round(decimals=4)

    ipp_uplift.to_csv(f"{output_folder}/IPP_Uplift.csv", index=False)
    final_df.to_csv(f'{output_folder}/final_output_with_opti_trx.csv',index=False)

    print(f"Overall_Uplift : {round(overall_lift,2)}%\n",
          f"Uplift 80% execution : {round(overall_lift4,2)}%\n",\
            f"Uplift 60% execution: {round(overall_lift3,2)}%\n")
    
    # for visualization

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
    uplift_output_path = os.path.join(output_folder,'Uplift','Uplift.png')
    plt.savefig(uplift_output_path, bbox_inches="tight")

    return None

# COMMAND ----------

def quarterly_to_monthly(recommendations):
    
    dist_rule_monthly = select_option(config_controls.get('dist_rule_monthly', {}))
    monthly_dist_split  = config_controls.get('monthly_dist_split','')

    refresh_date = config_run_period.get('refresh_date')
    refresh_date = pd.to_datetime(refresh_date)
    last_2_months = refresh_date - timedelta(days=60)
    last_2_months_str = last_2_months.strftime('%Y-%m-%d')
    
    historical_data_path = config_file_paths.get('historical_data','')
    
    global historical_data
    historical_data = pd.read_csv(historical_data_path)
    historical_data = historical_data.drop_duplicates()
    historical_data.columns = historical_data.columns.str.upper()
    historical_data.rename(columns={'CHANNEL_ID':'CHANNEL'},inplace=True)

    historical_data['EXPOSED_ON'] = pd.to_datetime(historical_data['EXPOSED_ON'])
    historical_data = historical_data.apply(lambda x: x.str.upper() if x.dtype == 'O' else x)
    historical_data = historical_data.apply(lambda x: x.str.replace(' ','_') if x.dtype == 'O' else x)

    historical_data['EXPOSED_ON'] = historical_data['EXPOSED_ON'].dt.strftime("%Y-%m-%d")
    historical_data['HCP_ID'] = historical_data['HCP_ID'].astype(str)
    last_2_mth_exp  = historical_data[historical_data['EXPOSED_ON']>=last_2_months_str]
    last_2_mth_tps = last_2_mth_exp.groupby(['HCP_ID', 'CHANNEL']).size().reset_index(name='last_2_mth_tps')
    
    recommendations = recommendations.merge(last_2_mth_tps.drop_duplicates(),how='left',on=['HCP_ID','CHANNEL'])
    recommendations['last_2_mth_tps'] = recommendations['last_2_mth_tps'].fillna(0)
    
    recommendations['Monthly_tps_eql_dist'] = np.floor(recommendations['FINAL_QTRLY_TOUCHPOINT']/3)
    recommendations['Monthly_tps_diff_lst_2_mth'] = recommendations['FINAL_QTRLY_TOUCHPOINT'] - recommendations['last_2_mth_tps']
    
    recommendations['Monthly_tps_diff_lst_2_mth'] = np.where(recommendations['Monthly_tps_diff_lst_2_mth']<0,0,recommendations['Monthly_tps_diff_lst_2_mth'])

    if dist_rule_monthly == 'EQUAL':
        recommendations['FINAL_MONTHLY_TOUCHPOINTS'] = recommendations['Monthly_tps_eql_dist']
    if dist_rule_monthly == 'PREDEFINED_SPLIT':
        if type(monthly_dist_split) is float and monthly_dist_split>0:
            recommendations['FINAL_MONTHLY_TOUCHPOINTS'] = recommendations['FINAL_QTRLY_TOUCHPOINT']*monthly_dist_split
        else:
            print('Monthly split not in expected format. Please provide a split between 0 to 1')
            logging.error('Monthly Split not in required format.')
            raise TypeError('Monthly split out of range')

    if dist_rule_monthly == 'HISTORICAL':
        recommendations['FINAL_MONTHLY_TOUCHPOINTS'] = recommendations['Monthly_tps_diff_lst_2_mth']
    
    recommendations = recommendations[['HCP_ID','CHANNEL','SEGMENT_NAME','SEGMENT_VALUE','FINAL_MONTHLY_TOUCHPOINTS','FINAL_QTRLY_TOUCHPOINT']]
    
    return recommendations

# COMMAND ----------

def weekly_tps(recommendations,tps_column=None,num_weeks=None):
    
    dist_rule_weekly = select_option(config_controls.get('dist_rule_weekly', {}))
    
    num_weeks =  4 # Assuming 4 weeks in a month
    tps_column = 'FINAL_MONTHLY_TOUCHPOINTS' 

    # makign columns for each week
    weekly_tps_cols = [f'WEEK_{i+1}_TOTAL_TPS' for i in range(num_weeks)]
    
    recommendations_hcp = recommendations.groupby('HCP_ID')[tps_column].sum().reset_index(name='FINAL_MONTHLY_TOUCHPOINTS_TOTAL')
    recommendations_hcp[weekly_tps_cols] = 0
    if dist_rule_weekly == 'AGGRESSIVE':
        try:
            weekly_max_tp = config_controls.get('weekly_max_tp_aggresive')
            weekly_max_tp = weekly_max_tp.astype(int)
        except:
            print('Weekly Max TP not provided in correct format for aggresive approach. Considering all monthly tps available to be sent in one week')
            logging.warning('Weekly Max TP not provided in correct format for aggresive approach')
    # print("Proceeding further")
    for tp in recommendations_hcp['FINAL_MONTHLY_TOUCHPOINTS_TOTAL'].unique():
        # print("run loop....")
        tp = int(tp)
        if dist_rule_weekly == 'EQUAL':
            if tp == num_weeks:
                weekly_tps = [1] * num_weeks
            
            elif tp<num_weeks:
                weekly_tps = [1] * tp + [0] * (num_weeks - tp) # distribute w equal gap
            
            elif tp > num_weeks:
                week_tps = (tp // num_weeks) 
                
                remaining_tps = (tp % num_weeks)
                
                weekly_tps = [0]*num_weeks
 
                for i in range (0,num_weeks,2):
                    if remaining_tps>0:
                        weekly_tps[i] = 1
                    remaining_tps -= 1
                for i in range (1,num_weeks,2):
                    if remaining_tps>0:
                        weekly_tps[i]  = 1
                    remaining_tps -= 1

                weekly_tps = [tps + week_tps for tps in weekly_tps]

        if dist_rule_weekly == 'AGGRESSIVE':
            try:
                if tp > weekly_max_tp:
                       tp = weekly_max_tp
            except:
                  pass
            weekly_tps = [tp]*num_weeks

        recommendations_hcp.loc[recommendations_hcp['FINAL_MONTHLY_TOUCHPOINTS_TOTAL'] == tp, weekly_tps_cols] = weekly_tps

    recommendations_hcp  = recommendations.merge(recommendations_hcp,how='left',on='HCP_ID')
    
    return recommendations_hcp

# COMMAND ----------

def weekly_seq(recommendations_hcp,historical_data):
    
    dist_rule_weekly = select_option(config_controls.get('dist_rule_weekly', {}))
    
    priority_file_path = config_file_paths.get('priority','')
    
    if priority_file_path is None:
        try:
            priority_file = model_input(config_file_paths)
        except Exception as e:
            logging.error(f"Error creating priority file using model: {e}")
            sys.exit(1)
    else:
        try:
            priority_file = pd.read_csv(priority_file_path)
            priority_file.columns = priority_file.columns.str.upper()
            priority_file['HCP_ID'] = priority_file['HCP_ID'].astype(str)
            priority_file = priority_file.apply(lambda x: x.str.upper() if x.dtype=='O' else x)
            priority_file = priority_file.apply(lambda x: x.str.replace(' ','_') if x.dtype=='O' else x)
        
        except Exception as e:
            logging.error(f"Error loading priority file, Please check the path : {e}")
            sys.exit(1)
    
    start_date = config_run_period.get('start_date','')
    refresh_date = config_run_period.get('refresh_date','')

    start_date = pd.to_datetime(start_date)
    refresh_date = pd.to_datetime(refresh_date)

    recommendations_hcp = recommendations_hcp.merge(priority_file,how='left',on=['HCP_ID','CHANNEL'])
    recommendations_hcp = recommendations_hcp.sort_values(['HCP_ID','PRIORITY','CHANNEL'])
    
    recommendations_hcp['PRIORITY'] =  recommendations_hcp.groupby('HCP_ID').cumcount() + 1
    
    recommendations_hcp['NOT_AVAILABLE'] = 0
    recommendations_hcp['FINAL_MONTHLY_TOUCHPOINTS_REMAINING'] = recommendations_hcp['FINAL_MONTHLY_TOUCHPOINTS'].copy()
    
    min_gap_path = config_file_paths.get('min_gap','')
    min_gap = pd.read_csv(min_gap_path)
    min_gap.columns = min_gap.columns.str.upper()
    min_gap = min_gap.apply(lambda x: x.str.upper() if x.dtype=='O' else x)
    min_gap = min_gap.apply(lambda x: x.str.replace(' ','_') if x.dtype=='O' else x)
    # Removing the rows where min_gap is less than 1 for cross channel
    min_gap = min_gap[(min_gap['CHANNEL_1'] == min_gap['CHANNEL_2']) | (min_gap['MIN_GAP'] >= 1)]
    min_gap = min_gap[min_gap['MIN_GAP']>0]
    
    historical_data['EXPOSED_ON'] = pd.to_datetime(historical_data['EXPOSED_ON'])
    week_list = ['WEEK_1','WEEK_2','WEEK_3','WEEK_4']

    if start_date != refresh_date:
       day_difference = (refresh_date - start_date).days
       week = min(4, int((day_difference // 7.1)) + 2)
       week_list = [f'WEEK_{i}' for i in range(week,5)]
    
    weekly_channel_tps_cols = [week + '_CHANNEL_TPS' for week in week_list]
    recommendations_hcp[weekly_channel_tps_cols] = 0

    for grp_name,grp in recommendations_hcp.groupby('HCP_ID'):
        historical_exp = historical_data[historical_data['HCP_ID']==grp_name]
        total_tps = 0
        if dist_rule_weekly == 'AGGRESSIVE':
            total_tps = grp['WEEK_1_TOTAL_TPS'].values[0]
        for week in week_list:
            if dist_rule_weekly == 'EQUAL':
                total_tps  = grp[week + '_' + 'TOTAL_TPS'].values[0] + total_tps 
            channels = set(grp['CHANNEL'].unique())

            not_available_channels = set()
            break_loop =False
            
            while total_tps > 0 and not channels.issubset(not_available_channels) and not break_loop:
                for idx, row_hcp in grp.iterrows():
                    if total_tps <= 0 or channels.issubset(not_available_channels):
                        break_loop = True
                        break
                    channel = recommendations_hcp.at[idx,'CHANNEL'] 
                    channel_monthly_tps = recommendations_hcp.at[idx,'FINAL_MONTHLY_TOUCHPOINTS_REMAINING']
                    available_channels = set()

                    if channel in not_available_channels or channel_monthly_tps<=0:
                        recommendations_hcp.loc[idx, 'NOT_AVAILABLE'] = 1
                        not_available_channels.add(channel)
                        continue
                
                    # for selecting min gap constraint applicable to that hcp on basis of his segment
                    min_gap_subset  = min_gap[(min_gap['CHANNEL_1'] == channel) | (min_gap['CHANNEL_2'] == channel)]
                    if not min_gap_subset.empty:
                        hcp_segment  = row_hcp['SEGMENT_VALUE']
                        min_gap_req = min_gap_subset[min_gap_subset['SEGMENT_VALUE']==hcp_segment]
                        if not min_gap_req.empty:
                            for i,row_min_gap in min_gap_req.iterrows():
                                channel1 = row_min_gap['CHANNEL_1'] 
                                channel2 = row_min_gap['CHANNEL_2']
                                min_gap_value = row_min_gap['MIN_GAP'] 

                                if min_gap_value >=1:
                                    lookback_period = refresh_date + timedelta(week_list.index(week)*7) - timedelta(days=(min_gap_value-1)*7)  # defininig the lookback period on basis of refrsh date and min gap
                                    future_period = refresh_date + timedelta(week_list.index(week)*7) + timedelta(days=(min_gap_value)*7)
                                if min_gap_value <1 and min_gap_value >0:
                                    lookback_period = refresh_date + timedelta(week_list.index(week)*7)
                                    future_period = refresh_date + timedelta(week_list.index(week)*7) + timedelta(days=7)
                                
                                channel1_data = historical_exp[(historical_exp['CHANNEL'] == channel1) & (historical_exp['EXPOSED_ON'] >= lookback_period) & (historical_exp['EXPOSED_ON'] <= future_period)]
                                channel2_data = historical_exp[(historical_exp['CHANNEL'] == channel2) & (historical_exp['EXPOSED_ON'] >= lookback_period) & (historical_exp['EXPOSED_ON'] <= future_period)]
                                
                                if channel1 == channel2:
                                   if len(channel1_data) >= 1/min_gap_value:
                                       not_available_channels.add(channel1)
                                       
                                if channel1 != channel2:     
                                    if not channel1_data.empty: # If channel 1 data is not empty add channel2 to not_available_channel
                                        not_available_channels.add(channel2)
                                        available_channels.add(channel2)
                                        # If channel 2 data is not empty, add channel1 to not_available_channels
                                    if not channel2_data.empty:
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
                        recommendations_hcp.loc[idx,'FINAL_MONTHLY_TOUCHPOINTS_REMAINING'] -= 1
                        channel_monthly_tps -=1
                        new_tp  = {'HCP_ID':grp_name,'CHANNEL':channel,'EXPOSED_ON':refresh_date + timedelta(week_list.index(week)*7)}
                        historical_exp = pd.concat([historical_exp, pd.DataFrame([new_tp])], ignore_index=True)
    
    req_col = ['HCP_ID', 'CHANNEL', 'SEGMENT_NAME', 'SEGMENT_VALUE','FINAL_MONTHLY_TOUCHPOINTS', 'FINAL_QTRLY_TOUCHPOINT','PRIORITY'] + recommendations_hcp.filter(like='CHANNEL_TPS').columns.tolist()               
    recommendations_final = recommendations_hcp[req_col]
    return recommendations_final

# COMMAND ----------

def model_input(config_file_paths):
    
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

def create_subsets(df_final):
    columns = df_final.columns.tolist()
    train_df = pd.DataFrame(columns=columns)

    chan_list = df_final['CHANNEL'].unique().tolist()
    seg_list = df_final['CONSTRAINT_SEGMENT_VALUE'].unique().tolist()

    for i in chan_list:
        for j in seg_list:
            df_temp = df_final[(df_final['CHANNEL'] == i) & (df_final['CONSTRAINT_SEGMENT_VALUE'] == j)].reset_index(drop=True)
            
            if df_temp['HCP_ID'].count() == 0:
                continue
            
            if df_temp['HCP_ID'].count() <= 500:
                train = df_temp
            else:
                single_value = df_temp[~df_temp['AFFINITY_SCORE'].duplicated(keep=False)]
                df_temp = df_temp[df_temp['AFFINITY_SCORE'].duplicated(keep=False)]
                n = 500 / df_temp['HCP_ID'].count()
                train, test = train_test_split(df_temp, test_size=1 - n, random_state=0, stratify=df_temp[['AFFINITY_SCORE']])
            
            train = train.reset_index(drop=True)
            train = train.sort_values(["AFFINITY_SCORE"], ascending=[False])

            train_df = pd.concat([train_df, train, single_value]).reset_index(drop=True)

    return train_df

# COMMAND ----------

def run_gekko_optimization(train_df):
    columns = train_df.columns.tolist()
    df_for_algo = pd.DataFrame(columns=columns)

    chan_list = train_df['CHANNEL'].unique().tolist()
    seg_list = train_df['CONSTRAINT_SEGMENT_VALUE'].unique().tolist()

    for i in chan_list:
        for j in seg_list:
            gekko_db = train_df[(train_df['CHANNEL'] == i) & (train_df['CONSTRAINT_SEGMENT_VALUE'] == j)].reset_index(drop=True)

            if gekko_db['HCP_ID'].count() == 0:
                continue

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

            x = [tp[i][0] for i in range(n)]
            x = np.array(x)
            gekko_db['GEKKO_MAX_TP'] = x

            gekko_db["GEKKO_MAX_TP"] = gekko_db["GEKKO_MAX_TP"].round(decimals=0)
            df_for_algo = pd.concat([gekko_db, df_for_algo]).reset_index(drop=True)

    return df_for_algo

# COMMAND ----------

def final_merge(df_final, algo_df):
    algo_df = algo_df.groupby(['CHANNEL', 'CONSTRAINT_SEGMENT_VALUE', 'AFFINITY_SCORE'])["GEKKO_MAX_TP"].max().reset_index()
    algo_df = algo_df.reset_index(drop=True)
    df_final_gekko = pd.merge(df_final, algo_df, how='right', on=['CHANNEL', 'CONSTRAINT_SEGMENT_VALUE', 'AFFINITY_SCORE'])
    df_final_gekko = df_final_gekko.sort_values(['CHANNEL', 'CONSTRAINT_SEGMENT_VALUE', 'AFFINITY_SCORE'], ascending=[True, True, False]).reset_index(drop=True)
    df_final_gekko["FINAL_TOUCHPOINT"] = 0
    
    return df_final_gekko

# COMMAND ----------

def run_constraint_module(config_path):
    # try:
    load_config(config_path) # Load configuration from the provided config file
    validation_result = validate_files(config_file_paths)  # validate files result
    # Check if the script is running in Databricks environment
    if 'databricks' in sys.modules:
        if validation_result:
            print('Processing files for optimization module')
            df_final = process_files(config_file_paths, allow_processing_with_errors=True) #process files
            if df_final is not None:
                print('Running Optimization Model in databrics environment')
                # run optimization model
            else:
                print('Errors in Processing Files. Check Logs for more details')    

        else:
            print("Validation failed. Please check the logs for details.")
            sys.exit(1)
    #todo undo else indentation
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
                sys.exit(1)
            else:
                monthly_plan  = quarterly_to_monthly(final_quarterly_plan) 
                if final_output_level == 'MONTHLY':
                    print('Monthly HCP Promotional Plan Created')
                    output_path = os.path.join(output_folder,'Final_Plan','HCP_Monthly_Plan.csv')
                    monthly_plan.to_csv(output_path,index=False)
                    sys.exit(1)
                else:
                    print('Creating Weekly HCP Promotional Plan')
                    weekly_plan = weekly_tps(monthly_plan)
                    final_weekly_seq  = weekly_seq(weekly_plan,historical_data)
                    output_path = os.path.join(output_folder,'Final_Plan','HCP_Weekly_Plan.csv')
                    final_weekly_seq.to_csv(output_path,index=False)
                    sys.exit(1)    
        else:
            print('Errors in Processing Files. Check Logs for more details')
            sys.exit(1)
    else:
        print("Processing aborted.Validation failed.Please check the logs for details.")
        sys.exit(1)
    return None

# COMMAND ----------

# if __name__ == "__main__":
def model_start():
    # config_path = sys.argv[1]
    config_path = "config.yaml"
    run_constraint_module(config_path)