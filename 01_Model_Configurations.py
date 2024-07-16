# Databricks notebook source
# MAGIC %md
# MAGIC # Adhere to the following set of rules
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variable Listing Section. Use the following set of variables as per the format
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### nba_start: 
# MAGIC #### Start date for the script execution, date on which script is run for the first time in a month
# MAGIC ##### Example: '2024-12-02'

# COMMAND ----------

# nba_start = 'datetime.now().strftime("%Y-%m-%d")'
nba_start = '2024-05-02'

# COMMAND ----------

# MAGIC %md
# MAGIC ### nba_end: 
# MAGIC #### End date after one month of start date for the script execution, date till which the data is collected
# MAGIC ##### Example: '2024-12-31'

# COMMAND ----------

# nba_end = (datetime.strptime(nba_start, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")
nba_end = '2024-12-31'

# COMMAND ----------

# MAGIC %md
# MAGIC ### nba_refresh: 
# MAGIC #### Refresh date for the script execution, dates on which the script is re-run in the same month after start date
# MAGIC ##### Use the following format for passing the value: "YYYY-MM-DD"
# MAGIC ##### Example: '2024-12-17'

# COMMAND ----------

nba_refresh = '2024-06-01'

# COMMAND ----------

# MAGIC %md
# MAGIC ### refresh_cadence: 
# MAGIC #### This is used to set the NBA refresh cadence.
# MAGIC ##### Set value from the options as per requirement: 'QUARTERLY', 'MONTHLY', 'WEEKLY'

# COMMAND ----------

refresh_cadence = 'QUARTERLY'

# COMMAND ----------

# MAGIC %md
# MAGIC ### weekly_tps: 
# MAGIC #### Defines the distribution rule for weekly touchpoints.
# MAGIC ##### Set value from the options as per requirement: 'EQUAL' or 'AGGRESSIVE'.
# MAGIC ###### Applicable only if "WEEKLY" is selected as an option in refresh_cadence.

# COMMAND ----------

weekly_tps = 'EQUAL'

# COMMAND ----------

# MAGIC %md
# MAGIC ### max_tps: 
# MAGIC #### Defines the weekly maximum touchpoints (TPS) for aggressive planning.
# MAGIC ##### Set value from the options as per requirement: NULL, 1, 2 etc. Only provide an integer value
# MAGIC ###### Olny applicable if weekly_tps is selected as "AGGRESSIVE".

# COMMAND ----------

max_tps = 1

# COMMAND ----------

# MAGIC %md
# MAGIC ### monthly_tps: 
# MAGIC #### Defines the distribution rule for monthly planning.
# MAGIC ##### Set value from the options as per requirement: 'EQUAL', 'PREDEFINED_SPLIT', 'HISTORICAL'
# MAGIC ###### Only applicable if "MONTHLY" is selected as option in refresh_cadence.

# COMMAND ----------

monthly_tps = 'PREDEFINED_SPLIT'

# COMMAND ----------

# MAGIC %md
# MAGIC ####asset_availability:
# MAGIC #####Finds the asset availability file path

# COMMAND ----------

asset_availability= './Data_Files/Asset_Availability.csv'

# COMMAND ----------

# MAGIC %md
# MAGIC #####constraint_segment:
# MAGIC #####Finds the constraint segment file path

# COMMAND ----------

constraint_segment= './Data_Files/Constraint_segment.csv'

# COMMAND ----------

# MAGIC %md
# MAGIC #####customer_data
# MAGIC #####Finds the customer data file path

# COMMAND ----------

customer_data='./Data_Files/Sample_HCP_Master_Demo.csv'

# COMMAND ----------

# MAGIC %md
# MAGIC #####engagement_goal
# MAGIC #####Finds the engagement goal file path

# COMMAND ----------

engagement_goal='./Data_Files/Eng_Target_File_Demo.csv'

# COMMAND ----------

# MAGIC %md
# MAGIC #####historical_data
# MAGIC #####Finds the file path for historical data

# COMMAND ----------

historical_data='./Data_Files/HCP_Historical_Data_Demo.csv'

# COMMAND ----------

# MAGIC %md
# MAGIC #####min_gap
# MAGIC #####Finds the file path for minimum gap constaints

# COMMAND ----------

min_gap='./Data_Files/Min_Gap_Constraints - Copy.csv'

# COMMAND ----------

# MAGIC %md
# MAGIC #####mmix_output
# MAGIC #####Finds the file path for Marketing Mix Output

# COMMAND ----------

mmix_output='./Data_Files/Marketing_Mix_Input_Demo.csv'

# COMMAND ----------

# MAGIC %md
# MAGIC #####priority
# MAGIC #####Finds the file path for HCP channel priority

# COMMAND ----------

priority='./Data_Files/HCP_Channel_Priority.csv'

# COMMAND ----------

# MAGIC %md
# MAGIC #####vendor_contract
# MAGIC #####Finds the file path for vendor contract

# COMMAND ----------

vendor_contract='./Data_Files/Vendor_Contract.csv'

# COMMAND ----------

# MAGIC %md
# MAGIC #####output_folder_path
# MAGIC #####Finds the path for module output

# COMMAND ----------

output_folder_path='./Outputs'

# COMMAND ----------

# MAGIC %md
# MAGIC ####run_id
# MAGIC #####A counter variable which defines the sequence of run 

# COMMAND ----------

# run_id='M1'
run_id = ''

# COMMAND ----------

# MAGIC %md
# MAGIC #### config_update: 
# MAGIC ##### The function used to update the 'config.yaml' file, which contains all the parametrs required for model execution.

# COMMAND ----------

import yaml

def config_update(nba_start, nba_end, nba_refresh, refresh_cadence, weekly_tps, max_tps, monthly_tps,
                  asset_availability, constraint_segment, customer_data, engagement_goal,
                  historical_data, min_gap, mmix_output, priority, vendor_contract,
                  output_folder_path, run_id):
    # Load configuration from the YAML file
    with open('./Archive/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Update run period information
    config['run_period']['start_date'] = nba_start
    config['run_period']['end_date'] = nba_end
    config['run_period']['refresh_date'] = nba_refresh

    # Update weekly touchpoint distribution rule
    config['control_parameters']['dist_rule_weekly']['EQUAL'] = 'Y' if weekly_tps == 'EQUAL' else 'N'
    config['control_parameters']['dist_rule_weekly']['AGGRESSIVE'] = 'Y' if weekly_tps != 'EQUAL' else 'N'
    config['control_parameters']['weekly_max_tp_aggresive'] = max_tps if weekly_tps != 'EQUAL' else None

    # Update monthly touchpoint distribution rule
    config['control_parameters']['dist_rule_monthly']['EQUAL'] = 'Y' if monthly_tps == 'EQUAL' else 'N'
    config['control_parameters']['dist_rule_monthly']['PREDEFINED_SPLIT'] = 'Y' if monthly_tps == 'PREDEFINED_SPLIT' else 'N'
    config['control_parameters']['dist_rule_monthly']['HISTORICAL'] = 'Y' if monthly_tps == 'HISTORICAL' else 'N'

    # Update final output level
    config['control_parameters']['final_output_level']['QUARTERLY'] = 'Y' if refresh_cadence == 'QUARTERLY' else 'N'
    config['control_parameters']['final_output_level']['MONTHLY'] = 'Y' if refresh_cadence == 'MONTHLY' else 'N'
    config['control_parameters']['final_output_level']['WEEKLY'] = 'Y' if refresh_cadence == 'WEEKLY' else 'N'

    # Dictionary to store file paths
    file_paths = {
        'asset_availability': asset_availability,
        'constraint_segment': constraint_segment,
        'customer_data': customer_data,
        'engagement_goal': engagement_goal,
        'historical_data': historical_data,
        'min_gap': min_gap,
        'mmix_output': mmix_output,
        'priority': priority,
        'vendor_contract': vendor_contract
    }

    # Update file paths
    for key, value in file_paths.items():
        config['file_paths'][key] = value

    # Update output folder path and run ID
    config['output']['output_folder_path'] = output_folder_path
    config['output']['run_id'] = run_id

    # Write the updated configuration back to the YAML file
    with open("./Archive/config.yaml", 'w') as file:
        yaml.dump(config, file)

# COMMAND ----------

# MAGIC %md
# MAGIC #### After defining all the required parameters, call the function 'config_update' passing the required parameters.

# COMMAND ----------

config_update(nba_start, nba_end, nba_refresh, refresh_cadence, weekly_tps, max_tps, monthly_tps, asset_availability, constraint_segment, customer_data, engagement_goal, historical_data, min_gap, mmix_output, priority, vendor_contract, output_folder_path, run_id)

# COMMAND ----------

output_folder_path

# COMMAND ----------


