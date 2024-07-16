# NBA - Omnichannel Prediction Model
The solution here aims to use AI/ML models to enhance NBA planning efficiency and effectiveness by leveraging machine learning techniques and comprehensive data analysis, and provide curated recommendation at a monthly/weekly level based on the types of constraints set.

## Solution Overview
Here we have create a ML model, which generates NBA predictions based on the input data and user input contraints from GUI.

1. Quarterly Guardrails: Establishes guardrails at the quarter level, encompassing engagement goals, vendor contracts, and other constraints to guide NBA planning.
2. Budget Optimization: Consolidates budgets for Integrated Promotional Planning (IPP) to optimize touchpoint volume recommendations at the HCP level.
3. Temporal Adjustment: Converts HCP quarter recommendations into monthly recommendations using historical two-month actual promotion data.
4. Optimal Touchpoint Distribution: Develops an optimal distribution and sequence of touchpoints utilizing decision tree-based models or other approaches to maximize effectiveness.
Data
5. Data Files include HCP Data including channel priority, historical hcp data, vendor contract and much more, which are all reference to generate NBA plan based on cadence date.

## Notebooks and Scripts
There is one Notebook and One Script in the package:

1. Model Configurations: Notebook for configuring the environment manually using the variables.
2. Model input parameters: Notebook containing the GUI for configuring the environment, and execute the model.
3. Data Files Setup: Notebook to take the filePath from environment and create/update the Databricks Catalog with the tables required for model execution.
4. Model Workflow: Notebook containing all the model functions and logic required for processing NBA.
5. Model Orchestrator: Notebook to run the model workflow with a single execution.
6. Dashboard setup instructions: This notebook lists all the steps required to detup the dashboard in the databricks environment.

## Setup
If you are new to Databricks, create an account at: https://databricks.com/try-databricks

## Coding Environment Setup
1. Create a Databricks Cluster with Databricks Compute 14.3 LTS.
2. Install the following libraries by going to your created cluster, and navigating to libraries tab:
3. Click on "Install New".
4. Navigate to PyPI.
5. Pass the Librariy Names one ata time and install all the required libraries for setup.

## Libraries Required
1. PyYAML
2. gekko
3. kaleido
4. pyspark

## Datasets used
Several omnichannel-specific datasets have been used to build and run the model. Sample datasets can be found in the "Data_Files" folder, along with an additional README "Data File Information.png" file containing data specifications.
