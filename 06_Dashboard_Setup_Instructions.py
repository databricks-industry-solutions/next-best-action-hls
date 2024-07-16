# Databricks notebook source
# MAGIC %md
# MAGIC # Dashboards Setup Instructions
# MAGIC
# MAGIC This Notebook will list all the steps and requirements necessary to import the dashboards in Azure databricks Environment.

# COMMAND ----------

# MAGIC %md
# MAGIC # Requirements
# MAGIC
# MAGIC ## Functional Requirements
# MAGIC Your Azure Databricks should support SQL and Dashboards functionality (Azure databricks Premium.)
# MAGIC
# MAGIC ## Data Requirements
# MAGIC 1. You should have access to model_performance, data_drift_table, and hcpo_weekly_plan files
# MAGIC 2. Access to Dashboard json files.

# COMMAND ----------

# MAGIC %md
# MAGIC # SQL Warehouse Setup
# MAGIC
# MAGIC 1. Go to Compute Tab in your Azure DataBricks Platform.
# MAGIC 2. Navigate to SQL Warehouses tab.
# MAGIC 3. Click on Create SQL Warehouse.
# MAGIC 4. Name the cluster, and select cluster size as per compute requirements (Small in this case).
# MAGIC 5. Click on Create.

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Import in Unity Catalog
# MAGIC
# MAGIC 1. Go to Data Ingestion within Data Engineering tab from you Databricks Menu.
# MAGIC 2. Click on Create or Modify table
# MAGIC 3. Either 'Drag and Drop' your files (in Data Requirements) or click on browse and select the required files (one by one)
# MAGIC 4. Select the option "Create new Table" if it does not exist or overwrite existing table if table exists.
# MAGIC 5. Click on Create(Overwrite) Table.
# MAGIC 6. Repeat the process for all required Dashboard data files.

# COMMAND ----------

# MAGIC %md
# MAGIC # Importing the Dashboards
# MAGIC The Dashboards should be imported after Data Import in Unity Catalog.
# MAGIC
# MAGIC 1. Go to Dashboards within SQl tab in your Databricks menu.
# MAGIC 2. Click on dropdown Menu beside Create DashBoard Button.
# MAGIC 3. Click on Import Dashboard from file.
# MAGIC 4. Click on choose file.
# MAGIC 5. Select the Dashboard json files from your local system.
# MAGIC 6. Click on Import dashboard.
