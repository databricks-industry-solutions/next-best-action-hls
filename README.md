<img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px">

[![DBR](https://img.shields.io/badge/DBR-CHANGE_ME-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/CHANGE_ME.html)
[![CLOUD](https://img.shields.io/badge/CLOUD-CHANGE_ME-blue?logo=googlecloud&style=for-the-badge)](https://databricks.com/try-databricks)

# NBA - Omnichannel Prediction Model
The solution here aims to use AI/ML models to enhance NBA planning efficiency and effectiveness by leveraging machine learning techniques and comprehensive data analysis, and provide curated recommendation at a monthly/weekly level based on the types of constraints set.

## Business Problem + Solution

### Enhancing NBA Efficiency with Data-Driven Insights

In today's competitive market, achieving optimal Next Best Action (NBA) planning is crucial for maintaining effective engagement with Healthcare Professionals (HCPs). Traditional methods often fail to consider the dynamic nature of HCP preferences and external constraints, leading to suboptimal promotional planning and budget utilization.

Our Omnichannel Prediction Model addresses this challenge by leveraging advanced AI/ML techniques to provide actionable insights and recommendations. By integrating comprehensive data analysis and machine learning models, we enable organizations to:

1. **Improve Planning Precision:** Establish quarterly guardrails and budget optimization strategies to ensure that promotional efforts align with engagement goals and vendor contracts.
2. **Maximize Budget Efficiency:** Optimize touchpoint volume recommendations at the HCP level, ensuring that resources are allocated where they will have the most impact.
3. **Adapt to Temporal Changes:** Convert quarterly recommendations into granular monthly plans, allowing for flexibility and responsiveness to changing market conditions.
4. **Enhance Touchpoint Effectiveness:** Utilize decision tree-based models to develop an optimal sequence and distribution of touchpoints, maximizing the effectiveness of each interaction.

By implementing this solution, organizations can significantly enhance their NBA planning processes, leading to increased engagement with HCPs, better budget management, and improved overall effectiveness of promotional activities.

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

---

© 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license]. All included or referenced third party libraries are subject to the licenses set forth below.

| Library         | Description                                     | License           | Source                                             |
|-----------------|-------------------------------------------------|-------------------|----------------------------------------------------|
| pandas          | Data manipulation and analysis                  | BSD 3-Clause      | https://github.com/pandas-dev/pandas               |
| numpy           | Numerical computing tools                       | BSD 3-Clause      | https://github.com/numpy/numpy                     |
| scikit-learn    | Machine learning library                        | BSD 3-Clause      | https://github.com/scikit-learn/scikit-learn       |
| gekko           | Optimization suite                              | MIT               | https://github.com/BYU-PRISM/GEKKO                 |
| joblib          | Serialization and deserialization               | BSD 3-Clause      | https://github.com/joblib/joblib                   |
| pyyaml          | YAML parsing and writing                        | MIT               | https://github.com/yaml/pyyaml                     |
| plotly          | Interactive plotting library                    | MIT               | https://github.com/plotly/plotly.py                |
| matplotlib      | Static plotting library                         | Matplotlib License| https://github.com/matplotlib/matplotlib           |
| mlflow          | Machine learning lifecycle management           | Apache 2.0        | https://github.com/mlflow/mlflow                   |
