# Databricks notebook source
# MAGIC %md
# MAGIC # Overview
# MAGIC
# MAGIC ## In this Notebook:
# MAGIC <ul> <li> Create the Widgets containing general config widgets and  model constraints widgets.
# MAGIC <li> Create the function to update the config file, and run the model workflow script.
# MAGIC <li> Integrate all the widgets and functions to generate a Interactive UI which will update the existing model config file and then start the model execution.

# COMMAND ----------

# MAGIC %md
# MAGIC # Initial Imports and Dependencies Setup

# COMMAND ----------

from IPython.display import display
from ipywidgets import interact, widgets
from ipywidgets import VBox, HBox, Button, DatePicker, Dropdown, BoundedIntText, FloatSlider, Accordion, Label, Text, Tab
import yaml
import asyncio

# COMMAND ----------

style = {'description_width': 'initial'}

# COMMAND ----------

# MAGIC %md
# MAGIC #General

# COMMAND ----------

# MAGIC %md
# MAGIC This section contains the general widget variables such as start date, end date and refresh date. These are general parameters and affect the overall lifecycle of the NBA Cadence

# COMMAND ----------

# MAGIC %md
# MAGIC #### NBA Start Date: Defines the Start Date for the run of module

# COMMAND ----------

# dbutils.widgets.text("1. NBA Start Date","")
NBA_START_DATE = DatePicker(
    description="NBA Start Date",
    disabled = False,
    style = style
    )

# COMMAND ----------

# MAGIC %md
# MAGIC #### NBA END Date: Marks the End of NBA Module

# COMMAND ----------

# dbutils.widgets.text("2. NBA End Date","NULL")
NBA_END_DATE = DatePicker(
    description="NBA End Date",
    disabled = False,
    style = style
    )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Marks the Refresh between start and end date

# COMMAND ----------

# dbutils.widgets.text("3. NBA Refresh Date","NULL")
NBA_REFRESH_DATE = DatePicker(
    description="NBA Refresh Date",
    disabled=False,
    style = style
    )

# COMMAND ----------

# MAGIC %md
# MAGIC #Constraints
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### This section contains the constraints which directly affect the output of the NBA model. This section contains:
# MAGIC <ol>
# MAGIC   <li>Refresh Cadence: This selects whether to generate output for weekly, monthly or quarterly NBA Cadence.
# MAGIC   <li>TouchPoint distribution weekly: This is used if Refresh cadence is selected as weekly. 
# MAGIC   <li>Max Touchpoints: This describes max touchpoints to be used if refresh cadence = weekly and touchpoint distribution weekly = aggressive
# MAGIC   <li>Touchpoint distribution monthly: how touchpoints are distributed if refresh cadence = monthly
# MAGIC   <li>Touchpoint distribution factor: has a value between 0-1
# MAGIC </ol>

# COMMAND ----------

# dbutils.widgets.dropdown("3. NBA Refresh Cadence", "MONTHLY", ["MONTHLY", "WEEKLY", "YEARLY", "QUATERLY", "BIWEEKLY", "HALF YEARLY"])
NBA_REFRESH_CADENCE = Dropdown(
    options=['MONTHLY','WEEKLY','QUATERLY'],
    value = 'MONTHLY',
    description = 'Select NBA Refresh Cadence',
    style = style
    )

# COMMAND ----------

# dbutils.widgets.dropdown("4. Touchpoint Distribution [Weekly]", "EQUAL", ["EQUAL", "AGGRESSIVE"])
TOUCHPOINT_DISTRIBUTION_WEEKLY = Dropdown(
    options=['EQUAL','AGGRESSIVE'],
    value = 'EQUAL',
    description = 'TouchPoint Distribution [Weekly]',
    style = style
    )

# COMMAND ----------

# dbutils.widgets.text("5. Max Touchpoints","NULL")
MAX_TOUCHPOINTS = BoundedIntText(
    min=1,
    step=1,
    description='Max TouchPoints(if weekly Agressive)',
    disabled=False,
    style = style
)

# COMMAND ----------

# dbutils.widgets.dropdown("Touchpoint Distribution [Monthly]", "EQUAL", ["EQUAL", "PREDEFINED_SPLIT", "HISTORICAL"])
TOUCHPOINT_DISTRIBUTION_MONTHLY = Dropdown(
    options=['EQUAL','PREDEFINED_SPLIT','HISTORICAL'],
    value='EQUAL',
    description="Touchpoint Distribution [Monthly]",
    disabled=False,
    style = style
)

# COMMAND ----------

# dbutils.widgets.text("6. Touchpoint Distribution Factor [0 to 1]","NULL")
TOUCHPOINT_DISTRIBUTION_FACTOR = HBox(children=[Label('Touchpoint Distribution Factor'),FloatSlider(
    min=0,
    max=1,
    step=0.1,
    # description='Touchpoint Distribution Factor',
    disabled = False,
    continuous_update = False,
    orientation = 'horizontal',
    readout = True,
    readout_format = '.1f',
    style = style
)])

# COMMAND ----------

# MAGIC %md
# MAGIC ## FilePaths for Input CSV Files

# COMMAND ----------

ASSET_AVAILABILITY = Text(
    value='./Data_Files/Asset_Availability.csv',
    placeholder='Type the file path here',
    description='asset availability file:',
    disabled=False,
    style = style
)

# COMMAND ----------

ASSET_AVAILABILITY_TABLE = Text(
    value='asset_availability',
    placeholder='Type table name here',
    description='asset availability table:',
    disabled=False,
    style = style
)

# COMMAND ----------

CONSTRAINT_SEGMENT = Text(
    value='./Data_Files/Constraint_segment.csv',
    placeholder='Type the file path here',
    description='constraint segment file:',
    disabled=False,
    style = style
)

# COMMAND ----------

CONSTRAINT_SEGMENT_TABLE = Text(
    value='constraint_segment',
    placeholder='Type the table name here',
    description='constraint segment table:',
    disabled=False,
    style = style
)

# COMMAND ----------

CUSTOMER_DATA = Text(
    value='./Data_Files/Sample_HCP_Master_Demo.csv',
    placeholder='Type the file path here',
    description='customer data file:',
    disabled=False,
    style = style
)

# COMMAND ----------

CUSTOMER_DATA_TABLE = Text(
    value='customer_data',
    placeholder='Type the table name here',
    description='customer data table:',
    disabled=False,
    style = style
)

# COMMAND ----------

ENGAGEMENT_GOAL = Text(
    value='./Data_Files/Eng_Target_File_Demo.csv',
    placeholder='Type the file path here',
    description='engagement goal file:',
    disabled=False,
    style = style
)

# COMMAND ----------

ENGAGEMENT_GOAL_TABLE = Text(
    value='engagement_goal',
    placeholder='Type the table name here',
    description='engagement goal table:',
    disabled=False,
    style = style
)

# COMMAND ----------

HISTORICAL_DATA = Text(
    value='./Data_Files/HCP_Historical_Data_Demo.csv',
    placeholder='Type the file path here',
    description='historical data file:',
    disabled=False,
    style = style
)

# COMMAND ----------

HISTORICAL_DATA_TABLE = Text(
    value='historical_data',
    placeholder='Type the table name here',
    description='historical data table:',
    disabled=False,
    style = style
)

# COMMAND ----------

MIN_GAP = Text(
    value='./Data_Files/Min_Gap_Constraints - Copy.csv',
    placeholder='Type the file path here',
    description='min gap file:',
    disabled=False,
    style = style
)

# COMMAND ----------

MIN_GAP_TABLE = Text(
    value='min_gap',
    placeholder='Type the table name here',
    description='min gap table:',
    disabled=False,
    style = style
)

# COMMAND ----------

MMIX_OUTPUT = Text(
    value='./Data_Files/Marketing_Mix_Input_Demo.csv',
    placeholder='Type the file path here',
    description='mmix output file:',
    disabled=False,
    style = style
)

# COMMAND ----------

MMIX_OUTPUT_TABLE = Text(
    value='mmix_output',
    placeholder='Type the table name here',
    description='mmix output table:',
    disabled=False,
    style = style
)

# COMMAND ----------

PRIORITY = Text(
    value='./Data_Files/HCP_Channel_Priority.csv',
    placeholder='Type the file path here',
    description='PRIORITY file:',
    disabled=False,
    style = style
)

# COMMAND ----------

PRIORITY_TABLE = Text(
    value='priority',
    placeholder='Type the table name here',
    description='PRIORITY table:',
    disabled=False,
    style = style
)

# COMMAND ----------

VENDOR_CONTRACT = Text(
    value='./Data_Files/Vendor_Contract.csv',
    placeholder='Type the file path here',
    description='vendor contract file:',
    disabled=False,
    style = style
)

# COMMAND ----------

VENDOR_CONTRACT_TABLE = Text(
    value='vendor_contract',
    placeholder='Type the table name here',
    description='vendor contract table:',
    disabled=False,
    style = style
)

# COMMAND ----------

OUTPUT_FOLDER_PATH = Text(
    value='./Outputs',
    placeholder='Type the Folder path here',
    description='Folder Path for Output Files',
    disabled=False,
    style = style
)

# COMMAND ----------

RUN_ID = Text(
    value='',
    placeholder='Type the run id for model (if required)',
    description='Run ID for the model run',
    disabled=False,
    style = style
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Config Edit Code

# COMMAND ----------

# MAGIC %md
# MAGIC #### This section of notebook contains the code which updates the config file when submit button is pressed based on the inputes in the general and constraints widgets.
# MAGIC This uses the yaml library of python to update the existing config file available. Change the location of config file as per the location in your system.

# COMMAND ----------

def config_update(nba_start, nba_end, nba_refresh, refresh_cadence, weekly_tps, max_tps, monthly_tps,
                  asset_availability, constraint_segment, customer_data, engagement_goal,
                  historical_data, min_gap, mmix_output, priority, vendor_contract,
                  output_folder_path, run_id, asset_availability_table, constraint_segment_table, customer_data_table, engagement_goal_table,
                  historical_data_table, min_gap_table, mmix_output_table, priority_table, vendor_contract_table):
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
    table_names = {
        'asset_availability': asset_availability_table,
        'constraint_segment': constraint_segment_table,
        'customer_data': customer_data_table,
        'engagement_goal': engagement_goal_table,
        'historical_data': historical_data_table,
        'min_gap': min_gap_table,
        'mmix_output': mmix_output_table,
        'priority': priority_table,
        'vendor_contract': vendor_contract_table
    }

    # Update file paths
    for key, value in file_paths.items():
        config['file_paths'][key] = value

    for key, value in table_names.items():
        config['table_names'][key] = value
    # Update output folder path and run ID
    config['output']['output_folder_path'] = output_folder_path
    config['output']['run_id'] = run_id

    # Write the updated configuration back to the YAML file
    with open("./Archive/config.yaml", 'w') as file:
        yaml.dump(config, file)
    print("Configs Saved")

# COMMAND ----------

# MAGIC %md
# MAGIC NBA START DATE: Defines the start date for nba cadence.<br>
# MAGIC Should be in format : dd-mm-yyyy

# COMMAND ----------

# MAGIC %md
# MAGIC # UI Generator

# COMMAND ----------

# MAGIC %md
# MAGIC #### The below code combines all the widgets (general and contarints), along with config update function execution to create a Interactive UI for user input on parameters

# COMMAND ----------

out = widgets.Output(layout={'border': '1px solid black'})

# COMMAND ----------

# MAGIC %md
# MAGIC ##### handle_submit function contains logic to be executed when submit button in UI is clicked.

# COMMAND ----------

def handle_submit(b):
  # Access user input from widget values
  out.clear_output()
  start_date = NBA_START_DATE.value
  end_date = NBA_END_DATE.value
  refresh_date = NBA_REFRESH_DATE.value
  refresh_cadence = NBA_REFRESH_CADENCE.value
  weekly_tps = TOUCHPOINT_DISTRIBUTION_WEEKLY.value
  max_tps = MAX_TOUCHPOINTS.value
  monthly_tps = TOUCHPOINT_DISTRIBUTION_MONTHLY.value
  tps_factor = TOUCHPOINT_DISTRIBUTION_FACTOR.children[1].value
  asset_availability = ASSET_AVAILABILITY.value
  constraint_segment = CONSTRAINT_SEGMENT.value
  customer_data = CUSTOMER_DATA.value
  engagement_goal = ENGAGEMENT_GOAL.value
  historical_data = HISTORICAL_DATA.value
  min_gap = MIN_GAP.value
  mmix_output = MMIX_OUTPUT.value
  priority = PRIORITY.value
  vendor_contract = VENDOR_CONTRACT.value
  asset_availability_table = ASSET_AVAILABILITY_TABLE.value
  constraint_segment_table = CONSTRAINT_SEGMENT_TABLE.value
  customer_data_table = CUSTOMER_DATA_TABLE.value
  engagement_goal_table = ENGAGEMENT_GOAL_TABLE.value
  historical_data_table = HISTORICAL_DATA_TABLE.value
  min_gap_table = MIN_GAP_TABLE.value
  mmix_output_table = MMIX_OUTPUT_TABLE.value
  priority_table = PRIORITY_TABLE.value
  vendor_contract_table = VENDOR_CONTRACT_TABLE.value
  output_folder_path = OUTPUT_FOLDER_PATH.value
  run_id = RUN_ID.value
  # with out:
  #   print(f"""Values stored!!!
  #         Here are the Values:
  #         Start Date:\t{start_date}
  #         End Date:\t{end_date}
  #         Refresh Date:\t{refresh_date}
  #         Refresh cadence:\t{refresh_cadence}
  #         Weekly Touchpoints:\t{weekly_tps}
  #         Max Touchpoints:\t{max_tps}
  #         Monthly Touchpoints:\t{monthly_tps}
  #         TouchPoints Factor:\t{tps_factor}""")
  # with out:
  # config_update(start_date,end_date,refresh_date,refresh_cadence,weekly_tps,max_tps,monthly_tps)
  with out:
    config_update(start_date, end_date, refresh_date, refresh_cadence, weekly_tps, max_tps, monthly_tps, asset_availability, constraint_segment, customer_data, engagement_goal, historical_data, min_gap, mmix_output, priority, vendor_contract, output_folder_path, run_id, asset_availability_table, constraint_segment_table, customer_data_table, engagement_goal_table, historical_data_table, min_gap_table, mmix_output_table, priority_table, vendor_contract_table)

  # with out:
  #   print("starting model run")
  # # import model_workflow as model_run
  # # model_run.model_start()
  # with out:
  #   print("Output files Created")
    

  # ... (Extract values from other widgets)
  
  # Call your Gekko script or other execution logic here
  # ...

# COMMAND ----------

# MAGIC %md
# MAGIC ##### model_start function starts the execution of the model_workflow after the config file is updated on the click of submit button.

# COMMAND ----------

async def run_notebook_async():
    result = await run_notebook()
    print(f"Notebook execution result: {result}")

# COMMAND ----------

async def run_notebook():
    # This function should be adapted to your needs
    try:
        result = dbutils.notebook.run("03. Model Workflow", timeout_seconds=3600)
        return result
    except Exception as e:
        return str(e)

# COMMAND ----------

running_task = None

# COMMAND ----------

def model_start(b):
    global running_task
    Start_Button.disabled = True
    stop_button.disabled = False
    with out:
        print("starting model run")
        running_task = asyncio.create_task(run_notebook_async())

# COMMAND ----------

def model_stop(b):
    global running_task
    
    # Cancel the running task if it exists
    if running_task:
        running_task.cancel()
    with out:
        out.clear_output()
        print("Model Execution Cancelled!")
    # Enable execute button and disable stop button
    Start_Button.disabled = False
    stop_button.disabled = True

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Combining all the widgets and functions into a final UI

# COMMAND ----------

# Define section titles
general_title = "General"
constraints_title = "Constraints"
filepaths_title = "FilePaths"
output_title = "Outputs"
table_title = "Table Names"
submit_button = Button(description="Save Configurations")
# Start_Button = Button(description="Execute Model")
Start_Button = Button(
    description='Execute Model',
    disabled=False,
    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click to Start Model Execution',
    icon='play'  # (FontAwesome names without the `fa-` prefix)
)
stop_button = Button(
    description='Stop Model Execution',
    disabled=True,
    button_style='danger',  # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click to Stop Model Execution',
    icon='stop',
    style = style  # (FontAwesome names without the `fa-` prefix)
)

# Organize widgets into sections using layout containers
general_section = VBox(children=[NBA_START_DATE, NBA_END_DATE, NBA_REFRESH_DATE])
# accordion = widgets.Accordion(children=[widgets.IntSlider(), widgets.Text()])

constraints_section = VBox(
    children=[NBA_REFRESH_CADENCE, TOUCHPOINT_DISTRIBUTION_WEEKLY, MAX_TOUCHPOINTS, TOUCHPOINT_DISTRIBUTION_MONTHLY, TOUCHPOINT_DISTRIBUTION_FACTOR]
)
filepaths_section = VBox(
    children=[ASSET_AVAILABILITY, CONSTRAINT_SEGMENT, CUSTOMER_DATA, ENGAGEMENT_GOAL, HISTORICAL_DATA, MIN_GAP, MMIX_OUTPUT, PRIORITY, VENDOR_CONTRACT]
)
tables_section = VBox(
    children=[ASSET_AVAILABILITY_TABLE, CONSTRAINT_SEGMENT_TABLE, CUSTOMER_DATA_TABLE, ENGAGEMENT_GOAL_TABLE, HISTORICAL_DATA_TABLE, MIN_GAP_TABLE, MMIX_OUTPUT_TABLE, PRIORITY_TABLE, VENDOR_CONTRACT_TABLE]
)
output_section = VBox(
    children=[OUTPUT_FOLDER_PATH, RUN_ID]
)
# Combine sections and button into a single layout
# settings_ui = VBox(
#     children=[Label(general_title),general_section,Label(constraints_title),constraints_section,Label(filepaths_title),filepaths_section,Label(table_title),tables_section,Label(output_title),output_section]
# )
tab_contents = [
    general_section,
    constraints_section,
    filepaths_section,
    tables_section,
    output_section
]

tab_titles = [
    general_title,
    constraints_title,
    filepaths_title,
    table_title,
    output_title
]

settings_ui = Tab()
settings_ui.children = tab_contents

# Set titles for each tab
for i, title in enumerate(tab_titles):
    settings_ui.set_title(i, title)

# settings_ui.set_title(0, general_title)
# settings_ui.set_title(1, constraints_title)

final_ui = VBox(children=[settings_ui,HBox(children=[submit_button,Start_Button,stop_button]),out])

# Function to handle button click (replace with your script execution logic)
# Link button click to the function
submit_button.on_click(handle_submit)
Start_Button.on_click(model_start)
stop_button.on_click(model_stop)

# COMMAND ----------

# MAGIC %md
# MAGIC # UI RUN

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Displaying the interactive UI

# COMMAND ----------

# Display the UI
out.clear_output()
display(final_ui)
