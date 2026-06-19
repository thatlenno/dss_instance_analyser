from dataiku.customwebapp import get_webapp_config
import dataiku

# 1. Fetch the user's complete UI selections as a dictionary
config = get_webapp_config()

# 2. Extract values using the precise 'name' keys from webapp.json
dataset_name = config.get("input_dataset")
folder_id = config.get("target_folder")
group_column = config.get("dimension_col")
numerical_metrics = config.get("metric_cols", [])
should_sample = config.get("enable_sampling", False)

# 3. Instantiate Dataiku objects dynamically
dataset = dataiku.Dataset(dataset_name)
df = dataset.get_dataframe()

# Now 'df' is ready to be handled by native Dash code (dcc, html, callback loops)!