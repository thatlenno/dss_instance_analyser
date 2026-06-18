# Code for custom code recipe get-instance-data (imported from a Python recipe)

# To finish creating your custom recipe from your original PySpark recipe, you need to:
#  - Declare the input and output roles in recipe.json
#  - Replace the dataset names by roles access in your code
#  - Declare, if any, the params of your custom recipe in recipe.json
#  - Replace the hardcoded params values by acccess to the configuration map

# See sample code below for how to do that.
# The code of your original recipe is included afterwards for convenience.
# Please also see the "recipe.json" file for more information.

# import the classes for accessing DSS objects from the recipe
import dataiku
# Import the helpers for custom recipes
from dataiku.customrecipe import get_input_names_for_role
from dataiku.customrecipe import get_output_names_for_role
from dataiku.customrecipe import get_recipe_config

# Inputs and outputs are defined by roles. In the recipe's I/O tab, the user can associate one
# or more dataset to each input and output role.
# Roles need to be defined in recipe.json, in the inputRoles and outputRoles fields.

# To  retrieve the datasets of an input role named 'input_A' as an array of dataset names:
input_A_names = get_input_names_for_role('input_A_role')
# The dataset objects themselves can then be created like this:
input_A_datasets = [dataiku.Dataset(name) for name in input_A_names]

# For outputs, the process is the same:
output_A_names = get_output_names_for_role('instance_data')
output_A_datasets = [dataiku.Dataset(name) for name in output_A_names]


# The configuration consists of the parameters set up by the user in the recipe Settings tab.

# Parameters must be added to the recipe.json file so that DSS can prompt the user for values in
# the Settings tab of the recipe. The field "params" holds a list of all the params for wich the
# user will be prompted for values.

# The configuration is simply a map of parameters, and retrieving the value of one of them is simply:
#my_variable = get_recipe_config()['parameter_name']

# For optional parameters, you should provide a default value in case the parameter is not present:
#my_variable = get_recipe_config().get('parameter_name', None)

# Note about typing:
# The configuration of the recipe is passed through a JSON object
# As such, INT parameters of the recipe are received in the get_recipe_config() dict as a Python float.
# If you absolutely require a Python int, use int(get_recipe_config()["my_int_param"])


#############################
# Your original recipe
#############################

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
from datetime import datetime, timezone

# Recipe type identifiers, grouped by language/category. Everything returned
# by list_recipes() that isn't in one of these sets is a "visual" recipe
# (sync, prepare, join, group, ...).
SQL_RECIPE_TYPES = {
    "sql_query", "sql_script", "spark_sql_query", "sparksql_query", "hive",
    "impala", "ksql",
}
PYTHON_RECIPE_TYPES = {"python", "pyspark"}
R_RECIPE_TYPES = {"r", "sparkr"}
OTHER_CODE_RECIPE_TYPES = {"spark_scala", "streaming_spark_scala", "shell"}
CODE_RECIPE_TYPES = SQL_RECIPE_TYPES | PYTHON_RECIPE_TYPES | R_RECIPE_TYPES | OTHER_CODE_RECIPE_TYPES

# LLM Mesh / GenAI visual recipes.
LLM_RECIPE_TYPES = {
    "prompt", "nlp_llm_rag_embedding", "embed_documents", "extract_content",
    "extract_fields",
}

ACTIVE_USER_WINDOW_DAYS = 30

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def ms_to_iso(ms):
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000.0).isoformat()


def get_dataset_stats(project):
    """Aggregate connections used, plus metrics/checks/data quality rule counts,
    across all datasets in a project. Requires one settings call per dataset,
    so this is the slow part of the script on instances with many datasets."""
    connections = set()
    num_metrics = 0
    num_checks = 0
    num_dq_rules = 0

    for ds_item in project.list_datasets():
        connection = ds_item.get("params", {}).get("connection")
        if connection:
            connections.add(connection)

        try:
            dataset = project.get_dataset(ds_item["name"])
            raw_settings = dataset.get_settings().get_raw()
            num_metrics += len(raw_settings.get("metrics", {}).get("probes", []))
            num_checks += len(raw_settings.get("metricsChecks", {}).get("checks", []))
            num_dq_rules += len(dataset.get_data_quality_rules().list_rules(as_type="dict"))
        except Exception:
            # Some dataset types/DSS versions don't support all of the above.
            pass

    return connections, num_metrics, num_checks, num_dq_rules


def get_scenario_stats(project):
    """Number of scenarios, how many are auto-triggered (vs. manual-only), and
    the most recent finished run across all of them - a proxy for how much of
    the project's workflow is automated vs. run by hand."""
    scenarios = project.list_scenarios()
    num_scenarios = len(scenarios)
    num_auto_triggered = sum(1 for s in scenarios if s.get("active"))

    last_run = None
    for scenario_item in scenarios:
        try:
            run = project.get_scenario(scenario_item["id"]).get_last_finished_run()
        except Exception:
            continue
        if run is not None and (last_run is None or run.start_time > last_run.start_time):
            last_run = run

    return num_scenarios, num_auto_triggered, last_run


def get_saved_model_stats(project):
    """Visual ML (Lab-trained) models vs. imported/external ones (MLflow,
    proxy models to external endpoints). A saved model has an origin ML task
    only if it was trained through the visual ML Lab."""
    saved_models = project.list_saved_models()
    num_saved_models = len(saved_models)
    num_visual_ml_models = 0
    for sm in saved_models:
        try:
            if project.get_saved_model(sm["id"]).get_origin_ml_task() is not None:
                num_visual_ml_models += 1
        except Exception:
            pass
    return num_saved_models, num_visual_ml_models


def get_automation_deployment_count(project_key, deployments_by_project, automation_deployer_available):
    """None when no Project Deployer/automation node is reachable from this
    design node, as opposed to 0 which means the deployer exists but this
    project has no deployments."""
    if not automation_deployer_available:
        return None
    return deployments_by_project.get(project_key, 0)


def get_active_collaborators(permissions, user_activity, user_activity_available):
    """Of the users with project permissions, how many had DSS UI activity in
    the last ACTIVE_USER_WINDOW_DAYS days. None if the activity API wasn't
    reachable (requires an admin API key)."""
    if not user_activity_available:
        return None
    cutoff_ms = (datetime.now(timezone.utc).timestamp() - ACTIVE_USER_WINDOW_DAYS * 86400) * 1000
    logins = [p.get("user") for p in permissions.get("permissions", []) if p.get("user")]
    return sum(1 for login in logins if user_activity.get(login, 0) >= cutoff_ms)


def get_govern_tracked(project):
    """Best-effort lookup: the public DSS API has no documented boolean for
    'is this project tracked in Govern', so scan project settings for any
    govern-related key and surface whatever is found."""
    try:
        raw_settings = project.get_settings().get_raw()
    except Exception:
        return None
    for key, value in raw_settings.items():
        if "govern" in key.lower():
            return value
    return None

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
client = dataiku.api_client()

# Instance-wide lookups done once, gracefully degrading to "unavailable" (None
# downstream) rather than failing the whole script if the feature isn't set
# up on this instance (no automation node connected, no admin API key, etc.).
try:
    deployments_by_project = {}
    for deployment in client.get_projectdeployer().list_deployments():
        published_project_key = deployment.get_settings().published_project_key
        deployments_by_project[published_project_key] = deployments_by_project.get(published_project_key, 0) + 1
    automation_deployer_available = True
except Exception:
    deployments_by_project = {}
    automation_deployer_available = False

try:
    user_activity = {
        ua.login: ua.get_raw().get("lastSessionActivity") or 0
        for ua in client.list_users_activity()
    }
    user_activity_available = True
except Exception:
    user_activity = {}
    user_activity_available = False

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
rows = []
for project_key in client.list_project_keys():
    project = client.get_project(project_key)
    summary = project.get_summary()

    creation_tag = summary.get("creationTag") or {}
    version_tag = summary.get("versionTag") or {}

    permissions = project.get_permissions()
    # Permission entries can be users or groups, so this is an upper-bound
    # proxy for "number of people with access", not an exact head count.
    num_collaborators = len(permissions.get("permissions", []))

    recipes = project.list_recipes()
    num_recipes = len(recipes)
    num_sql_recipes = sum(1 for r in recipes if r.get("type") in SQL_RECIPE_TYPES)
    num_python_recipes = sum(1 for r in recipes if r.get("type") in PYTHON_RECIPE_TYPES)
    num_r_recipes = sum(1 for r in recipes if r.get("type") in R_RECIPE_TYPES)
    num_other_code_recipes = sum(1 for r in recipes if r.get("type") in OTHER_CODE_RECIPE_TYPES)
    num_code_recipes = num_sql_recipes + num_python_recipes + num_r_recipes + num_other_code_recipes
    num_llm_recipes = sum(1 for r in recipes if r.get("type") in LLM_RECIPE_TYPES)
    num_visual_recipes = num_recipes - num_code_recipes - num_llm_recipes

    connections, num_metrics, num_checks, num_dq_rules = get_dataset_stats(project)
    num_scenarios, num_auto_triggered_scenarios, last_scenario_run = get_scenario_stats(project)
    num_saved_models, num_visual_ml_models = get_saved_model_stats(project)

    try:
        num_api_services = len(project.list_api_services())
    except Exception:
        num_api_services = None

    try:
        num_dashboards = len(project.list_dashboards())
    except Exception:
        num_dashboards = None

    try:
        num_webapps = len(project.list_webapps())
    except Exception:
        num_webapps = None

    try:
        num_wiki_articles = len(list(project.get_wiki().list_articles()))
    except Exception:
        num_wiki_articles = None

    rows.append({
        "project_key": project_key,
        "project_name": summary.get("name"),
        "owner": permissions.get("owner"),
        "created_by": (creation_tag.get("lastModifiedBy") or {}).get("login"),
        "creation_date": ms_to_iso(creation_tag.get("lastModifiedOn")),
        "last_modified_by": (version_tag.get("lastModifiedBy") or {}).get("login"),
        "last_modified_date": ms_to_iso(version_tag.get("lastModifiedOn")),
        "num_collaborators": num_collaborators,
        "num_active_collaborators_30d": get_active_collaborators(permissions, user_activity, user_activity_available),
        "connections_used": ";".join(sorted(connections)),
        "num_connections": len(connections),
        "num_recipes": num_recipes,
        "num_visual_recipes": num_visual_recipes,
        "num_code_recipes": num_code_recipes,
        "num_code_recipes_sql": num_sql_recipes,
        "num_code_recipes_python": num_python_recipes,
        "num_code_recipes_r": num_r_recipes,
        "num_code_recipes_other": num_other_code_recipes,
        "num_llm_recipes": num_llm_recipes,
        "num_metrics": num_metrics,
        "num_checks": num_checks,
        "num_data_quality_rules": num_dq_rules,
        "num_scenarios": num_scenarios,
        "num_auto_triggered_scenarios": num_auto_triggered_scenarios,
        "last_scenario_run_date": last_scenario_run.start_time.isoformat() if last_scenario_run else None,
        "last_scenario_run_outcome": last_scenario_run.outcome if last_scenario_run else None,
        "num_saved_models": num_saved_models,
        "num_visual_ml_models": num_visual_ml_models,
        "num_api_services": num_api_services,
        "num_automation_deployments": get_automation_deployment_count(
            project_key, deployments_by_project, automation_deployer_available
        ),
        "num_dashboards": num_dashboards,
        "num_webapps": num_webapps,
        "num_wiki_articles": num_wiki_articles,
        "govern_tracked": get_govern_tracked(project),
    })

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
instance_data_df = pd.DataFrame(rows)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
instance_data_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
instance_data = dataiku.Dataset("instance_data")
instance_data2.write_with_schema(instance_data_df)
