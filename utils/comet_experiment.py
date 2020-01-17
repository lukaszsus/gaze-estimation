from comet_ml import Experiment


class CometExperiment(Experiment):
    def __init__(self, api_key=None, project_name=None, team_name=None, workspace=None, log_code=True, log_graph=True,
                 auto_param_logging=True, auto_metric_logging=True, parse_args=True, auto_output_logging="default",
                 log_env_details=True, log_git_metadata=True, log_git_patch=True, disabled=False):
        super().__init__(self, api_key=api_key, project_name=project_name, team_name=team_name, workspace=workspace,
                         log_code=log_code, log_graph=log_graph, auto_param_logging=auto_param_logging,
                         auto_metric_logging=auto_metric_logging, parse_args=parse_args,
                         auto_output_logging=auto_output_logging, log_env_details=log_env_details,
                         log_git_metadata=log_git_metadata, log_git_patch=log_git_patch, disabled=disabled)
