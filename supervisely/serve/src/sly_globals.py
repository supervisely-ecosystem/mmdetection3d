import os
import supervisely_lib as sly
import pathlib
import sys

my_app = sly.AppService()
api = my_app.public_api
task_id = my_app.task_id

team_id = int(os.environ['context.teamId'])
workspace_id = int(os.environ['context.workspaceId'])
remote_weights_path = os.environ['modal.state.slyFile']

modelWeightsOptions = os.environ['modal.state.modelWeightsOptions']
pretrained_weights = os.environ['modal.state.selectedModel']
custom_weights = os.environ['modal.state.weightsPath']

device = 'cuda'

root_source_path = str(pathlib.Path(sys.argv[0]).parents[3])
sly.logger.info(f"Root source directory: {root_source_path}")
sys.path.append(root_source_path)

model = None
meta: sly.ProjectMeta = None
