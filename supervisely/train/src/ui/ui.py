import supervisely_lib as sly
import sly_globals as g
import input_project as input_project
import splits as train_val_split
import tags
import validate_training_data
import augs
import architectures as model_architectures
import hyperparameters as hyperparameters
import hyperparameters_python as hyperparameters_python
import monitoring as monitoring


@sly.timeit
def init(data, state):
    state["activeStep"] = 1
    state["restartFrom"] = None

    input_project.init(data, state)
    train_val_split.init(g.project_info, g.project_meta, data, state)
    tags.init(data, state)
    validate_training_data.init(data, state)
    augs.init(data, state)
    model_architectures.init(data, state)
    hyperparameters.init(data, state)
    hyperparameters_python.init(data, state)
    monitoring.init(data, state)


@g.my_app.callback("restart")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def restart(api: sly.Api, task_id, context, state, app_logger):
    restart_from_step = state["restartFrom"]
    data = {}
    state = {}

    monitoring.init(data, state)

    if restart_from_step <= 2:
        train_val_split.init(g.project_info, g.project_meta, data, state)
    if restart_from_step <= 3:
        if restart_from_step == 3:
            tags.restart(data, state)
        else:
            tags.init(data, state)
    if restart_from_step <= 4:
        validate_training_data.init(data, state)
    if restart_from_step <= 5:
        if restart_from_step == 5:
            augs.restart(data, state)
        else:
            augs.init(data, state)
    if restart_from_step <= 6:
        if restart_from_step == 6:
            model_architectures.restart(data, state)
        else:
            model_architectures.init(data, state)
    if restart_from_step <= 7:
        if restart_from_step == 7:
            hyperparameters.restart(data, state)
        else:
            hyperparameters.init(data, state)
    if restart_from_step <= 8:
        if restart_from_step == 8:
            hyperparameters_python.restart(data, state)
        else:
            hyperparameters_python.init(data, state)


    fields = [
        {"field": "data", "payload": data, "append": True, "recursive": False},
        {"field": "state", "payload": state, "append": True, "recursive": False},
        {"field": "state.restartFrom", "payload": None},
        {"field": f"state.collapsed{restart_from_step}", "payload": False},
        {"field": f"state.disabled{restart_from_step}", "payload": False},
        {"field": "state.activeStep", "payload": restart_from_step},
    ]
    g.api.app.set_fields(g.task_id, fields)
    g.api.app.set_field(task_id, "data.scrollIntoView", f"step{restart_from_step}")
