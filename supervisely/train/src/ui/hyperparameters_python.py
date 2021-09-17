import os
import supervisely_lib as sly
import train_config
import sly_globals as g

opts = {
    "mode": 'ace/mode/python',
    "showGutter": False,
    "maxLines": 100,
    "highlightActiveLine": False
}
opts_read = {
    **opts,
    "readOnly": True
}
opts_write = {
    **opts,
    "readOnly": False
}


def init(data, state):
    state["trainConfigLines"] = ""

    data["configsPyViewOptionsRead"] = opts_read
    data["configsPyViewOptionsWrite"] = opts_write
    state["pyConfigsViewOptions"] = opts_read
    state["advancedPy"] = False

    state["collapsed8"] = True
    state["disabled8"] = True
    data["done8"] = False



def restart(data, state):
    data["done8"] = False


@g.my_app.callback("preview_configs")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def preview_configs(api: sly.Api, task_id, context, state, app_logger):
    train_config_path = train_config.generate_config(state)

    with open(train_config_path) as f:
        config_str = f.read()

    fields = [
        {"field": "state.trainConfigLines", "payload": config_str},
        {"field": "state.trainConfigPath", "payload": train_config_path}
    ]
    api.task.set_fields(task_id, fields)


@g.my_app.callback("accept_py_configs")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def accept_py_configs(api: sly.Api, task_id, context, state, app_logger):
    train_config.save_from_state(state)
    fields = [
        {"field": "data.done8", "payload": True},
        {"field": "state.collapsed9", "payload": False},
        {"field": "state.disabled9", "payload": False},
        {"field": "state.activeStep", "payload": 9},
        {"field": "state.pyConfigsViewOptions", "payload": opts_read},
    ]
    g.api.app.set_fields(g.task_id, fields)
