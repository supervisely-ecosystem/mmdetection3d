import os
import supervisely_lib as sly
import sly_globals as g


def init(data, state):
    state["epochs"] = 5 # max_epoch
    state["gpusId"] = '0'

    state["steps_per_epoch_train"] = 10
    state["batchSizeTrain"] = 1
    state["batchSizeVal"] = 1
    state["checkpointInterval"] = 5 # save_ckpt_freq

    state["lr"] = 0.001
    state["weightDecay"] = 0.0001
    state["gradClipNorm"] = 2

    state["collapsed7"] = True
    state["disabled7"] = True
    state["done7"] = False


def restart(data, state):
    data["done7"] = False


@g.my_app.callback("use_hyp")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_hyp(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "data.done7", "payload": True},
        {"field": "state.collapsed8", "payload": False},
        {"field": "state.disabled8", "payload": False},
        {"field": "state.activeStep", "payload": 8},
    ]
    g.api.app.set_fields(g.task_id, fields)
