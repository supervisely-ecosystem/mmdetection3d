import supervisely_lib as sly
import sly_globals as g


def init(data, state):
    state["useAugs"] = True
    state["collapsed5"] = True
    state["disabled5"] = True
    data["done5"] = False

    state["pointShuffle"] = False
    state["objectNoise"] = False
    state["rangeFilter"] = False


def restart(data, state):
    data["done5"] = False



@g.my_app.callback("use_augs")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_augs(api: sly.Api, task_id, context, state, app_logger):
    fields = [
        {"field": "data.done5", "payload": True},
        {"field": "state.collapsed6", "payload": False},
        {"field": "state.disabled6", "payload": False},
        {"field": "state.activeStep", "payload": 6},
    ]

    fields.extend([
        {"field": "state.pointShuffle", "payload": state["pointShuffle"]},
        {"field": "state.objectNoise", "payload": state["objectNoise"]},
        {"field": "state.rangeFilter", "payload": state["rangeFilter"]}
    ])


    g.api.app.set_fields(g.task_id, fields)
