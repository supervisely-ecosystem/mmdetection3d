import os
import supervisely_lib as sly
import sly_globals as g

train_set = None
val_set = None


def init(project_info, project_meta: sly.ProjectMeta, data, state):
    data["randomSplit"] = [
        {"name": "train", "type": "success"},
        {"name": "val", "type": "primary"},
        {"name": "total", "type": "gray"},
    ]
    data["totalImagesCount"] = project_info.items_count

    train_percent = 80
    train_count = int(project_info.items_count / 100 * train_percent)
    state["randomSplit"] = {
        "count": {
            "total": project_info.items_count,
            "train": train_count,
            "val": project_info.items_count - train_count
        },
        "percent": {
            "total": 100,
            "train": train_percent,
            "val": 100 - train_percent
        },
        "shareImagesBetweenSplits": False,
        "sliderDisabled": False,
    }

    state["splitMethod"] = "random"

    state["trainTagName"] = ""
    if project_meta.tag_metas.get("train") is not None:
        state["trainTagName"] = "train"
    state["valTagName"] = ""
    if project_meta.tag_metas.get("val") is not None:
        state["valTagName"] = "val"

    state["trainDatasets"] = []
    state["valDatasets"] = []
    state["untaggedImages"] = "train"
    state["splitInProgress"] = False
    data["trainImagesCount"] = None
    data["valImagesCount"] = None
    data["done2"] = False
    state["collapsed2"] = True
    state["disabled2"] = True



def verify_train_val_sets(train_set, val_set):
    if train_set == 0:
        raise ValueError("Train set is empty, check or change split configuration")
    if val_set == 0:
        raise ValueError("Val set is empty, check or change split configuration")

@g.my_app.callback("create_splits")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def create_splits(api: sly.Api, task_id, context, state, app_logger):
    step_done = False
    global train_set, val_set
    try:
        api.task.set_field(task_id, "state.splitInProgress", True)

        split_method = state["splitMethod"]
        if split_method == "random":
            train_count = state["randomSplit"]["count"]["train"]
            val_count = state["randomSplit"]["count"]["val"]
        else:
            raise ValueError(f"Unknown split method: {split_method}")

        sly.logger.info(f"Train set: {train_count} images")
        sly.logger.info(f"Val set: {val_count} images")
        verify_train_val_sets(train_count, val_count)
        step_done = True
    except Exception as e:
        train_count = None
        val_count = None
        step_done = False
        raise e
    finally:
        api.task.set_field(task_id, "state.splitInProgress", False)
        fields = [
            {"field": "state.splitInProgress", "payload": False},
            {"field": "data.done2", "payload": step_done},
            {"field": "data.trainImagesCount", "payload":  train_count},
            {"field": "data.valImagesCount", "payload": val_count},
        ]
        if step_done is True:
            fields.extend([
                {"field": "state.collapsed3", "payload": False},
                {"field": "state.disabled3", "payload": False},
                {"field": "state.activeStep", "payload": 3},
            ])
        g.api.app.set_fields(g.task_id, fields)
