from collections import defaultdict
import supervisely_lib as sly

import input_project
import splits
import numpy as np
import sly_globals as g
from sly_train_progress import get_progress_cb, reset_progress, init_progress

tag2images = None
tag2urls = None
disabled_tags = []
ccount = {}

progress_index = 3
_preview_height = 120
_max_examples_count = 12

_ignore_tags = ["train", "val"]
_allowed_tag_types = [sly.geometry.cuboid_3d.Cuboid3d]


selected_tags = None


def init(data, state):
    state["selectedTags"] = []
    state["tagsInProgress"] = False
    state["classBalance"] = None

    data["done3"] = False
    data["skippedTags"] = []
    state["collapsed3"] = True
    state["disabled3"] = True

    data["ranges"] = {}
    data["sizes"] = {}

    init_progress(progress_index, data)


def init_classes_switches(state, class_balance):
    for v in class_balance:
        state[f"classSwitch{v['class_name']}"] = True



def restart(data, state):
    data["done3"] = False



def count_classes(progress_cb, allow_classes=None):
    allow_all = not isinstance(allow_classes, list)

    project_fs = sly.PointcloudProject.read_single(g.project_dir)
    tag_names = []
    tags_count = {}
    figs = []
    for dataset_fs in project_fs:
        for item_name in dataset_fs:
            item_path, related_images_dir, ann_path = dataset_fs.get_item_paths(item_name)
            ann_json = sly.io.json.load_json_file(ann_path)
            ann = sly.PointcloudAnnotation.from_json(ann_json, project_fs.meta)

            if len(ann.figures) == 0:
                progress_cb(1)
                continue
            else:
                for fig in ann.figures:
                    tag_name = fig.parent_object.obj_class.name
                    figs.append(fig)
                    if tag_name == 'DontCare':
                        continue
                    if allow_all:
                        tag_names.append(tag_name)
                    elif tag_name in allow_classes:
                        tag_names.append(tag_name)
            progress_cb(1)
    for tag_name in tag_names:
        tags_count[tag_name] = tag_names.count(tag_name)

    return tags_count, figs


def class_balance_table(count_classes_dict, allow_classes=None):
    if not isinstance(allow_classes, list):
        allow_classes = list(count_classes_dict.keys())

    class_balance = []

    filtered_count_dict = {k:v for k,v in count_classes_dict.items() if k in allow_classes}
    if len(filtered_count_dict):
        max_val = max(filtered_count_dict.values())
    else:
        max_val = 1

    for class_name, count_value in count_classes_dict.items():
        if class_name in allow_classes:
            percentage = int(count_value / sum(list(filtered_count_dict.values())) * 100)  # by count
            #percentage = int(abs(abs(count_value - max_val) / max_val * 100 - 100))  # relative
        else:
            percentage = 0

        enabled = class_name in allow_classes
        class_balance.append({
            "class_name": class_name,
            "count": count_value,
            "percentage": percentage,
            "enabled": enabled,
        })

    return class_balance

@g.my_app.callback("switchChanged")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def on_change(api: sly.Api, task_id, context, state, app_logger):
    global ccount
    active_classes = []
    for item in state['classBalance']:
        if item['enabled']:
            active_classes.append(item['class_name'])

    class_balance = class_balance_table(ccount, allow_classes=active_classes)
    fields = [
        {"field": "state.classBalance", "payload": class_balance},
        {"field": "state.selectedTags", "payload": active_classes}
    ]
    g.api.app.set_fields(g.task_id, fields)

def calc_ranges_sizes(figs, progress_cb):
    sizes = {}
    ranges = {}
    for fig in figs:
        class_name = fig.parent_object.obj_class.name
        if class_name not in sizes.keys():
            sizes[class_name] = []
            ranges[class_name] = []

        d = fig.geometry.dimensions
        p = fig.geometry.position
        sizes[class_name].append([d.x, d.y, d.z])
        ranges[class_name].append([p.x, p.y, p.z])
        progress_cb(1)

    for class_name in sizes.keys():
        size = np.array(sizes[class_name]).mean(axis=0)
        _range = np.array(ranges[class_name])
        max_range = _range.max(axis=0)
        min_range = _range.min(axis=0)
        #x_min, y_min, z_min, x_max, y_max, z_max
        ranges[class_name] = [*min_range, *max_range]
        sizes[class_name] = size.tolist()
    return ranges, sizes


@g.my_app.callback("show_tags")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def show_tags(api: sly.Api, task_id, context, state, app_logger):
    global ccount, pointclouds_without_figures

    progress = get_progress_cb(progress_index, "Calculate stats", g.project_info.items_count)

    ccount, figs = count_classes(progress)
    progress = get_progress_cb(progress_index, "Calculate figures stats", len(figs))
    ranges, sizes = calc_ranges_sizes(figs, progress)
    class_balance = class_balance_table(ccount)


    fields = [
        {"field": "state.tagsInProgress", "payload": False},
        {"field": "state.classBalance", "payload": class_balance},
        {"field": "data.ranges", "payload": ranges},
        {"field": "data.sizes", "payload": sizes},
        {"field": "state.selectedTags", "payload": [x['class_name'] for x in class_balance]}
    ]


    init_classes_switches(state, class_balance)

    reset_progress(progress_index)
    g.api.app.set_fields(g.task_id, fields)



@g.my_app.callback("use_tags")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def use_tags(api: sly.Api, task_id, context, state, app_logger):
    global selected_tags
    selected_tags = state["selectedTags"]

    fields = [
        {"field": "data.done3", "payload": True},
        {"field": "state.collapsed4", "payload": False},
        {"field": "state.disabled4", "payload": False},
        {"field": "state.activeStep", "payload": 4},
    ]
    g.api.app.set_fields(g.task_id, fields)