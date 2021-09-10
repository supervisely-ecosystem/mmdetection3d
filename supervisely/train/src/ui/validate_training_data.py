from collections import defaultdict
import os
import supervisely_lib as sly
import sly_globals as g
import input_project
import math
import tags
from sly_train_progress import get_progress_cb

report = []
tags_count = {}
pointclouds_without_figures = []
final_tags = []
final_tags2images = defaultdict(lambda: defaultdict(list))

def init(data, state):
    data["done4"] = False
    state["collapsed4"] = True
    state["disabled4"] = True
    data["validationReport"] = None
    data["cntErrors"] = 0
    data["cntWarnings"] = 0
    data["report"] = None



def init_cache(progress_cb, selected_tags):
    global tags_count, pointclouds_without_figures
    pointclouds_without_figures.clear()
    tags_count.clear()
    project_fs = sly.PointcloudProject.read_single(g.project_dir)
    tag_names = []
    for dataset_fs in project_fs:
        for item_name in dataset_fs:
            item_path, related_images_dir, ann_path = dataset_fs.get_item_paths(item_name)
            ann_json = sly.io.json.load_json_file(ann_path)
            ann = sly.PointcloudAnnotation.from_json(ann_json, project_fs.meta)

            if len(ann.figures) == 0:
                pointclouds_without_figures.append(item_name)
            else:
                t_names = []
                for fig in ann.figures:
                    tag_name = fig.parent_object.obj_class.name
                    if tag_name in selected_tags:
                        t_names.append(tag_name)
                if len(t_names) == 0:
                    pointclouds_without_figures.append(item_name)
                else:
                    tag_names.extend(t_names)
            progress_cb(1)
    for tag_name in tag_names:
        tags_count[tag_name] = tag_names.count(tag_name)





@g.my_app.callback("validate_data")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def validate_data(api: sly.Api, task_id, context, state, app_logger):
    global tags_count, pointclouds_without_figures, train_size, val_size

    progress = get_progress_cb(4, "Calculate stats", g.project_info.items_count)
    init_cache(progress, state["selectedTags"])

    report.clear()
    final_tags.clear()
    final_tags2images.clear()

    report.append({
        "type": "info",
        "title": "Total figures in project",
        "count": sum(tags_count.values()),
        "description": None
    })

    for t, v in tags_count.items():
        report.append({
            "type": "info",
            "title": f'Class: "{t}"',
            "count": v,
            "description": None
        })



    report.append({
        "title": "Selected tags for training",
        "count": len(tags_count.items()),
        "type": "info",
        "description": None
    })

    report.append({
        "type": "info",
        "title": "Total pointclouds in project",
        "count": g.project_info.items_count,
    })

    report.append({
        "title": "Pointclouds without figures",
        "count": len(pointclouds_without_figures),
        "type": "warning" if len(pointclouds_without_figures) > 0 else "pass",
        "description": "Such Pointclouds don't have any figures, so they will ignored and will not be used for training."
    })


    final_pc_count = g.project_info.items_count - len(pointclouds_without_figures)
    report.append({
        "title": "Final pointclouds count",
        "count": final_pc_count,
        "type": "error" if final_pc_count == 0 else "pass",
        "description": "Number of pointclouds (train + val) after collisions removal"
    })



    train_size = g.api.app.get_field(g.task_id, "data.trainImagesCount")
    val_size = g.api.app.get_field(g.task_id, "data.valImagesCount")

    old_percentage = val_size / (train_size + val_size)
    new_val_size = math.floor(final_pc_count * old_percentage)
    new_train_size = final_pc_count - new_val_size

    sly.logger.warning(new_val_size)
    sly.logger.warning(new_train_size)

    report.append({
        "title": "Train set size",
        "count": new_train_size,
        "type": "error" if new_train_size < 1 else "pass",
        "description": "Size of training set after collisions removal"
    })
    report.append({
        "title": "Val set size",
        "count": new_val_size,
        "type": "error" if new_val_size < 1 else "pass",
        "description": "Size of validation set after collisions removal"
    })


    cnt_errors = 0
    cnt_warnings = 0
    for item in report:
        if item["type"] == "error":
            cnt_errors += 1
        if item["type"] == "warning":
            cnt_warnings += 1

    fields = [
        {"field": "data.report", "payload": report},
        {"field": "data.done4", "payload": True},
        {"field": "data.cntErrors", "payload": cnt_errors},
        {"field": "data.cntWarnings", "payload": cnt_warnings},
        {"field": "data.pointcloudsWithoutFigures", "payload": pointclouds_without_figures},
        {"field": "data.cntWarnings", "payload": cnt_warnings},
        {"field": "data.valImagesCount", "payload": new_val_size},
        {"field": "data.trainImagesCount", "payload": new_train_size}
    ]
    if cnt_errors == 0:
        # save selected tags

        # save splits
        #final_tags2images[tag_name][split].extend(_final_infos)

        fields.extend([
            {"field": "state.collapsed5", "payload": False},
            {"field": "state.disabled5", "payload": False},
            {"field": "state.activeStep", "payload": 5},
        ])
    g.api.app.set_fields(g.task_id, fields)

