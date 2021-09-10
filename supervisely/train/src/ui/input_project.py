import sly_globals as g
from sly_train_progress import get_progress_cb, reset_progress, init_progress
import supervisely_lib as sly

project_fs: sly.Project = None



def init(data, state):
    data["projectId"] = g.project_info.id
    data["projectName"] = g.project_info.name
    data["projectImagesCount"] = g.project_info.items_count

    init_progress("InputProject", data)

    data["done1"] = False
    state["collapsed1"] = False


@g.my_app.callback("download_project")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download(api: sly.Api, task_id, context, state, app_logger):
    global project_fs
    if not sly.fs.dir_exists(g.project_dir):
        download_progress_project = get_progress_cb("InputProject", "Downloading project", g.project_info.items_count)
        sly.project.pointcloud_project.download_pointcloud_project(g.api, g.project_id, g.project_dir,
                                                                   download_items=True, log_progress=True)
        download_progress_project(g.project_info.items_count)

    project_fs = sly.PointcloudProject.read_single(g.project_dir)  # TODO: not only single?

    fields = [
        {"field": "data.done1", "payload": True},
        {"field": "state.collapsed2", "payload": False},
        {"field": "state.disabled2", "payload": False},
        {"field": "state.activeStep", "payload": 2},
    ]
    g.api.app.set_fields(g.task_id, fields)
