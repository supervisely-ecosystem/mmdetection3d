import errno
import os
import requests
from pathlib import Path
import zipfile
import sly_globals as g
import supervisely_lib as sly
from sly_train_progress import get_progress_cb, reset_progress, init_progress

local_weights_path = None


def get_models_list():
    res = [
        {
            "config": "supervisely/train/configs/hv_pointpillars_secfpn_6x8_160e_sly-3d-3class.py",
            "model": "PointPillars"
        },
        {
            "config": "supervisely/train/configs/hv_ssn_secfpn_sbn-all_2x16_2x_sly.py",
            "model": "Shape Signature Network(SSN)"
        }
    ]
    _validate_models_configs(res)
    return res


def get_table_columns():
    return [
        {"key": "model", "title": "Model", "subtitle": None}
    ]


def get_model_info_by_name(name):
    models = get_models_list()
    for info in models:
        if info["model"] == name:
            return info
    raise KeyError(f"Model {name} not found")


def get_pretrained_weights_by_name(name):
    return get_model_info_by_name(name)["weightsUrl"]


def _validate_models_configs(models):
    res = []
    for model in models:
        train_config_path = os.path.join(g.root_source_dir, model["config"])
        if not sly.fs.file_exists(train_config_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), train_config_path)
        res.append(model)
    return res


def init(data, state):
    models = get_models_list()
    data["models"] = models
    data["modelColumns"] = get_table_columns()
    state["selectedModel"] = "PointPillars"
    state["weightsInitialization"] = "KITTI"
    state["collapsed6"] = True
    state["disabled6"] = True
    init_progress(6, data)

    state["weightsPath"] = ""
    data["done6"] = False


def restart(data, state):
    data["done6"] = False
    # state["collapsed6"] = True
    # state["disabled6"] = True


@g.my_app.callback("random_weights")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def random_weights(api: sly.Api, task_id, context, state, app_logger):
    model_config_example = get_model_info_by_name(state["selectedModel"])['config']


    fields = [
        {"field": "data.done6", "payload": True},
        {"field": "state.collapsed7", "payload": False},
        {"field": "state.disabled7", "payload": False},
        {"field": "state.activeStep", "payload": 7},
        {"field": "state.localWeightsPath", "payload": None},
        {"field": "state.modelConfigExample", "payload": model_config_example}
    ]
    g.api.app.set_fields(g.task_id, fields)

@g.my_app.callback("download_weights")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def download_weights(api: sly.Api, task_id, context, state, app_logger):
    global local_weights_path
    try:
        if state["weightsInitialization"] == "custom":
            weights_path_remote = state["weightsPath"]
            filename = sly.fs.get_file_name(weights_path_remote)

            local_weights_dir = os.path.join(g.my_app.data_dir, sly.fs.get_file_name(weights_path_remote))
            local_weights_path = os.path.join(local_weights_dir, sly.fs.get_file_name_with_ext(weights_path_remote))
            local_index_path  = os.path.join(local_weights_dir, filename + '.index')

            index_path_remote = os.path.join(os.path.dirname(weights_path_remote), filename + '.index')

            sly.fs.mkdir(local_weights_dir, remove_content_if_exists=True)

            file_info = g.api.file.get_info_by_path(g.team_id, weights_path_remote)
            if file_info is None:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), weights_path_remote)
            progress_cb = get_progress_cb(6, "Download weights", file_info.sizeb, is_size=True, min_report_percent=1)
            g.api.file.download(g.team_id, weights_path_remote, local_weights_path, g.my_app.cache, progress_cb)
            g.api.file.download(g.team_id, index_path_remote, local_index_path, g.my_app.cache)

            local_weights_path = os.path.join(local_weights_dir, filename)

            reset_progress(6)

        elif state["weightsInitialization"] == "KITTI":
            weights_url = get_pretrained_weights_by_name(state["selectedModel"])
            local_zip_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name_with_ext(weights_url))
            local_weights_path = os.path.join(g.my_app.data_dir, sly.fs.get_file_name(weights_url))
            if not sly.fs.file_exists(local_zip_path) and not sly.fs.file_exists(local_zip_path):
                response = requests.head(weights_url, allow_redirects=True)
                sizeb = int(response.headers.get('content-length', 0))
                progress_cb = get_progress_cb(6, "Download weights", sizeb, is_size=True, min_report_percent=1)
                sly.fs.download(weights_url, local_zip_path, g.my_app.cache, progress_cb)

                reset_progress(6)
            if sly.fs.file_exists(local_zip_path) and not sly.fs.file_exists(local_weights_path):
                with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(local_weights_path)

            local_weights_name = os.listdir(local_weights_path)[0].split('.')[0]
            local_weights_path = os.path.join(local_weights_path, local_weights_name)
        elif state["weightsInitialization"] == "random":
            local_weights_path = None
        else:
            raise KeyError(f'Wrong state: {state["weightsInitialization"]}')
        sly.logger.info("Pretrained weights has been successfully downloaded",
                        extra={"weights": local_weights_path})
    except Exception as e:
        reset_progress(6)
        raise e

    model_config_example = get_model_info_by_name(state["selectedModel"])['config']

    fields = [
        {"field": "data.done6", "payload": True},
        {"field": "state.collapsed7", "payload": False},
        {"field": "state.disabled7", "payload": False},
        {"field": "state.activeStep", "payload": 7},
        {"field": "state.localWeightsPath", "payload": local_weights_path},
        {"field": "state.modelConfigExample", "payload": model_config_example}
    ]
    g.api.app.set_fields(g.task_id, fields)