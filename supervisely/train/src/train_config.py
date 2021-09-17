import os
import re
import supervisely_lib as sly

from mmcv.utils.config import Config

import sly_globals as g
import architectures
import augs

model_config_name = "model_config.py"
dataset_config_name = "dataset_config.py"
schedule_config_name = "schedule_config.py"
runtime_config_name = "runtime_config.py"
main_config_name = "train_config.py"

configs_dir = os.path.join(g.artifacts_dir, "configs")
model_config_path = os.path.join(configs_dir, model_config_name)
dataset_config_path = os.path.join(configs_dir, dataset_config_name)
schedule_config_path = os.path.join(configs_dir, schedule_config_name)
runtime_config_path = os.path.join(configs_dir, runtime_config_name)
main_config_path = os.path.join(configs_dir, main_config_name)

main_config_template = f"""
_base_ = [
    './{model_config_name}', './{dataset_config_name}',
    './{schedule_config_name}', './{runtime_config_name}'
]
"""

sly.fs.mkdir(configs_dir)


def _replace_function(var_name, var_value, template, match):
    m0 = match.group(0)
    m1 = match.group(1)
    return template.format(var_name, var_value)


def get_cfg_str(path_to_config):
    path2cfg = os.path.join(g.root_source_dir, path_to_config)
    with open(path2cfg) as f:
        model_config = f.read()
    return model_config


def generate_config(state):
    model_name = state["selectedModel"]
    model_info = architectures.get_model_info_by_name(model_name)

    lib_model_config_path = os.path.join(g.root_source_dir, model_info["config"])
    with open(lib_model_config_path) as f:
        py_config = f.read()

    py_config = py_config.replace('naiveSyncBN', 'BN')
    num_classes = g.project_meta.obj_classes.__len__()

    py_config = re.sub(r"num_classes*=(\d+),",
                       lambda m: _replace_function("num_classes", num_classes, "{}={},", m),  # num_tags
                       py_config, 0, re.MULTILINE)

    with open(model_config_path, 'w') as f:
        f.write(py_config)
    return model_config_path


def generate_dataset_config(state):
    config_path = os.path.join(g.root_source_dir, "supervisely/train/configs/dataset.py")
    if augs.augs_config_path is None:
        config_path = os.path.join(g.root_source_dir, "supervisely/train/configs/dataset_no_augs.py")
    with open(config_path) as f:
        py_config = f.read()

    if augs.augs_config_path is not None:
        py_config = re.sub(r"augs_config_path\s*=\s*(None)",
                           lambda m: _replace_function("augs_config_path", augs.augs_config_path, "{} = '{}'", m),
                           py_config, 0, re.MULTILINE)

    py_config = re.sub(r"batch_size_per_gpu\s*=\s*(\d+)",
                       lambda m: _replace_function("batch_size_per_gpu", state["batchSizePerGPU"], "{} = {}", m),
                       py_config, 0, re.MULTILINE)

    py_config = re.sub(r"num_workers_per_gpu\s*=\s*(\d+)",
                       lambda m: _replace_function("num_workers_per_gpu", state["workersPerGPU"], "{} = {}", m),
                       py_config, 0, re.MULTILINE)

    py_config = re.sub(r"validation_interval\s*=\s*(\d+)",
                       lambda m: _replace_function("validation_interval", state["valInterval"], "{} = {}", m),
                       py_config, 0, re.MULTILINE)

    # https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/hooks/evaluation.html#EvalHook
    save_best = None if state["saveBest"] is False else "'auto'"
    py_config = re.sub(r"save_best\s*=\s*([a-zA-Z]+)\s",
                       lambda m: _replace_function("save_best", save_best, "{} = {}\n", m),
                       py_config, 0, re.MULTILINE)

    py_config = re.sub(r"project_dir\s*=\s*(None)",
                       lambda m: _replace_function("project_dir", g.project_dir, "{} = '{}'", m),
                       py_config, 0, re.MULTILINE)

    py_config = re.sub(r"crop_size\s*=\s*(None)",
                       lambda m: _replace_function("crop_size", (state["imgSize"], state["imgSize"]), "{} = {}", m),
                       py_config, 0, re.MULTILINE)
    py_config = re.sub(r"img_scale\s*=\s*(\d+,\s\d+)",
                       lambda m: _replace_function("crop_size", (state["imgWidth"], state["imgHeight"]), "{} = {}", m),
                       py_config, 0, re.MULTILINE)
    # img_scale = (500, 300)
    with open(dataset_config_path, 'w') as f:
        f.write(py_config)
    return dataset_config_path, py_config


def generate_schedule_config(state):
    momentum = f"momentum={state['momentum']}, " if state['optimizer'] == 'SGD' else ''
    optimizer = f"optimizer = dict(type='{state['optimizer']}', " \
                f"lr={state['lr']}, " \
                f"{momentum}" \
                f"weight_decay={state['weightDecay']}" \
                f"{', nesterov=True' if (state['nesterov'] is True and state.optimizer == 'SGD') else ''})"

    grad_clip = f"optimizer_config = dict(grad_clip=None)"
    if state["gradClipEnabled"] is True:
        grad_clip = f"optimizer_config = dict(grad_clip=dict(max_norm={state['maxNorm']}))"

    lr_updater = ""
    if state["lrPolicyEnabled"] is True:
        py_text = state["lrPolicyPyConfig"]
        py_lines = py_text.splitlines()
        num_uncommented = 0
        for line in py_lines:
            res_line = line.strip()
            if res_line != "" and res_line[0] != "#":
                lr_updater += res_line
                num_uncommented += 1
        if num_uncommented == 0:
            raise ValueError(
                "LR policy is enabled but not defined, please uncomment and modify one of the provided examples")
        if num_uncommented > 1:
            raise ValueError("several LR policies were uncommented, please keep only one")

    runner = f"runner = dict(type='EpochBasedRunner', max_epochs={state['epochs']})"
    if lr_updater == "":
        lr_updater = "lr_config = dict(policy='fixed')"
    py_config = optimizer + os.linesep + \
                grad_clip + os.linesep + \
                lr_updater + os.linesep + \
                runner + os.linesep

    with open(schedule_config_path, 'w') as f:
        f.write(py_config)
    return schedule_config_path, py_config


def generate_runtime_config(state):
    config_path = os.path.join(g.root_source_dir, "supervisely/train/configs/runtime.py")
    with open(config_path) as f:
        py_config = f.read()

    # https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/hooks/checkpoint.html
    add_ckpt_to_config = []

    def _get_ckpt_arg(arg_name, state_flag, state_field, suffix=","):
        flag = True if state_flag is None else state[state_flag]
        if flag is True:
            add_ckpt_to_config.append(True)
            return f" {arg_name}={state[state_field]}{suffix}"
        return ""

    checkpoint = "checkpoint_config = dict({interval}{max_keep_ckpts}{save_last})".format(
        interval=_get_ckpt_arg("interval", None, "checkpointInterval"),
        max_keep_ckpts=_get_ckpt_arg("max_keep_ckpts", "maxKeepCkptsEnabled", "maxKeepCkpts"),
        save_last=_get_ckpt_arg("save_last", "saveLast", "saveLast", suffix=""),
    )
    py_config = re.sub(r"(checkpoint_config = dict\(interval=1\))",
                       lambda m: checkpoint,
                       py_config, 0, re.MULTILINE)

    # logger hook
    # https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/hooks/logger/text.html
    py_config = re.sub(r"log_interval\s*=\s*(\d+)",
                       lambda m: _replace_function("log_interval", state["metricsPeriod"], "{} = {}", m),
                       py_config, 0, re.MULTILINE)

    py_config = re.sub(r"load_from\s*=\s*(None)",
                       lambda m: _replace_function("load_from", architectures.local_weights_path, "{} = '{}'", m),
                       py_config, 0, re.MULTILINE)

    with open(runtime_config_path, 'w') as f:
        f.write(py_config)

    return runtime_config_path, py_config



def save_from_state(state):
    with open(state["trainConfigPath"], 'w') as f:
        f.write(state["trainConfigLines"])

def save_config(cfg, train_config_path):
    with open(train_config_path, 'w') as f:
        f.write(cfg)