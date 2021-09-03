import functools
import os
import sys

sys.path.append('')
import supervisely_lib as sly

import sly_globals as g
import nn_utils



@sly.timeit
def get_weights():


    # # download weights
    # progress = sly.Progress("Downloading weights", 1, is_size=True, need_info_log=True)
    # local_path = os.path.join(my_app.data_dir, "weights.pt")


    print(g.modelWeightsOptions)
    print(g.pretrained_weights)
    print(g.custom_weights)

    if g.modelWeightsOptions == "pretrained":
        url = g.pretrained_weights
        final_weights = url
        print("final", final_weights)

    # elif modelWeightsOptions == "custom":
    #     final_weights = custom_weights
    #     configs = os.path.join(Path(custom_weights).parents[1], 'opt.yaml')
    #     configs_local_path = os.path.join(my_app.data_dir, 'opt.yaml')
    #     file_info = my_app.public_api.file.get_info_by_path(TEAM_ID, custom_weights)
    #     progress.set(current=0, total=file_info.sizeb)
    #     my_app.public_api.file.download(TEAM_ID, custom_weights, local_path, my_app.cache, progress.iters_done_report)
    #     my_app.public_api.file.download(TEAM_ID, configs, configs_local_path)
    # else:
    #     raise ValueError("Unknown weights option {!r}".format(modelWeightsOptions))
    #
    # # load model on device
    # model, half, device, imgsz, stride = load_model(local_path, device=DEVICE_STR)
    # meta = construct_model_meta(model)
    # sly.logger.info("Model has been successfully deployed")


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            request_id = kwargs["context"]["request_id"]
            g.my_app.send_response(request_id, data={"error": repr(e)})
        return value

    return wrapper


@g.my_app.callback("get_custom_inference_settings")
@sly.timeit
@send_error_data
def get_custom_inference_settings(api: sly.Api, task_id, context, state, app_logger):
    # TODO: it should be YML with comments
    info = {
        "threshold": "(Float[0.0, 1.0], default 0.3) Boxes with confidence less than the threshold will"
                     " be skipped in the response."
    }

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=info)


@g.my_app.callback("get_output_classes_and_tags")
@sly.timeit
@send_error_data
def get_output_classes_and_tags(api: sly.Api, task_id, context, state, app_logger):
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=g.meta.to_json())


@g.my_app.callback("get_session_info")
@sly.timeit
@send_error_data
def get_session_info(api: sly.Api, task_id, context, state, app_logger):
    info = {
        "app": "MMDetection 3D Detection Serve",
        "weights": g.remote_weights_path,
        "device": g.device,
        "session_id": task_id,
        "classes_count": len(g.meta.obj_classes),
    }
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=info)


def _inference(api, pointcloud_id, threshold=None):
    local_pointcloud_path = os.path.join(g.my_app.data_dir, sly.rand_str(15) + ".pcd")

    api.pointcloud.download_path(pointcloud_id, local_pointcloud_path)

    results = nn_utils.inference_model(g.model, local_pointcloud_path,
                                       thresh=threshold if threshold is not None else 0.3)
    sly.fs.silent_remove(local_pointcloud_path)
    return results


@g.my_app.callback("inference_pointcloud_id")
@sly.timeit
@send_error_data
def inference_pointcloud_id(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    results = _inference(api, state["pointcloud_id"], state.get("threshold"))
    request_id = context["request_id"]
    g.my_app.send_response(request_id, data={"results": results.to_json()})


@g.my_app.callback("inference_pointcloud_ids")
@sly.timeit
@send_error_data
def inference_pointcloud_ids(api: sly.Api, task_id, context, state, app_logger):
    app_logger.debug("Input data", extra={"state": state})
    results = []
    for pointcloud_id in state["pointcloud_ids"]:
        result = _inference(api, pointcloud_id, state.get("threshold"))
        results.append(result.to_json())

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data={"results": results})


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.team_id,
        "context.workspaceId": g.workspace_id,
#        "modal.state.slyFile": g.remote_weights_path,
        "device": g.device
    })

    get_weights()
    exit(1)
    nn_utils.download_model_and_configs()
    nn_utils.construct_model_meta()
    nn_utils.deploy_model()

    g.my_app.run()


if __name__ == "__main__":
    sly.main_wrapper("main", main)
