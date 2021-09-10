import os
import supervisely_lib as sly
from sly_train_progress import init_progress
import sly_globals as g
#from scripts import run_pipeline


_open_lnk_name = "open_app.lnk"


def init(data, state):
    init_progress("Epoch", data)
    init_progress("Iter", data)
    init_progress("UploadDir", data)
    init_charts(data, state)

    state["collapsed9"] = True
    state["disabled9"] = True
    state["done9"] = False

    state["started"] = False

    data["outputName"] = None
    data["outputUrl"] = None

    state["curEpochAcc"] = None
    data["acc3DTable"] = []
    data["accBevTable"] = []


def restart(data, state):
    data["done9"] = False


def init_chart(title, names, xs, ys, smoothing=None, yrange=None, decimals=None, xdecimals=None):
    series = []
    for name, x, y in zip(names, xs, ys):
        series.append({
            "name": name,
            "data": [[px, py] for px, py in zip(x, y)]
        })
    result = {
        "options": {
            "title": title,
            #"groupKey": "my-synced-charts",
        },
        "series": series
    }
    if smoothing is not None:
        result["options"]["smoothingWeight"] = smoothing
    if yrange is not None:
        result["options"]["yaxisInterval"] = yrange
    if decimals is not None:
        result["options"]["decimalsInFloat"] = decimals
    if xdecimals is not None:
        result["options"]["xaxisDecimalsInFloat"] = xdecimals
    return result


def init_charts(data, state):
    data["chartTrainLossSum"] = init_chart("Loss Sum", names=["train"], xs=[[]], ys=[[]], smoothing=0.6, xdecimals=2)
    data["chartTrainLossBbox"] = init_chart("Loss Bbox", names=["train"], xs=[[]], ys=[[]], smoothing=0.6, xdecimals=2)
    data["chartTrainLossCls"] = init_chart("Loss Cls", names=["train"], xs=[[]], ys=[[]], smoothing=0.6, xdecimals=2)
    data["chartTrainLossDir"] = init_chart("Loss Dir", names=["train"], xs=[[]], ys=[[]], smoothing=0.6, xdecimals=2)

    data["chartValLossSum"] = init_chart("Val Loss Sum", names=["val"], xs=[[]], ys=[[]], smoothing=0.6, xdecimals=2)
    data["chartValLossBbox"] = init_chart("Val Loss Bbox", names=["val"], xs=[[]], ys=[[]], smoothing=0.6, xdecimals=2)
    data["chartValLossCls"] = init_chart("Val Loss Cls", names=["val"], xs=[[]], ys=[[]], smoothing=0.6, xdecimals=2)
    data["chartValLossDir"] = init_chart("Val Loss Dir", names=["val"], xs=[[]], ys=[[]], smoothing=0.6, xdecimals=2)

    data["chartTime"] = init_chart("Time", names=["time"], xs=[[]], ys=[[]], xdecimals=2)
    data["chartDataTime"] = init_chart("Data Time", names=["data_time"], xs=[[]], ys=[[]], xdecimals=2)
    data["chartMemory"] = init_chart("Memory", names=["memory"], xs=[[]], ys=[[]], xdecimals=2)
    state["smoothing"] = 0.6


def _save_link_to_ui(local_dir, app_url):
    # save report to file *.lnk (link to report)
    local_path = os.path.join(local_dir, _open_lnk_name)
    sly.fs.ensure_base_path(local_path)
    with open(local_path, "w") as text_file:
        print(app_url, file=text_file)


from sly_train_progress import _update_progress_ui
from sly_train_args import init_script_arguments
from functools import partial


def upload_artifacts_and_log_progress():
    _save_link_to_ui(g.artifacts_dir, g.my_app.app_url)

    def upload_monitor(monitor, api: sly.Api, task_id, progress: sly.Progress):
        if progress.total == 0:
            progress.set(monitor.bytes_read, monitor.len, report=False)
        else:
            progress.set_current_value(monitor.bytes_read, report=False)
        _update_progress_ui("UploadDir", g.api, g.task_id, progress)

    progress = sly.Progress("Upload directory with training artifacts to Team Files", 0, is_size=True)
    progress_cb = partial(upload_monitor, api=g.api, task_id=g.task_id, progress=progress)

    remote_dir = f"/artifacts/{g.task_id}_{g.project_info.name}"

    res_dir = g.api.file.upload_directory(g.team_id, g.artifacts_dir, remote_dir, progress_size_cb=progress_cb)
    return res_dir


@g.my_app.callback("train")
@sly.timeit
@g.my_app.ignore_errors_and_show_dialog_window()
def train(api: sly.Api, task_id, context, state, app_logger):
    try:
        sly.json.dump_json_file(state, os.path.join(g.info_dir, "ui_state.json"))


        exclude_items = g.api.app.get_field(g.task_id, "data.pointcloudsWithoutFigures")
        #convert(g.project_dir, exclude_items=exclude_items)


        #init_script_arguments(state)

        #run_pipeline.main()
        sly.logger.info("starting training...")
        raise NotImplementedError

        # hide progress bars and eta
        fields = [
            {"field": "data.progressEpoch", "payload": None},
            {"field": "data.progressIter", "payload": None},
            {"field": "data.eta", "payload": None},
        ]
        g.api.app.set_fields(g.task_id, fields)

        remote_dir = upload_artifacts_and_log_progress()
        file_info = api.file.get_info_by_path(g.team_id, os.path.join(remote_dir, _open_lnk_name))
        api.task.set_output_directory(task_id, file_info.id, remote_dir)

        # show result directory in UI
        fields = [
            {"field": "data.outputUrl", "payload": g.api.file.get_url(file_info.id)},
            {"field": "data.outputName", "payload": remote_dir},
            {"field": "state.done9", "payload": True},
            {"field": "state.started", "payload": False},
        ]
        g.api.app.set_fields(g.task_id, fields)
    except Exception as e:
        api.app.set_field(task_id, "state.started", False)
        raise e  # app will handle this error and show modal window

    # stop application
    g.my_app.stop()