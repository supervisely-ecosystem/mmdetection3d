import datetime

import numpy as np
import supervisely_lib as sly


import sys
sys.path.append('../train/src')
import sly_globals as g
from sly_train_progress import add_progress_to_request
from supervisely_lib import logger

class SlyDashboardLogger:
    def __init__(self):
        if not g.inference:
            self._lrs = []
            self.time_sec_tot = datetime.time()
            self.max_iters = g.api.app.get_field(g.task_id, "state.steps_per_epoch_train")
            self.batch_size = g.api.app.get_field(g.task_id, "state.batchSizeTrain")
            self.progress_epoch = sly.Progress("Epochs", g.api.app.get_field(g.task_id, "state.epochs"))
            self.progress_iter = sly.Progress("Iterations", int(self.max_iters / self.batch_size) )
            self.acc_tables_bev = []
            self.acc_tables_3D = []

    @staticmethod
    def _loss_field_map(mode):
        mode = mode.capitalize()
        assert mode in ["Train", "Val"]

        loss_field_map = {
            "loss_cls": f"chart{mode}LossCls",
            "loss_bbox": f"chart{mode}LossBbox",
            "loss_dir": f"chart{mode}LossDir",
            "loss_sum": f"chart{mode}LossSum"
        }
        return loss_field_map

    def log(self, mode=None, epoch=None, iter=None, loss=None):
        fields = []

        if epoch:
            self.progress_epoch.set_current_value(epoch)
        if iter and mode != 'val':
            self.progress_iter.set_current_value(iter + 1)
        if mode == 'val':
            self.progress_iter.set_current_value(0)
            fields.extend([{"field": "state.curEpochAcc", "payload": self.progress_epoch.current}])


        add_progress_to_request(fields, "Epoch", self.progress_epoch)
        add_progress_to_request(fields, "Iter", self.progress_iter)

        epoch_float = float(self.progress_epoch.current) + \
                      float(self.progress_iter.current) / float(self.progress_iter.total)

        lfm = self._loss_field_map(mode)
        all_losses = np.array(list(loss.values())[:3])
        sum_losses = round(float(sum(all_losses.T[-1])), 6)
        loss['loss_sum'] = [sum_losses]

        for loss_name, loss_value in loss.items():
            try:
                field_name =  f"data.{lfm[loss_name]}.series[0].data"
            except KeyError:
                continue
            loss_value =  round(float(loss_value[-1]),6)
            fields.extend([{"field": field_name, "payload": [[epoch_float, loss_value]], "append": True}])

        g.api.app.set_fields(g.task_id, fields)


    def submit_map_table(self, name, difficulties, classes, ap):
        acc_table = []

        for i, c in enumerate(classes):
            colnames = [f"difc_{x}" for x in difficulties]
            accs = [round(float(x), 4)  if not np.isnan(x) else -1.0 for x in ap[i, :, 0]]
            accs = zip(colnames, accs)
            row = {"class": c}
            row = dict(row, **dict(accs))
            acc_table.append(row)
        # overall = "Overall: {:.2f}".format(np.mean(ap[:, -1])) TODO: Display ovearall mAP

        if name == "3D":
            self.acc_tables_3D.append(acc_table)
            g.api.app.set_field(g.task_id, "data.acc3DTable", self.acc_tables_3D)
        else:
            self.acc_tables_bev.append(acc_table)
            g.api.app.set_field(g.task_id, "data.accBevTable", self.acc_tables_bev)




