import sys
import sly_globals as g
import train_config


def init_script_arguments(state):
    sys.argv.append("tf")
    #sys.argv.append(os.path.join(g.root_source_dir, "configs/resnet/resnet18_b16x8_cifar10.py"))
    sys.argv.extend(["--cfg_file", state['trainConfigPath']])
    sys.argv.extend(["--main_log_dir", g.checkpoints_dir])
    sys.argv.extend(["--pipeline", "ObjectDetection"])

    #sys.argv.extend(["--device", "cuda"])
    #sys.argv.extend(["--gpu-ids", state["gpusId"]]) # TODO: add gpus check
