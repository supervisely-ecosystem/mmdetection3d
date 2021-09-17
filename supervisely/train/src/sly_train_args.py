import sys
import sly_globals as g
import train_config


def init_script_arguments(state):
    sys.argv.append(state['trainConfigPath'])
    sys.argv.extend(["--work-dir", g.checkpoints_dir])


    #sys.argv.extend(["--device", "cuda"])
    #sys.argv.extend(["--gpu-ids", state["gpusId"]]) # TODO: add gpus check
