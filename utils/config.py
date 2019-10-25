import yaml
from easydict import EasyDict as edict


def _get_default_config():
    c = edict()

    c.DATA_DIR = ''

    c.TRAIN_DIR = ''
    c.CSV = ''

    c.PARALLEL = False
    c.DEBUG = False
    c.DEBUG_IMAGE = False
    c.PRINT_EVERY = 1

    c.NUM_WORKERS = 2

    c.TRAIN = edict()
    c.TRAIN.BATCH_SIZE = 16
    c.TRAIN.NUM_EPOCHS = 100

    c.EVAL = edict()
    c.EVAL.BATCH_SIZE = 32

    c.DATA = edict()
    c.DATA.IMG_H = 256
    c.DATA.IMG_W = 256

    c.MODEL = edict()
    c.MODEL.NAME = ''
    c.MODEL.FC_TYPE = 0

    c.LOSS = edict()
    c.LOSS.NAME = 'mse'
    c.LOSS.PARAMS = edict()

    c.OPTIMIZER = edict()
    c.OPTIMIZER.NAME = 'adam'
    c.OPTIMIZER.LR = 0.001
    c.OPTIMIZER.PARAMS = edict()

    c.SCHEDULER = edict()
    c.SCHEDULER.NAME = 'none'
    c.SCHEDULER.PARAMS = edict()

    return c


def _merge_config(src, dst):
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


def load(config_path):
    with open(config_path, 'r') as fid:
        yaml_config = edict(yaml.load(fid))

    config = _get_default_config()
    _merge_config(yaml_config, config)

    return config
