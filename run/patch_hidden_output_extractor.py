# -*-coding:UTF-8-*-
"""
# Part  of localization phase
# get prediction for each backend
"""
import sys
import os
import pickle
import argparse
from scripts.tools.utils import DataUtils,ModelUtils
from scripts.logger.lemon_logger import Logger
import configparser
import warnings
import traceback
import numpy as np

np.random.seed(20200501)
warnings.filterwarnings("ignore")


def _get_hidden_output(test_data,backend,select_model,model_dir,data_index):
    """
        layers_output: list of ndarray which store outputs in each layer
        The result stored in redis like:
        (lenet5-mnist_origin0_theano,layers_output)
    """
    model_pathname = os.path.join(model_dir, "{}.h5".format(select_model))
    model = keras.models.load_model(model_pathname,custom_objects=ModelUtils.custom_objects())

    model_idntfr_backend = "{}_{}_{}".format(select_model, backend, data_index)
    select_data = np.expand_dims(test_data[data_index], axis=0)
    layers_output = ModelUtils.layers_output(model, select_data)
    with open(os.path.join(localize_output_dir,model_idntfr_backend),"wb") as fw:
        pickle.dump(layers_output,fw)


if __name__ == "__main__":

    """Parser of command args"""
    parse = argparse.ArgumentParser()
    parse.add_argument("--backend", type=str, help="name of backends")
    parse.add_argument("--exp", type=str, help="experiments identifiers")
    parse.add_argument("--output_dir", type=str, help="relative path of output dir(from root dir)")
    parse.add_argument("--data_index", type=int, help="redis db port")
    parse.add_argument("--config_name", type=str, help="config name")
    parse.add_argument("--model_idntfr", type=str, help="redis db port")
    flags, unparsed = parse.parse_known_args(sys.argv[1:])
    mylogger = Logger()

    """Load Configuration"""
    warnings.filterwarnings("ignore")
    lemon_cfg = configparser.ConfigParser()
    lemon_cfg.read(f"./config/{flags.config_name}")
    parameters = lemon_cfg['parameters']
    gpu_ids = parameters['gpu_ids']
    gpu_list = parameters['gpu_ids'].split(",")

    """Init cuda"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    warnings.filterwarnings("ignore")

    batch_size = 64
    """Switch backend"""
    bk_list = ['tensorflow', 'theano', 'cntk', 'mxnet']
    bk = flags.backend
    os.environ['KERAS_BACKEND'] = bk
    os.environ['PYTHONHASHSEED'] = '0'

    if bk == 'tensorflow':
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
        import tensorflow as tf
        mylogger.info(tf.__version__)

    if bk == 'theano':
        if len(gpu_list) == 2:
            os.environ[
                'THEANO_FLAGS'] = f"device=cuda,contexts=dev{gpu_list[0]}->cuda{gpu_list[0]};dev{gpu_list[1]}->cuda{gpu_list[1]}," \
                                  f"force_device=True,floatX=float32,lib.cnmem=1"
        else:
            os.environ['THEANO_FLAGS'] = f"device=cuda,contexts=dev{gpu_list[0]}->cuda{gpu_list[0]}," \
                                         f"force_device=True,floatX=float32,lib.cnmem=1"
        batch_size = 32
        import theano as th

        mylogger.info(th.__version__)
    if bk == "cntk":
        batch_size = 32
        from cntk.device import try_set_default_device, gpu
        try_set_default_device(gpu(int(gpu_list[0])))
        import cntk as ck
        mylogger.info(ck.__version__)

    if bk == "mxnet":
        batch_size = 32
        import mxnet as mxnet
        mylogger.info(mxnet.__version__)

    from keras import backend as K
    import keras
    mylogger.logger.info("Using {} as backend for states extraction| {} is wanted".format(K.backend(),bk))

    """Get model hidden output on selected_index data on specific backend"""
    try:
        backend_input_dict = {}
        localize_output_dir = os.path.join(flags.output_dir,flags.exp,"localize_tmp")
        x, y = DataUtils.get_data_by_exp(flags.exp)
        mut_dir = os.path.join(flags.output_dir,flags.exp,"mut_model")
        _get_hidden_output(test_data=x, backend=bk,select_model=flags.model_idntfr,model_dir=mut_dir,data_index=flags.data_index)
        mylogger.logger.info("Hidden output extracting done!")
    except:
        traceback.print_exc()
        sys.exit(-1)


