# -*-coding:UTF-8-*-
"""get prediction for each backend
"""
import sys
import os
import redis
import pickle
import argparse
import configparser
from scripts.tools.utils import DataUtils
from scripts.logger.lemon_logger import Logger
import warnings

main_logger = Logger()


def custom_objects():

    def no_activation(x):
        return x

    def leakyrelu(x):
        import keras.backend as K
        return K.relu(x, alpha=0.01)

    objects = {}
    objects['no_activation'] = no_activation
    objects['leakyrelu'] = leakyrelu
    return objects


def _get_prediction(bk, x, y, model_path,batch_size):
    """
    Get prediction of models on different backends
    """
    test_x, test_y = x[:flags.test_size],y[:flags.test_size]
    predict_model = keras.models.load_model(model_path,custom_objects=custom_objects())
    # predict_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    main_logger.info("INFO:load model and compile done!")
    res = predict_model.predict(test_x,batch_size=batch_size)
    main_logger.info("SUCCESS:Get prediction for {} successfully on {}!".format(mut_model_name,bk))
    """Store prediction result to redis"""
    redis_conn.hset("prediction_{}".format(mut_model_name),bk,pickle.dumps(res))


if __name__ == "__main__":

    """Parser of command args"""
    parse = argparse.ArgumentParser()
    parse.add_argument("--backend", type=str, help="name of backends")
    parse.add_argument("--exp", type=str, help="experiments identifiers")
    parse.add_argument("--test_size", type=int, help="amount of testing image")
    parse.add_argument("--model", type=str, help="path of the model to predict")
    parse.add_argument("--redis_db", type=int)
    parse.add_argument("--config_name", type=str)
    flags, unparsed = parse.parse_known_args(sys.argv[1:])

    """Load Configuration"""
    warnings.filterwarnings("ignore")
    lemon_cfg = configparser.ConfigParser()
    lemon_cfg.read(f"./config/{flags.config_name}")
    pool = redis.ConnectionPool(host=lemon_cfg['redis']['host'], port=lemon_cfg['redis']['port'],db=flags.redis_db)
    redis_conn = redis.Redis(connection_pool=pool)

    parameters = lemon_cfg['parameters']
    gpu_ids = parameters['gpu_ids']
    gpu_list = parameters['gpu_ids'].split(",")

    """Init cuda"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    warnings.filterwarnings("ignore")

    batch_size= 32
    """Switch backend"""
    bk_list = ['tensorflow', 'theano', 'cntk','mxnet']
    bk = flags.backend
    os.environ['KERAS_BACKEND'] = bk
    os.environ['PYTHONHASHSEED'] = '0'
    if bk == 'tensorflow':
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
        import tensorflow as tf
        main_logger.info(tf.__version__)
        batch_size = 128
        import keras
    if bk == 'theano':
        if len(gpu_list) == 2:
            os.environ['THEANO_FLAGS'] = f"device=cuda,contexts=dev{gpu_list[0]}->cuda{gpu_list[0]};dev{gpu_list[1]}->cuda{gpu_list[1]}," \
                                         f"force_device=True,floatX=float32,lib.cnmem=1"
        else:
            os.environ['THEANO_FLAGS'] = f"device=cuda,contexts=dev{gpu_list[0]}->cuda{gpu_list[0]}," \
                                         f"force_device=True,floatX=float32,lib.cnmem=1"
        import theano as th
        import keras
        main_logger.info(th.__version__)
    if bk == "cntk":
        from cntk.device import try_set_default_device,gpu
        try_set_default_device(gpu(int(gpu_list[0])))
        import cntk as ck
        main_logger.info(ck.__version__)
        import keras

    if bk == "mxnet":
        import mxnet as mxnet
        main_logger.info(f"mxnet_version {mxnet.__version__}")
        import keras

        batch_size = 16
    from keras import backend as K


    try:
        """Get model prediction"""
        main_logger.info("INFO:Using {} as backend for states extraction| {} is wanted".format(K.backend(),bk))
        x, y = DataUtils.get_data_by_exp(flags.exp)
        mut_model_name = os.path.split(flags.model)[-1]
        _get_prediction(bk=bk, x=x, y=y, model_path=flags.model,batch_size=batch_size)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(-1)
