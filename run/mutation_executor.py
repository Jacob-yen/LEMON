# -*-coding:UTF-8-*-
import argparse
import sys
import os
from scripts.logger.lemon_logger import Logger
import warnings
import datetime
import configparser
from scripts.tools import utils

"""Init cuda"""
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
"""Setting random seed"""

if __name__ == '__main__':

    config_name = sys.argv[1]
    lemon_cfg = configparser.ConfigParser()
    lemon_cfg.read(f"./config/{config_name}")
    parameters = lemon_cfg['parameters']

    flags = argparse.Namespace(
        mutate_ops=parameters['mutate_ops'],
        exps=parameters['exps'].lstrip().rstrip().split(" "),
        origin_model_dir=parameters['origin_model_dir'],
        output_dir=parameters['output_dir'],
        backend=parameters['backend'],
        mutate_num=parameters.getint('mutate_num'),
        mutate_ratio=parameters.getfloat('mutate_ratio'),
        test_size=parameters.getint('test_size'),
        threshold=parameters.getfloat('threshold'),
        redis_db=lemon_cfg['parameters'].getint('redis_db'),
        python_prefix = parameters['python_prefix'].rstrip("/")
    )

    if not os.path.exists(flags.output_dir):
        os.makedirs(flags.output_dir)

    main_log = Logger()

    """Lemon process"""
    main_log.logger.info("Success: Lemon start successfully!")
    start_time = datetime.datetime.now()
    for exp_identifier in flags.exps:

        """Make directory"""
        experiment_dir = os.path.join(flags.output_dir, exp_identifier)  # exp : like lenet5-mnist
        mut_dir = os.path.join(experiment_dir, "mut_model")
        crash_dir = os.path.join(experiment_dir, "crash")
        nan_dir = os.path.join(experiment_dir, "nan")
        inner_output_dir = os.path.join(experiment_dir, "inner_output")
        metrics_result_dir = os.path.join(experiment_dir, "metrics_result")

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        if not os.path.exists(mut_dir):
            os.makedirs(mut_dir)
        if not os.path.exists(crash_dir):
            os.makedirs(crash_dir)
        if not os.path.exists(nan_dir):
            os.makedirs(nan_dir)
        if not os.path.exists(inner_output_dir):
            os.makedirs(inner_output_dir)
        if not os.path.exists(metrics_result_dir):
            os.makedirs(metrics_result_dir)

        try:
            """Mutate and get output of different backends"""
            main_log.info("INFO:Lemon mutation starting!")
            main_log.info("INFO:Lemon for exp: {}".format(exp_identifier))
            origin_model_name = "{}_origin.h5".format(exp_identifier)
            origin_model_file = os.path.join(flags.origin_model_dir,origin_model_name)
            mutate_lemon = "{}/lemon/bin/python -u -m run.mutate_lemon --mutate_op {} --model {} --output_dir {}" \
                            " --backends {} --mutate_num {} --mutate_ratio {} --exp {} --test_size {} --redis_db {} --config_name {}"\
                            .format(flags.python_prefix,flags.mutate_ops,origin_model_file,flags.output_dir,flags.backend,
                                    flags.mutate_num,flags.mutate_ratio,exp_identifier,flags.test_size,flags.redis_db,config_name)
            os.system(mutate_lemon)

        except Exception:
            main_log.error("Error: Lemon for exp:{} Failed!".format(exp_identifier))
            main_log.exception(sys.exc_info())

    end_time = datetime.datetime.now()
    time_delta = end_time - start_time
    h,m,s = utils.ToolUtils.get_HH_mm_ss(time_delta)
    main_log.info("INFO:Lemon is done: Time used: {} hour,{} min,{} sec".format(h,m,s))


