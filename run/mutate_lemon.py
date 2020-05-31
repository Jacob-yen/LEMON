# -*-coding:UTF-8-*-
from scripts.logger.lemon_logger import Logger
from scripts.tools.mutator_selection_logic import MCMC, Roulette
import argparse
import sys
import ast
import os
import numpy as np
from itertools import combinations
import redis
import pickle
from scripts.tools import utils
import shutil
import re
import datetime
import configparser
import warnings

np.random.seed(20200501)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

"""Load Configuration"""

def check_has_NaN(predictions,bk_num):
    """
    Check if there is NAN in the result
    """
    def get_NaN_num(nds):
        _nan_num = 0
        for nd in nds:
            if np.isnan(nd).any():
                _nan_num += 1
        return _nan_num

    # Three Backends
    if len(predictions) == bk_num:
        for t in zip(*predictions):
            nan_num = get_NaN_num(t)
            if 0 < nan_num < bk_num:
                return True
            else:
                continue
        return False
    else:
        raise Exception("wrong backend amounts")


def get_mutate_time(seedname):
    regex_str = seedname[:-3]
    match = re.search(r"\d+$",regex_str)
    return int(match.group())


def save_mutate_history(mcmc:MCMC,invalid_history:dict,mutant_history:list):
    mutator_history_path = os.path.join(experiment_dir,"mutator_history.csv")
    mutant_history_path = os.path.join(experiment_dir,"mutant_history.txt")
    with open(mutator_history_path,"w+") as fw:
        fw.write("Name,Success,Invalid,Total\n")
        for op in invalid_history.keys():
            mtrs = mcmc.mutators[op]
            invalid_cnt = invalid_history[op]
            fw.write("{},{},{},{}\n".format(op,mtrs.delta_bigger_than_zero,invalid_cnt,mtrs.total))
    with open(mutant_history_path,"w+") as fw:
        for mutant in mutant_history:
            fw.write("{}\n".format(mutant))

def _generate_and_predict(res_dict,filename,mutate_num,mutate_ops,test_size,exp,backends):
    """
    Generate models using mutate operators and store them
    """
    mutate_op_history = { k:0 for k in mutate_ops}
    mutate_op_invalid_history = {k: 0 for k in mutate_ops}
    mutant_history = []

    origin_model_name = "{}_origin0.h5".format(exp)
    origin_save_path = os.path.join(mut_dir,origin_model_name)
    shutil.copy(src=filename,dst=origin_save_path)
    _,res_dict,inconsistency, _ = _get_model_prediction(res_dict,origin_save_path,origin_model_name,exp,test_size,backends)
    mcmc = MCMC(mutate_ops)
    roulette = Roulette([origin_model_name])
    last_used_mutator = None
    last_inconsistency = inconsistency

    mutant_counter = 0

    while mutant_counter < mutate_num:
        picked_seed = utils.ToolUtils.select_mutant(roulette)
        selected_op = utils.ToolUtils.select_mutator(mcmc, last_used_mutator=last_used_mutator)
        mutate_op_history[selected_op] += 1
        last_used_mutator = selected_op
        mutator = mcmc.mutators[selected_op]
        mutant = roulette.mutants[picked_seed]
        # mutator.total += 1
        mutant.selected += 1

        new_seed_name = "{}-{}{}.h5".format(picked_seed[:-3],selected_op,mutate_op_history[selected_op])
        if new_seed_name not in roulette.mutants.keys():
            new_seed_path = os.path.join(mut_dir, new_seed_name)
            picked_seed_path = os.path.join(mut_dir,picked_seed)
            mutate_st = datetime.datetime.now()
            mutate_status = os.system("{}/lemon/bin/python -m  scripts.mutation.model_mutation_generators --model {} "
                                      "--mutate_op {} --save_path {} --mutate_ratio {}".format(python_prefix,picked_seed_path, selected_op,
                                                                             new_seed_path,flags.mutate_ratio))
            mutate_et = datetime.datetime.now()
            mutate_dt = mutate_et - mutate_st
            h, m, s = utils.ToolUtils.get_HH_mm_ss(mutate_dt)
            mutate_logger.info("INFO:Mutate Time Used on {} : {}h, {}m, {}s".format(selected_op, h, m, s))

            if mutate_status == 0:
                mutate_logger.info("INFO: Mutation progress {}/{}".format(mutant_counter+1,mutate_num))
                predict_status,res_dict,inconsistency,model_outputs = _get_model_prediction(res_dict,new_seed_path,new_seed_name,exp,test_size,backends)

                mutator.total += 1
                if predict_status :
                    #mutant_counter += 1
                    mutant_history.append(new_seed_name)
                    if utils.ModelUtils.is_valid_model(model_outputs):
                        mutant_counter += 1
                        # The sum of the values of the inconsistency boost of the new model on the three backends
                        delta = 0
                        # for every backend
                        for key in inconsistency.keys():
                            # compare with last time
                            delta += inconsistency[key] - last_inconsistency[key]
                        # if sum of increments on three backends is greater than zero
                        # then add it into seed pool
                        if delta > 0:
                            mutator.delta_bigger_than_zero += 1
                            if roulette.pool_size >= pool_size:
                                roulette.pop_one_mutant()
                            roulette.add_mutant(new_seed_name)
                        else:
                            mutate_logger.warning("WARN: {} would not be put into pool".format(new_seed_name))
                        last_inconsistency = inconsistency
                        mutate_logger.info("SUCCESS:{} pass testing!".format(new_seed_name))
                    else:
                        mutate_op_invalid_history[selected_op] += 1
                        mutate_logger.error("ERROR: invalid model Found!")
                else:
                    mutate_logger.error("ERROR:Crashed or NaN model Found!")
            else:
                mutate_logger.error("ERROR:Exception raised when mutate {} with {}".format(picked_seed,selected_op))
            mutate_logger.info("Mutated op used history:")
            mutate_logger.info(mutate_op_history)

            mutate_logger.info("Invalid mutant generated history:")
            mutate_logger.info(mutate_op_invalid_history)

    save_mutate_history(mcmc,mutate_op_invalid_history,mutant_history)
    return res_dict


def generate_metrics_result(res_dict,predict_output,model_idntfr):
    mutate_logger.info("INFO: Generating Metrics Result")
    inconsistency_score = {}
    for pair in combinations(predict_output.items(), 2):
        bk_prediction1, bk_prediction2 = pair[0], pair[1]
        bk1, prediction1 = bk_prediction1[0], bk_prediction1[1]
        bk2, prediction2 = bk_prediction2[0], bk_prediction2[1]
        bk_pair = "{}_{}".format(bk1, bk2)
        for metrics_name, metrics_result_dict in res_dict.items():
            metrics_func = utils.MetricsUtils.get_metrics_by_name(metrics_name)

            if metrics_name == 'D_MAD':
                deltas = metrics_func(prediction1, prediction2, y_test[:flags.test_size])
                inconsistency_score[bk_pair] = sum(deltas)
                for i, delta in enumerate(deltas):
                    dk = "{}_{}_{}_input{}".format(model_idntfr, bk1, bk2, i)
                    metrics_result_dict[dk] = delta

    mutate_logger.info(inconsistency_score)
    return True, res_dict,inconsistency_score, predict_output


def _get_model_prediction(res_dict,model_path,model_name,exp,test_size,backends):
    """
    Get model prediction on different backends and calculate distance by metrics
    """
    predict_output = {b: [] for b in backends}
    predict_status = set()
    model_idntfr = model_name[:-3]

    for bk in backends:
        python_bin = f"{python_prefix}/{bk}/bin/python"
        predict_st = datetime.datetime.now()
        pre_status_bk = os.system(f"{python_bin} -u -m run.patch_prediction_extractor --backend {bk} "
                                  f"--exp {exp} --test_size {test_size} --model {model_path} "
                                  f"--redis_db {lemon_cfg['redis'].getint('redis_db')} --config_name {flags.config_name}")
        predict_et = datetime.datetime.now()
        predict_td = predict_et - predict_st
        h, m, s = utils.ToolUtils.get_HH_mm_ss(predict_td)
        mutate_logger.info("INFO:Prediction Time Used on {} : {}h, {}m, {}s".format(bk,h,m,s))

        if pre_status_bk == 0:  # If no exception is thrown,save prediction result
            data = pickle.loads(redis_conn.hget("prediction_{}".format(model_name), bk))
            predict_output[bk] = data
        else:  # record the crashed backend
            mutate_logger.error("ERROR:{} crash on backend {} when predicting ".format(model_name,bk))
        predict_status.add(pre_status_bk)

    if 0 in predict_status and len(predict_status) == 1:
        """If all backends are working fine, check if there is NAN in the result"""
        predictions = list(predict_output.values())
        has_NaN = check_has_NaN(predictions,len(backends))

        if has_NaN:
            nan_model_path = os.path.join(nan_dir, model_name)
            mutate_logger.error("Error: move NAN model")
            shutil.move(model_path, nan_model_path)
            return False, res_dict,None, None
        else:
            mutate_logger.info("INFO: Saving prediction")
            with open("{}/prediction_{}.pkl".format(inner_output_dir, model_idntfr), "wb+") as f:
                pickle.dump(predict_output, file=f)
            with open("{}/patch_prediction_{}.pkl".format(inner_output_dir, model_idntfr), "wb+") as f:
                pickle.dump(predict_output, file=f)
            return generate_metrics_result(res_dict=res_dict,predict_output=predict_output,model_idntfr=model_idntfr)

    else:  # record the crashed model
        mutate_logger.error("Error: move crash model")
        crash_model_path = os.path.join(crash_dir, model_name)
        shutil.move(model_path, crash_model_path)
        return False, res_dict,None, None


if __name__ == "__main__":

    starttime = datetime.datetime.now()
    """Parser of command args"""
    parse = argparse.ArgumentParser()
    parse.add_argument("--is_mutate", type=ast.literal_eval, default=False,
                       help="parameter to determine mutation option")
    parse.add_argument("--mutate_op", type=str, nargs='+',
                       choices=['WS', 'GF', 'NEB', 'NAI', 'NS', 'ARem', 'ARep', 'LA', 'LC', 'LR', 'LS','MLA']
                       , help="parameter to determine mutation option")
    parse.add_argument("--model", type=str, help="relative path of model file(from root dir)")
    parse.add_argument("--output_dir", type=str, help="relative path of output dir(from root dir)")
    parse.add_argument("--backends", type=str, nargs='+', help="list of backends")
    parse.add_argument("--mutate_num", type=int, help="number of variant models generated by each mutation operator")
    parse.add_argument("--mutate_ratio", type=float, help="ratio of mutation")
    parse.add_argument("--exp", type=str, help="experiments identifiers")
    parse.add_argument("--test_size", type=int, help="amount of testing image")
    parse.add_argument("--config_name", type=str, help="config name")
    flags, unparsed = parse.parse_known_args(sys.argv[1:])

    warnings.filterwarnings("ignore")
    lemon_cfg = configparser.ConfigParser()
    lemon_cfg.read(f"./config/{flags.config_name}")

    mutate_logger = Logger()
    pool = redis.ConnectionPool(host=lemon_cfg['redis']['host'], port=lemon_cfg['redis']['port'],db=lemon_cfg['redis'].getint('redis_db'))
    redis_conn = redis.Redis(connection_pool=pool)

    for k in redis_conn.keys():
        if flags.exp in k.decode("utf-8"):
            redis_conn.delete(k)

    experiment_dir = os.path.join(flags.output_dir,flags.exp)  # exp : like lenet5-mnist
    mut_dir = os.path.join(experiment_dir, "mut_model")
    crash_dir = os.path.join(experiment_dir, "crash")
    nan_dir = os.path.join(experiment_dir, "nan")
    inner_output_dir = os.path.join(experiment_dir,"inner_output")
    metrics_result_dir = os.path.join(experiment_dir,"metrics_result")

    x, y = utils.DataUtils.get_data_by_exp(flags.exp)
    x_test, y_test = x[:flags.test_size], y[:flags.test_size]
    pool_size = lemon_cfg['parameters'].getint('pool_size')
    python_prefix = lemon_cfg['parameters']['python_prefix'].rstrip("/")

    try:
        metrics_list = lemon_cfg['parameters']['metrics'].split(" ")
        lemon_results = {k: dict() for k in metrics_list}
        lemon_results = _generate_and_predict(lemon_results,flags.model,flags.mutate_num,flags.mutate_op,
                                               flags.test_size,flags.exp,flags.backends)
        with open("{}/{}_lemon_results.pkl".format(experiment_dir,flags.exp),"wb+") as f:
            pickle.dump(lemon_results, file=f)
        utils.MetricsUtils.generate_result_by_metrics(metrics_list,lemon_results,metrics_result_dir,flags.exp)

    except Exception as e:
        mutate_logger.exception(sys.exc_info())

    from keras import backend as K
    K.clear_session()

    endtime = datetime.datetime.now()
    time_delta = endtime - starttime
    h,m,s = utils.ToolUtils.get_HH_mm_ss(time_delta)
    mutate_logger.info("INFO:Mutation process is done: Time used: {} hour,{} min,{} sec".format(h,m,s))
