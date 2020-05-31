"""
# Part  of localization phase
"""
import os
import sys
import math
from itertools import combinations
import configparser
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
start_time = datetime.now()
distance_threshold = 0.005


def get_all_inputs():
    """
    divide inconsistencies into different backends pair
    """
    ultimate_localization_inconststency = {exp: {bk_pr: list() for bk_pr in backend_pairs} for exp in exps}
    print(ultimate_localization_inconststency)
    for exp in exps:
        exp_dir = os.path.join(output_dir, exp)
        metrics_dir = os.path.join(exp_dir, "metrics_result")

        exp_metrics_path = os.path.join(metrics_dir, "{}_D_MAD_result.csv".format(exp))
        metrics_result = {}
        with open(exp_metrics_path, "r") as fr:
            lines = fr.readlines()[1:]
            for line in lines:
                line_split = line.split(",")
                # incon_idntfr like mobilenet.1.00.224-imagenet_origin0_theano_cntk_input1494
                incon_idntfr, incon_value = line_split[0], float(line_split[1])
                metrics_result[incon_idntfr] = incon_value

        for incon_idntfr in metrics_result.keys():
            incon_idntfr_splits = incon_idntfr.split("_")
            bks_pair = "{}_{}".format(incon_idntfr_splits[2], incon_idntfr_splits[3])
            if bks_pair in backend_pairs:
                incon_tuple = (incon_idntfr, metrics_result[incon_idntfr])
                ultimate_localization_inconststency[exp][bks_pair].append(incon_tuple)
    return ultimate_localization_inconststency


def set_calculation(incons: list):
    origin_incons = dict()
    mutated_incons = dict()
    for incon in incons:
        # incon_idntfr like mobilenet.1.00.224-imagenet_origin0_theano_cntk_input1494
        incon_idntfr, incon_value = incon[0].replace("\n", ""), incon[1]
        incon_tuple = (incon_idntfr, incon_value)
        if not math.isnan(incon_value) and incon_value >= threshold:
            incon_idntfr_splits = incon_idntfr.split("_")
            # input_key = incon_idntfr_splits[-1] #input1494
            input_key = f"{incon_idntfr_splits[0]}_{incon_idntfr_splits[-1]}"  # lenet5-mnist_input1494
            if incon_idntfr_splits[1] == "origin0":
                origin_incons = add_into_dict(input_key, incon_tuple, origin_incons)
            else:
                mutated_incons = add_into_dict(input_key, incon_tuple, mutated_incons)

    mutated_greater = list()
    origin_greater = list()

    """mutated higher"""
    for ik, t in mutated_incons.items():
        if ik not in origin_incons.keys():
            mutated_greater.append(t)

    """origin higher"""
    for ik, t in origin_incons.items():
        if ik not in mutated_incons.keys():
            origin_greater.append(t)

    return list(origin_incons.values()), list(mutated_incons.values()), origin_greater, mutated_greater


def add_into_dict(input_key, incon_tuple, incon_dict):
    """
    Two step:
    0. under the same backends pair
    * 1. the same input, choose largest.
    2. different inputs with small distance. Do not update
    """
    if input_key not in incon_dict.keys() or incon_dict[input_key][1] < incon_tuple[1]:
        incon_dict[input_key] = incon_tuple
    return incon_dict


def close_incons_reduction(incons: list):
    """
    Two step:
    0. under the same backends pair
    1. the same input, choose largest.(done before)
    * 2. different inputs with small distance. Do not update(not used)
    """

    def is_duplicate(t: tuple, li: list):
        """unique inconsistency"""
        for l in li:
            if abs(t[1] - l[1]) <= distance_threshold:
                return True,l
        return False,None

    result = list()
    relation_dict = dict()
    for incon in incons:
        status, l = is_duplicate(incon, result)
        if not status:
            result.append(incon)
        else:
            relation_dict[incon] = l
    return result,relation_dict


def get_diff_set(a_list, b_list):
    """Get results of a - b"""
    a_dict = {tpl[0]: tpl for tpl in a_list}
    b_dict = {tpl[0]: tpl for tpl in b_list}
    result_set = list()
    for ik, t in a_dict.items():
        if ik not in b_dict.keys():
            result_set.append(t)
    return set(result_set)


def update_localize_model_inputs(exp, idntfrs: list, localizes: dict):
    for idntfr in idntfrs:
        # idntfr like mobilenet.1.00.224-imagenet_origin0_theano_cntk_input1494
        idntfr_splits = idntfr.split("_")
        model_input = "_".join([idntfr_splits[0], idntfr_splits[1], idntfr_splits[-1]])
        localizes[exp].add(model_input)
    return localizes


if __name__ == '__main__':
    # get experiments configuration

    config_name = sys.argv[1]
    lemon_cfg = configparser.ConfigParser()
    lemon_cfg.read(f"./config/{config_name}")
    parameters = lemon_cfg['parameters']
    output_dir = parameters['output_dir']
    output_dir = output_dir[:-1] if output_dir.endswith("/") else output_dir
    threshold = parameters.getfloat('threshold')
    current_container = output_dir.rstrip("/").split("/")[-1]
    backend_choices = [1,2,3]

    # you can try different threshold
    unique_inconsistencies_dict = set()
    total_model_inputs = dict()
    all_relations = dict()

    for backend_choice in backend_choices:
        if backend_choice == 1:
            backends = ['tensorflow', 'theano', 'cntk']
        elif backend_choice == 2:
            backends = ['tensorflow', 'theano', 'mxnet']
        else:
            backends = ['tensorflow', 'cntk', 'mxnet']
        print(current_container,backends)
        backends_str = "-".join(backends)
        backend_pairs = [f"{pair[0]}_{pair[1]}" for pair in combinations(backends, 2)]

        """Get all exps"""
        exps = parameters['exps'].lstrip().rstrip().split(" ")
        exps.sort(key=lambda x: x)
        compare_columns = ['M-O', 'O-M', 'O&M']
        localize_model_inputs = {exp: set() for exp in exps}
        unique_incons = dict()
        exp_analysis = {exp: {bkpair: list() for bkpair in backend_pairs} for exp in exps}

        """Generate unique inconsistency"""
        exp_inputs_dict = get_all_inputs()
        for exp_id, backends_incons in exp_inputs_dict.items():
            print("######{}######".format(exp_id))
            exp_dict = dict()
            for bk_pair, incons in backends_incons.items():
                print("------{}------".format(bk_pair))
                # a list of tuples. like(incon_idntfr,incon_value)
                origin_incons, mutated_incons, _, _ = set_calculation(incons)
                origin_set,_ = close_incons_reduction(origin_incons)
                mutated_set,_ = close_incons_reduction(mutated_incons)

                for incon in origin_set:
                    unique_inconsistencies_dict.add((bk_pair, 'O', incon))
                for incon in mutated_set:
                    unique_inconsistencies_dict.add((bk_pair, 'M', incon))

                localize_model_inputs = update_localize_model_inputs(exp_id, [t[0] for t in origin_set],
                                                                     localize_model_inputs)
                localize_model_inputs = update_localize_model_inputs(exp_id, [t[0] for t in mutated_set],
                                                                     localize_model_inputs)

        """print model_inputs to localize"""
        print("\n########Localize model_inputs##########")
        # check how many model input has been localized
        non_localized_cnt = localized_cnt = 0

        with open(os.path.join(output_dir, f"localize_model_inputs-{backends_str}.pkl"), "wb") as fw:
            pickle.dump(localize_model_inputs, fw)

        for exp_id, model_set in localize_model_inputs.items():
            if exp_id not in total_model_inputs.keys():
                total_model_inputs[exp_id] = set()
            for mi in model_set:
                total_model_inputs[exp_id].add(mi)

    for exp_id,mis in total_model_inputs.items():
        O_num,M_num = 0,0
        for mi in mis:
            if mi.split("_")[1] == 'origin0':
                O_num += 1
            else:
                M_num +=1
        print(f"{exp_id}: {len(mis)} O:{O_num} M:{M_num}")

    with open(os.path.join(output_dir, f"unique_inconsistencies.pkl"), "wb") as fw:
        pickle.dump(unique_inconsistencies_dict, fw)

end_time = datetime.now()
print("Time cost:",end_time-start_time)