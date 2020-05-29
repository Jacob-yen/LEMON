import sys
import os
import numpy as np
import math
import pandas as pd
import warnings
import configparser
from itertools import combinations

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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

    pre_backends = parameters['backend'].split(" ")
    backends_str = "-".join(pre_backends)
    backend_pairs = [f"{pair[0]}_{pair[1]}" for pair in combinations(pre_backends, 2)]

    compare_columns = ["original","mutant(max)"]
    """Get all exps"""
    exps = parameters['exps'].lstrip().rstrip().split(" ")
    exps.sort(key=lambda x:x)
    backends_exp_indntfrs = {bk_p:{exp_id:list() for exp_id in exps} for bk_p in backend_pairs}
    for exp in exps:
        print('#########{}#########'.format(exp))
        exp_dir = os.path.join(output_dir, exp)
        metrics_dir = os.path.join(exp_dir, "metrics_result")
        exp_metrics_path = os.path.join(metrics_dir, "{}_D_MAD_result.csv".format(exp))
        with open(exp_metrics_path, "r") as fr:
            lines = fr.readlines()[1:]
            for line in lines:
                line_split = line.split(",")
                # incon_idntfr like mobilenet.1.00.224-imagenet_origin0_theano_cntk_input1494
                incon_idntfr, incon_value = line_split[0], float(line_split[1])
                # print(incon_value)
                incon_idntfr_split = incon_idntfr.split("_")
                incon_tuple = (incon_idntfr, incon_value )
                backend_pair,input_key = "{}_{}".format(incon_idntfr_split[2],incon_idntfr_split[3]),incon_idntfr_split[-1]
                if not math.isnan(incon_value):
                    backends_exp_indntfrs[backend_pair][exp].append(incon_tuple)

    all_inputs_name = {bk_p: {exp_id: set() for exp_id in exps} for bk_p in backend_pairs}

    for bk_pair in backend_pairs:
        exp_idntfrs = backends_exp_indntfrs[bk_pair]
        for exp_id,incon_idntfrs in exp_idntfrs.items():
            for incon_tuple in incon_idntfrs:
                (incon_idntfr, incon_value) = incon_tuple
                incon_idntfr_split = incon_idntfr.split("_")
                input_k = incon_idntfr_split[-1]
                model_type = incon_idntfr_split[1]
                if incon_value >= threshold:
                    all_inputs_name[bk_pair][exp_id].add(input_k)

    for bk_pair in backend_pairs:
        exp_idntfrs = backends_exp_indntfrs[bk_pair]
        exp_analysis = {}
        for exp_id,incon_idntfrs in exp_idntfrs.items():
            exp_analysis[exp_id] = {input_name:[0,0] for input_name in all_inputs_name[bk_pair][exp_id]}
            for incon_tuple in incon_idntfrs:
                (incon_idntfr, incon_value) = incon_tuple
                incon_idntfr_split = incon_idntfr.split("_")
                input_k = incon_idntfr_split[-1]
                if input_k in exp_analysis[exp_id].keys():
                    model_type = incon_idntfr_split[1]
                    if model_type == "origin0":
                        exp_analysis[exp_id][input_k][0] = incon_value
                    else:
                        if incon_value > exp_analysis[exp_id][input_k][1]:
                            exp_analysis[exp_id][input_k][1] = incon_value

            input_keys = list(exp_analysis[exp_id].keys())
            input_keys.sort()
            data = np.zeros((len(input_keys), 2), dtype=np.float)
            tuples = []
            df = pd.DataFrame(data, columns=compare_columns,index=input_keys)

            for input_idx in input_keys:
                df.loc[input_idx,compare_columns[0]] = exp_analysis[exp_id][input_idx][0]
                df.loc[input_idx,compare_columns[1]] = exp_analysis[exp_id][input_idx][1]

            # print(df)
            save_dir = os.path.join(output_dir, exp_id,"{}_inter".format(exp_id))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            df.to_csv(os.path.join(save_dir,"{}.csv".format(bk_pair)))