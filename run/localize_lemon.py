# -*-coding:UTF-8-*-
"""
# Part  of localization phase
"""
import argparse
import sys
import os
import pickle
import configparser
from scripts.tools.utils import ModelUtils
import keras
from keras.engine.input_layer import InputLayer
import warnings
import datetime
from scripts.logger.lemon_logger import Logger
import shutil
from itertools import combinations
import keras.backend as K
warnings.filterwarnings("ignore")


def is_lstm_not_exists(exp_id,output_id):
    if exp_id in ['lstm0-sinewave','lstm2-price'] and output_id in ['experiment4','experiment5']:
        return True
    else:
        return False


def get_HH_mm_ss(td):
    days,seconds = td.days,td.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600)//60
    secs = seconds % 60
    return hours,minutes,secs


def generate_report(localize_res,savepath):
    with open(savepath,"w+") as fw:
        for localize_header, value in localize_res.items():
            fw.write("current_layer, delta,Rl,previous_layer\n".format(localize_header))
            for layer_res in value:
                fw.write("{},{},{},{}\n".format(layer_res[0],layer_res[1],layer_res[2],layer_res[3]))


def localize(mut_model_dir,select_idntfr, exp_name,localize_tmp_dir,report_dir,backends):
    """
    select_idntfrs: lenet5-mnist_origin0_input17
    """
    # get layer_output for all models coming from specific exp on all backends
    identifier_split = select_idntfr.split("_")
    data_index = int(identifier_split[-1][5:])
    model_idntfr = "{}_{}".format(identifier_split[0], identifier_split[1])
    model_path = "{}/{}.h5".format(mut_model_dir, model_idntfr)
    #
    # # check if indntfr hasn't been localized
    # for bk1, bk2 in combinations(backends, 2):
    #     report_path = os.path.join(report_dir, "{}_{}_{}_input{}.csv".format(model_idntfr, bk1, bk2, data_index))
    #     # not exists; continue fo localize
    #     if not os.path.exists(report_path):
    #         break
    # # all file exist; return
    # else:
    #     mylogger.logger.info(f"{select_idntfr} has been localized")
    #     return

    for bk in backends:
        python_bin = f"{python_prefix}/{bk}/bin/python"
        return_stats = os.system(
            f"{python_bin} -u -m run.patch_hidden_output_extractor --backend {bk} --output_dir {output_dir} --exp {exp_name}"
            f" --model_idntfr {model_idntfr} --data_index {data_index} --config_name {config_name}")
        # assert return_stats==0,"Getting hidden output failed!"
        if return_stats != 0:
            mylogger.logger.info("Getting hidden output failed!")
            failed_list.append(select_idntfr)
            return
    mylogger.logger.info("Getting localization for {}".format(select_idntfr))
    model = keras.models.load_model(model_path, custom_objects=ModelUtils.custom_objects())
    for bk1, bk2 in combinations(backends, 2):
        local_res = {}
        local_res = get_outputs_divation_onbackends(model=model, backends=[bk1, bk2],
                                                    model_idntfr=model_idntfr, local_res=local_res,
                                                    data_index=data_index, localize_tmp_dir=localize_tmp_dir)
        mylogger.logger.info("Generating localization report for {} on {}-{}!".format(model_idntfr,bk1,bk2))
        report_path = os.path.join(report_dir, "{}_{}_{}_input{}.csv".format(model_idntfr,bk1,bk2, data_index))
        generate_report(local_res, report_path)
    del model
    K.clear_session()


def get_outputs_divation_onbackends(model,backends,model_idntfr,local_res,data_index,localize_tmp_dir):
    backend1 = backends[0]
    backend2 = backends[1]
    with open(os.path.join(localize_tmp_dir, "{}_{}_{}".format(model_idntfr, backend1,data_index)), "rb") as fr:
        model_layers_outputs_1 = pickle.load(fr)
    with open(os.path.join(localize_tmp_dir, "{}_{}_{}".format(model_idntfr, backend2,data_index)), "rb") as fr:
        model_layers_outputs_2 = pickle.load(fr)
    divations = ModelUtils.layers_divation(model, model_layers_outputs_1, model_layers_outputs_2)
    compare_res = []

    for i, layer in enumerate(model.layers):
        if isinstance(layer, InputLayer):
            continue
        delta, divation, inputlayers = divations[i]
        layer_compare_res = [layer.name, delta[0], divation[0],",".join(inputlayers)]  # batch accepted default
        compare_res.append(layer_compare_res)
    identifier = "{}_{}_{}_input_{}".format(model_idntfr,backend1,backend2,data_index)
    idntfr_localize = "{}_localize".format(identifier)
    local_res[idntfr_localize] = compare_res
    return local_res


if __name__ == "__main__":

    starttime = datetime.datetime.now()

    # get id of experiments
    config_name = sys.argv[1]
    lemon_cfg = configparser.ConfigParser()
    lemon_cfg.read(f"./config/{config_name}")
    parameters = lemon_cfg['parameters']

    output_dir = parameters['output_dir']
    output_dir = output_dir[:-1] if output_dir.endswith("/") else output_dir
    current_container = os.path.split(output_dir)[-1]
    python_prefix = parameters['python_prefix'].rstrip("/")

    """Initialization"""
    mylogger = Logger()
    backend_choices = [1,2,3]
    exps = parameters['exps'].lstrip().rstrip().split(" ")
    exps.sort(key=lambda x: x)
    all_model_inputs = {e:set() for e in exps}
    items_lists = list()
    for backend_choice in backend_choices:
        if backend_choice == 1:
            pre_backends = ['tensorflow', 'theano', 'cntk']
        elif backend_choice == 2:
            pre_backends = ['tensorflow', 'theano', 'mxnet']
        else:
            pre_backends = ['tensorflow', 'cntk', 'mxnet']
        backends_str = "-".join(pre_backends)
        backend_pairs = [f"{pair[0]}_{pair[1]}" for pair in combinations(pre_backends, 2)]

        with open(os.path.join(output_dir, f"localize_model_inputs-{backends_str}.pkl"), "rb") as fr:
            localize_model_inputs = pickle.load(fr)
            for exp_id,model_set in localize_model_inputs.items():
                if exp_id in exps:
                    for mi in model_set:
                        all_model_inputs[exp_id].add(mi)

    for exp,mi_set in all_model_inputs.items():
        print(exp,len(mi_set))
    failed_list = []
    """Print result of inconsistency distribution"""
    for exp_idntfr,model_inputs_set in all_model_inputs.items():
        if len(model_inputs_set) > 0:
            if exp_idntfr == 'inception.v3-imagenet' or exp_idntfr == 'densenet121-imagenet' or is_lstm_not_exists(exp_idntfr,current_container):
                # inception and densenet can't run on mxnet.
                # lstm can't run on mxnet before mxnet version 1.3.x
                backends = ['tensorflow', 'theano', 'cntk']
            else:
                backends = ['tensorflow', 'theano', 'cntk','mxnet']
            print("Localize for {} : {} left.".format(exp_idntfr,len(model_inputs_set)))
            mut_dir = os.path.join(output_dir,exp_idntfr, "mut_model")
            localization_dir = os.path.join(output_dir,exp_idntfr, "localization_result")
            localize_output_dir = os.path.join(output_dir,exp_idntfr, "localize_tmp")

            """make dir for hidden_output and localization dir """
            if not os.path.exists(localize_output_dir):
                os.makedirs(localize_output_dir)
            if not os.path.exists(localization_dir):
                os.makedirs(localization_dir)

            """Localization"""
            for idx,select_identifier in enumerate(model_inputs_set):
                print("{} of {} {}".format(idx,len(model_inputs_set),select_identifier))
                localize(mut_model_dir=mut_dir,select_idntfr=select_identifier,exp_name=exp_idntfr,
                         localize_tmp_dir=localize_output_dir,report_dir=localization_dir
                         ,backends=backends)

            shutil.rmtree(localize_output_dir)

    with open(os.path.join(output_dir, f"failed_idntfrs.txt"), "w") as fw:
        if len(failed_list) > 0:
            mylogger.logger.warning(f"{len(failed_list)} idntfrs fail to localize")
            lists = [f"{line} \n" for line in failed_list]
            fw.writelines(lists)
        else:
            mylogger.logger.info("all idntfrs localize successfully")

    endtime = datetime.datetime.now()
    time_delta = endtime - starttime
    h,m,s = get_HH_mm_ss(time_delta)
    mylogger.logger.info("Localization precess is done: Time used: {} hour,{} min,{} sec".format(h,m,s))





