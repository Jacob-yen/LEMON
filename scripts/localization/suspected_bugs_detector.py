"""
# Part  of localization phase
# suspected bug detection:
# 1. Tensorflow,Theano,CNTK
# 2. Tensorflow,Theano,MXNET
#
# voting process
# -> a. inconsistency -> error backend,error layer.
#    b. check error backend in new container(whether inconsistency disappears).
# """
#
import numpy as np
import os
import sys
import configparser
from scripts.tools.filter_bugs import filter_bugs
import pickle
import pandas as pd
from itertools import combinations,product
from datetime import datetime
np.random.seed(20200501)


def get_model_inputs(idntfrs):
    idntfrs_set = set()
    for idntfr in idntfrs:
        idntfr_splits = idntfr.split("_")
        model_input = "{}_{}_{}".format(idntfr_splits[0], idntfr_splits[1], idntfr_splits[-1])
        idntfrs_set.add(model_input)
    return idntfrs_set


def voted_by_inconsistency(res_dict):
    votes = {}
    for back_pair,incon_value in res_dict.items():
        if incon_value >= threshold:
            back1,back2 = back_pair.split("_")
            votes[back1] = votes.setdefault(back1, 0) + 1
            votes[back2] = votes.setdefault(back2, 0) + 1

    for bk, cnt in votes.items():
        if cnt == 2:
            return bk

def get_metrics_of_exp(exp_id,exp_dir):
    res_dict = dict()
    metrics_dir = os.path.join(exp_dir, "metrics_result")
    file_name = f"{exp_id}_D_MAD_result.csv"
    with open(os.path.join(metrics_dir, file_name), "r") as fr:
        fr.readline()
        for line in fr.readlines():
            line = line.rstrip("\n").split(",")
            res_dict[line[0]] = float(line[1])
    return res_dict


def get_metrics_of_model_input(model_input,backend_pairs,exp_metrics_dict):
    mis = model_input.split("_")
    res_dict = dict()
    for backend_pair in backend_pairs:
        model_idntfr = f"{mis[0]}_{mis[1]}_{backend_pair}_{mis[-1]}"
        res_dict[backend_pair] = exp_metrics_dict[model_idntfr]
    return res_dict


def choose_error_layer(localize_res: dict):
    def get_common_layer(res, error_layer):
        bk_dict = {}
        for bk_pair, tu in res.items():
            if tu[0] == error_layer:
                bk1, bk2 = bk_pair.split("_")[0], bk_pair.split("_")[1]
                bk_dict[bk1] = bk_dict.setdefault(bk1, 0) + 1
                bk_dict[bk2] = bk_dict.setdefault(bk2, 0) + 1
        for bk, cnt in bk_dict.items():
            if cnt == 2:
                return bk

    layers_list = list(localize_res.values())
    unique_layer_dict = dict()
    for layer_tuple in layers_list:
        unique_layer_dict[layer_tuple[0]] = unique_layer_dict.setdefault(layer_tuple[0], 0) + 1
    unique_layer_dict = list(unique_layer_dict.items())
    if len(unique_layer_dict) == 1:
        return unique_layer_dict[0][0], "-".join(backends)
    if len(unique_layer_dict) == 2:
        error_layer = unique_layer_dict[0][0] if unique_layer_dict[0][1] == 2 else unique_layer_dict[1][0]
        return error_layer, get_common_layer(localize_res, error_layer)
    if len(unique_layer_dict) == 3:
        return None, None


def get_layer_values(bk_res: dict, layer_name):
    values = list()
    for bk_p in backend_pairs:
        df = bk_res[bk_p]
        select_row = df[df['current_layer'].isin([layer_name])]
        values.append("{}:{}".format(bk_p, select_row['Rl'].values[0]))
    return "|".join(values)


def get_rate(value:str):
    """
    v:values:'tensorflow_theano:325317.28125-tensorflow_cntk:325317.28125-tensorflow_mxnet:325317.28125-theano_cntk:0.07708668-theano_mxnet:0.09217975-cntk_mxnet:0.0887682'
    rate: max_Rl
    """
    if 'inf' in value:
        return 'inf'
    else:
        try:
            value_splits = value.split("|")
            value_list = [abs(float(val.split(":")[1])) for val in value_splits]
        except ValueError as e:
            print(value)
            raise e
        max_rl,min_rl = max(value_list),min(value_list)
        return max_rl / (min_rl + 1e-10)


def update_suspected_bugs(res_dict:dict, row:dict):
    """
    select suspected bugs from inconsistencies by their rate
    rate: max_Rl / min_Rl
    row is like: {error_backend:theano,error_layer:conv2d_copy_LA1,
                  model_input:'alexnet-cifar10_origin0-NAI1-LS6-WS4-NS1-ARep8_input228',
                  values:'tensorflow_theano:325317.28125-tensorflow_cntk:325317.28125-tensorflow_mxnet:325317.28125-theano_cntk:0.07708668-theano_mxnet:0.09217975-cntk_mxnet:0.0887682'}
    """
    # if not exists;add
    # else update
    error_bk,layer_name = row['error_backend'],simplify_layer_name(row['error_layer'])
    if (error_bk,layer_name) not in res_dict.keys():
        res_dict[(error_bk,layer_name)] = set()
    res_dict[(error_bk,layer_name)].add(row['model_input'])
    # print(row['error_layer'],simplify_layer_name(row['error_layer']))
    return res_dict


def simplify_layer_name(layer_name:str):
    """
    simplify layer name 'conv2d_copy_LA' -> conv2d
    """
    if 'copy' in layer_name:
        layer_name = layer_name.split("_copy_")[0]
    if 'insert' in layer_name:
        layer_name = layer_name.split("_insert_")[0]

    # '_' in str and str doesn't endwiths '_'
    if "_" in layer_name:
        last_chr = layer_name.rfind("_")
        if last_chr == len(layer_name) -1 or layer_name[last_chr+1].isdigit():
            layer_name = layer_name[:last_chr]
    return layer_name


def get_inconsistency_value(bk_values):
    res_list = []
    for bk,values in bk_values.items():
        res_list.append(f"{bk}:{values}")
    return "|".join(res_list)


def get_largest_error_layer(error_bk,bk_local_res,top_layers):
    def get_layer_value_other_bkp(layer_name,layer_stacks):
        for idx, row in layer_stacks.iterrows():
            if row['current_layer'] == layer_name:
                return float(row['Rl'])

    layerset = set()
    layer_value_dict = dict()
    error_bk_pairs = [bkp for bkp in backend_pairs if error_bk in bkp]
    # length == 1
    other_pair = [bkp for bkp in backend_pairs if error_bk not in bkp]
    for bkp in error_bk_pairs:
        layerset.add(top_layers[bkp][0])
        layer_value_dict[bkp] = (top_layers[bkp][0],get_layer_value_other_bkp(top_layers[bkp][0],bk_local_res[other_pair[0]]))
    if len(layerset) == 1:
        return list(layerset)[0]
    else:
        if layer_value_dict[error_bk_pairs[0]][1] < layer_value_dict[error_bk_pairs[1]][1]:
            return layer_value_dict[error_bk_pairs[0]][0]
        else:
            return layer_value_dict[error_bk_pairs[1]][0]


def get_higher_value_count(l):
    higher_cnt = 0
    for val in l:
        if val >= threshold:
            higher_cnt += 1
    return higher_cnt


def get_another_tuple(idntfr,unique_incon_dict:list):
    """unique_incon_dict is list of ('theano_cntk','O&M',('lenet5-mnist_origin0_theano_cntk_input1',0.35))"""
    idntfr_splits = idntfr.split("_")
    bkp = f"{idntfr_splits[2]}_{idntfr_splits[3]}"
    if idntfr_splits[1] == 'origin0':
        # mutated should be added
        for iu in unique_incon_dict:
            iu_idntfr = iu[2][0]
            iu_idntfr_splits = iu_idntfr.split("_")
            if iu[0] == bkp and iu[1] =='O&M' and idntfr_splits[0] == iu_idntfr_splits[0] and iu_idntfr_splits[1] != 'origin0' and idntfr_splits[-1] == iu_idntfr_splits[-1]:
                return iu[2]
        else:
            raise Exception(f"Can't find equivalent mutated inconsistency for {idntfr}")
    else:
        # origin should be added
        origin_idntfr = f"{idntfr_splits[0]}_origin0_{idntfr_splits[2]}_{idntfr_splits[3]}_{idntfr_splits[-1]}"

        for iu in unique_incon_dict:
            if iu[0] == bkp and iu[1] == 'O&M' and origin_idntfr == iu[2][0]:
                return iu[2]
        else:
            print(origin_idntfr)
            raise Exception(f"Can't find equivalent origin inconsistency for {idntfr}")


def is_all_original(model_inputs):
    for mi in model_inputs:
        mi_splits = mi.split("_")
        if mi_splits[1] != 'origin0':
            return False
    else:
        return True


def is_all_original_on_exp(model_inputs,exp):
    for mi in model_inputs:
        mi_splits = mi.split("_")
        if mi_splits[0] == exp and mi_splits[1] != 'origin0':
            return False
    else:
        return True


def is_all_mutant(model_inputs):
    for mi in model_inputs:
        mi_splits = mi.split("_")
        if mi_splits[1] == 'origin0':
            return False
    else:
        return True


def is_all_mutant_on_exp(model_inputs,exp):
    for mi in model_inputs:
        mi_splits = mi.split("_")
        if mi_splits[0] == exp and mi_splits[1] == 'origin0':
            return False
    else:
        return True


def is_exp_bug(model_inputs,exp):
    for mi in model_inputs:
        mi_splits = mi.split("_")
        if mi_splits[0] == exp:
            return True
    else:
        return False


if __name__ == '__main__':
    start_time = datetime.now()

    config_name = sys.argv[1]
    lemon_cfg = configparser.ConfigParser()
    lemon_cfg.read(f"./config/{config_name}")
    parameters = lemon_cfg['parameters']
    output_dir = parameters['output_dir']
    output_dir = output_dir[:-1] if output_dir.endswith("/") else output_dir
    threshold = parameters.getfloat('threshold')
    current_container = output_dir.rstrip("/").split("/")[-1]
    backend_choices = [1, 2, 3]
    print("current_container",current_container)

    exps = parameters['exps'].lstrip().rstrip().split(" ")
    exps.sort(key=lambda x:x)
    global_backend_pairs = [f"{pair[0]}_{pair[1]}" for pair in combinations(['tensorflow', 'theano', 'cntk','mxnet'], 2)]

    pd_exps = list()
    success_cnt = fail_cnt = 0
    fail_model_inputs = list()
    reduced_bugs = dict()

    columns_cnt = int(3*(len(exps) + 1))
    content = np.zeros((6,columns_cnt),dtype='int64')

    # create an empty DataFrame
    dict_exps = list()
    for e in exps:
        dict_exps.append(f"{e}+O-M")
        dict_exps.append(f"{e}+M-O")
        dict_exps.append(f"{e}+O&M")
        pd_exps.append(f"{e}+LE")
        pd_exps.append(f"{e}+Mu")
        pd_exps.append(f"{e}+In")

    pd_exps.append(f"Total+LE")
    pd_exps.append(f"Total+Mu")
    pd_exps.append(f"Total+In")

    bug_analysis_keys = list(product(dict_exps, global_backend_pairs))
    exp_bkp_tuples = list(product(pd_exps, global_backend_pairs))
    bug_analysis = {t:set() for t in bug_analysis_keys}
    bug_df = pd.DataFrame(content,columns=pd_exps,index=global_backend_pairs)
    model_input_localize = {}

    for backend_choice in backend_choices:
        if backend_choice == 1:
            backends = ['tensorflow', 'theano', 'cntk']
        elif backend_choice == 2:
            backends = ['tensorflow', 'theano', 'mxnet']
        else:
            backends = ['tensorflow', 'cntk', 'mxnet']
        backend_str = "-".join(backends)

        backend_pairs = [f"{pair[0]}_{pair[1]}" for pair in combinations(backends, 2)]
        """Get all exps"""
        unsolved_columns = backend_pairs.copy()
        unsolved_columns.insert(0,'model_input')
        unsolved_df = pd.DataFrame(columns=unsolved_columns)
        solved_df = pd.DataFrame(columns=['error_layer', "error_backend", "model_input"])

        with open(os.path.join(output_dir, f"localize_model_inputs-{backend_str}.pkl"), "rb") as fr:
            localize_model_inputs:dict = pickle.load(fr)

        for exp,model_inputs in localize_model_inputs.items():

            exp_dir = os.path.join(output_dir, exp)
            # get model_inputs:
            localize_res_dir = os.path.join(output_dir,exp, "localization_result")
            exp_metrics_dict = get_metrics_of_exp(exp, exp_dir)
            for model_input in model_inputs:
                # get metrics of model_input
                top_layers = dict()
                bk_local_res = dict()
                second_layers = dict()
                third_layers = dict()
                model_input_split = model_input.split("_")
                tmp_store = dict()
                for bk_p in backend_pairs:
                    local_file_name = "{}_{}_{}_{}.csv".format(model_input_split[0], model_input_split[1], bk_p,
                                                               model_input_split[-1])
                    try:
                        df = pd.read_csv(os.path.join(localize_res_dir, local_file_name), error_bad_lines=False,
                                         usecols=[0, 1, 2])
                        df = df.sort_values(by=['Rl'], ascending=False)
                        bk_local_res[bk_p] = df.copy()
                        top_layers[bk_p] = (df.iloc[0]['current_layer'], float(df.iloc[0]['Rl']))
                        second_layers[bk_p] = (df.iloc[1]['current_layer'], float(df.iloc[1]['Rl']))
                        third_layers[bk_p] = (df.iloc[2]['current_layer'], float(df.iloc[2]['Rl']))

                        tmp_store[bk_p] ={'first':(df.iloc[0]['current_layer'], float(df.iloc[0]['Rl'])),
                                          'second':(df.iloc[1]['current_layer'], float(df.iloc[1]['Rl'])),
                                          'third':(df.iloc[2]['current_layer'], float(df.iloc[2]['Rl']))}
                    except:
                        print(f"{os.path.join(localize_res_dir, local_file_name)} doesn't exists")
                        print(f"No localization result of {model_input} for {backends}")
                        fail_cnt += 1
                        break
                        # raise Exception(local_file_name)
                else:
                    if model_input not in model_input_localize.keys():
                        model_input_localize[model_input] = tmp_store
                    else:
                        model_input_localize[model_input].update(tmp_store)
                    success_cnt += 1
                    metrics_dict = get_metrics_of_model_input(model_input=model_input, backend_pairs=backend_pairs,exp_metrics_dict=exp_metrics_dict)

                    higher_cnt = get_higher_value_count(metrics_dict.values())
                    # localization result exist
                    if higher_cnt == 3:
                        error_layer, error_backend = choose_error_layer(top_layers)
                        if error_layer is not None:
                            # error backend may be like 'tensorflow-theano-cntk'
                            error_backends = error_backend.split("-")
                            for eb in error_backends:
                                solved_row = dict()
                                solved_row['model_input'] = model_input
                                solved_row['error_layer'] = error_layer
                                solved_row['error_backend'] = eb
                                solved_row['values'] = get_inconsistency_value(metrics_dict)
                                # solved_df = solved_df.append([solved_row], ignore_index=True)
                                reduced_bugs = update_suspected_bugs(reduced_bugs, solved_row)

                        else:
                            unsolve_row = dict()
                            unsolve_row['model_input'] = model_input
                            for bk_pair, tu in top_layers.items():
                                unsolve_row[bk_pair] = "{}-{}".format(tu[0], tu[1])
                            unsolved_df = unsolved_df.append([unsolve_row], ignore_index=True)

                    elif higher_cnt == 2:
                        voted_backend = voted_by_inconsistency(metrics_dict)
                        solved_row = dict()
                        solved_row['model_input'] = model_input
                        solved_row['error_layer'] = get_largest_error_layer(voted_backend,bk_local_res,top_layers)
                        solved_row['error_backend'] = voted_backend
                        solved_row['values'] = get_inconsistency_value(metrics_dict)
                        # solved_df = solved_df.append([solved_row], ignore_index=True)
                        reduced_bugs = update_suspected_bugs(reduced_bugs, solved_row)
                    else:
                        fail_model_inputs.append([backend_str,model_input])

    print(f"{success_cnt} model_inputs vote successfully!")
    print(f"{fail_cnt} model_inputs fail to vote !")

    bug_list = list(reduced_bugs.items())
    bug_list.sort(key= lambda t:f"{t[0][0]}+{t[0][1]}") # sort by 'tensorflow+conv2d'
    bug_list = filter_bugs(bug_list=bug_list, output_dir=output_dir)

    with open(os.path.join(output_dir, f"unique_inconsistencies.pkl"), "rb") as fw:
        unique_inconsistencies_dict = pickle.load(fw)

    incon_bugs = dict()
    bug_incons = {idx:set() for idx in range(len(bug_list))}
    for incon_tuple in unique_inconsistencies_dict:
        "incon_tuple is like ('theano-cntk','O-M',('lenet5-mnist_origin0_theano-cntk_input1',0.35))"
        bkp,incon_idntfr = incon_tuple[0],incon_tuple[2][0]
        incon_idntfr_splits = incon_idntfr.split("_")
        incon_mi = f"{incon_idntfr_splits[0]}_{incon_idntfr_splits[1]}_{incon_idntfr_splits[-1]}"
        cur_exp = incon_idntfr_splits[0]
        for idx,bug_item in enumerate(bug_list):
            error_bk = bug_item[0][0]
            if error_bk in bkp and incon_mi in bug_item[1]:
                if incon_idntfr not in incon_bugs.keys():
                    incon_bugs[incon_idntfr] = set()
                incon_bugs[incon_idntfr].add(idx)
                bug_incons[idx].add(incon_tuple[2])

    bug_store = {"O-M":set(),'M-O':set(),"O&M":set()}
    # bug_list like [ (('tensorflow', 'conv2d_1'),['mi1,mmi2,..']), ]
    for idx,bug_item in enumerate(bug_list):
        # check if it's O-bug or M-bug
        error_bk,layer_name = bug_item[0][0], bug_item[0][1]
        mis = bug_item[1]
        if is_all_mutant(mis):
            bug_store['M-O'].add(idx)
        elif is_all_original(mis):
            bug_store['O-M'].add(idx)
        else:
            bug_store['O&M'].add(idx)

        for incon_tuple in unique_inconsistencies_dict:
            "incon_tuple is like ('theano_cntk','O-M',('lenet5-mnist_origin0_theano_cntk_input1',0.35))"
            bkp, incon_idntfr = incon_tuple[0], incon_tuple[2][0]
            incon_idntfr_splits = incon_idntfr.split("_")
            incon_mi = f"{incon_idntfr_splits[0]}_{incon_idntfr_splits[1]}_{incon_idntfr_splits[-1]}"
            cur_exp = incon_idntfr_splits[0]
            if error_bk in bkp and incon_mi in mis:
                if incon_idntfr not in incon_bugs.keys():
                    incon_bugs[incon_idntfr] = set()
                incon_bugs[incon_idntfr].add(idx)
                bug_incons[idx].add(incon_tuple[2])

                if is_all_mutant_on_exp(model_inputs=mis,exp=cur_exp):
                    cluster = 'M-O'
                elif is_all_original_on_exp(model_inputs=mis,exp=cur_exp):
                    cluster = 'O-M'
                else:
                    cluster = 'O&M'
                exp_bkp_tuple = (f"{cur_exp}+{cluster}", bkp)
                if exp_bkp_tuple not in bug_analysis.keys():
                    bug_analysis[exp_bkp_tuple] = set()
                bug_analysis[exp_bkp_tuple].add(idx)

    exp_bkps = list(product(exps, global_backend_pairs))
    total_bugs_dict = {p:set() for p in list(product(['Total+LE','Total+Mu','Total+In'], global_backend_pairs))}

    all_bug_O_M = set()
    all_bug_M_O = set()
    all_bug_O_and_M = set()

    for tu in exp_bkps:

        O_M_str,M_O_str,O_and_M_str = f"{tu[0]}+O-M",f"{tu[0]}+M-O",f"{tu[0]}+O&M"
        bkp = tu[1]
        le_set = (bug_analysis[(O_M_str,bkp)] | bug_analysis[(M_O_str,bkp)] | bug_analysis[(O_and_M_str,bkp)])

        mu_set = bug_analysis[(M_O_str,bkp)]
        in_set = bug_analysis[(O_and_M_str,bkp)]

        bug_df[f"{tu[0]}+LE"][bkp] = str(le_set)
        bug_df[f"{tu[0]}+Mu"][bkp] = str(mu_set)
        bug_df[f"{tu[0]}+In"][bkp] = str(in_set)

        total_bugs_dict[('Total+LE',bkp)].update(le_set)
        total_bugs_dict[('Total+Mu',bkp)].update(mu_set)
        total_bugs_dict[('Total+In',bkp)].update(in_set)

    final_bug_O_M = bug_store['O-M']
    final_bug_M_O = bug_store['M-O']
    final_bug_O_and_M = bug_store['O&M']

    final_LE = final_bug_O_M | final_bug_M_O | final_bug_O_and_M
    final_bug_O = final_bug_O_M | final_bug_O_and_M
    bug_df[f"Bug+Mu"] = str(final_bug_M_O)
    bug_df[f"Bug+In"] = str(final_bug_O_M)
    bug_df[f"Bug+origin"] = str(final_bug_O)
    bug_df[f"Bug+LE"] = str(final_LE)

    print("Original",sorted(list(final_bug_O)))
    print("M-O",sorted(list(final_bug_M_O)))
    print("O-M",sorted(list(final_bug_O_M)))
    print("O&M",sorted(list(final_bug_O_and_M)))

    for k,v in total_bugs_dict.items():
        bug_df[k[0]][k[1]] = str(v)

    bug_df['Total'] = len(bug_list)

    with open(os.path.join(output_dir,"bug_list.txt"),"w") as fw:
        for bug_id,incon_set in bug_incons.items():
            print("###############")
            print(f"# {bug_id} Bug: {bug_list[bug_id][0][0]}-{bug_list[bug_id][0][1]}")

            fw.write("###############\n")
            fw.write(f"# {bug_id} Bug: {bug_list[bug_id][0][0]}-{bug_list[bug_id][0][1]}\n")
            origin_max = 0
            mutated_max = 0
            for incon_tuple in incon_set:
                incon_idntfr,incon_value = incon_tuple[0],incon_tuple[1]
                if incon_idntfr.split("_")[1] == 'origin0':
                    origin_max = incon_value if incon_value > origin_max else origin_max
                else:
                    mutated_max = incon_value if incon_value > mutated_max else mutated_max
            print(f"{len(incon_set)} inconsistencies!")
            print(f"Max original value:{origin_max}")
            print(f"Max mutated value:{mutated_max}")

            fw.write(f"{len(incon_set)} inconsistencies!\n")
            fw.write(f"Max original value:{origin_max}\n")
            fw.write(f"Max mutated value:{mutated_max}\n")

            if bug_id in final_bug_O_M:
                print(f"Type:Only Initial can found\n")
                fw.write(f"Type:Only Initial can found\n")
            if bug_id in final_bug_M_O:
                print(f"Type:Only Mutated can found\n")
                fw.write(f"Type:Only Mutated can found\n")
            if bug_id in final_bug_O_and_M:
                print(f"Type:O&M can found\n")
                fw.write(f"Type:O&M can found\n")

            ordered_mi_set = sorted(list(bug_list[bug_id][1]))
            for idx,mi in enumerate(ordered_mi_set):
                print(f"{idx}.{mi}")
                fw.write(f"{idx}.{mi}\n")
                local_res:dict = model_input_localize[mi]
                for bkp,res in local_res.items():
                    print(bkp,res['first'],res['second'],res['third'])
                    fw.write(f"{bkp} {res['first']} {res['second']} {res['third']}\n")
                print("------------")
                fw.write("------------\n")
            print("###############\n\n")
            fw.write("###############\n\n")

    endtime = datetime.now()
    print("Time cost:",endtime - start_time)