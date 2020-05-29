"""
# Part  of localization phase
"""
import sys
import os
import pickle
import keras
import keras.backend as K
from scripts.tools.utils import ModelUtils

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
        # print(layer_name[last_chr+1],layer_name[last_chr+1].isdigit())
        if last_chr == len(layer_name) -1 or layer_name[last_chr+1].isdigit():
            layer_name = layer_name[:last_chr]
            # print("After",layer_name)
    return layer_name


def update_bug_list(final_bugs:dict,bug_item,bug_type):
    bkp,layer_name,model_inputs = bug_item[0][0],bug_item[0][1],bug_item[1]
    if (bkp,bug_type) not in final_bugs.keys():
        final_bugs[(bkp,bug_type)] = set()
    for mi in model_inputs:
        final_bugs[(bkp, bug_type)].add(mi)
    return final_bugs


def filter_bugs(bug_list,output_dir):
    for idx,bug in enumerate(bug_list):
        print(f"#{idx}",bug[0],f"{len(bug[1])} model inputs")
    # load bugs
    final_bugs = dict()
    for idx,bug_item in enumerate(bug_list):
        print(f"Bug #{idx}")
        # bug_item is like [(('tensorflow', 'batch_normalization'),[mi1.mi2,mi3])]
        exp_models = [f"{s.split('_')[0]}_{s.split('_')[1]}.h5" for s in bug_item[1]]
        print(f"{len(exp_models)} to load. choose first")
        layer_type_set = set()
        model_name = exp_models[0]
        exp = model_name.split("_")[0]
        model_path = f"{output_dir}/{exp}/mut_model/{model_name}"
        model = keras.models.load_model(model_path,custom_objects=ModelUtils.custom_objects())
        layers = model.layers
        for lyr in layers:
            if simplify_layer_name(lyr.name) == bug_item[0][1]:
                layer_type_set.add(lyr.__class__.__name__)
        K.clear_session()
        print(bug_item[0][1],layer_type_set)
        if len(layer_type_set) > 1:
            raise Exception("found more than one layer!")
        bug_type = list(layer_type_set)[0]
        final_bugs = update_bug_list(final_bugs=final_bugs,bug_item=bug_item,bug_type=bug_type)
    final_bugs = list(final_bugs.items())
    for idx,bug in enumerate(final_bugs):
        print(f"#{idx}",bug[0],f"{len(bug[1])} model inputs")
    return final_bugs


if __name__ == "__main__":
    save_path = sys.argv[1]
    current_container = save_path.rstrip("/").split("/")[-1]
    bug_list_path = os.path.join(save_path, "bug_list.pkl")

    with open(bug_list_path,"rb") as fr:
        bug_list = pickle.load(fr)
    filter_bugs(bug_list,current_container)