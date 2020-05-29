import sys
from scripts.mutation.model_mutation_operators import *
import argparse
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
mylogger = Logger()

def generate_model_by_model_mutation(model, operator,mutate_ratio=0.3):
    """
    Generate models using specific mutate operator
    :param model: model loaded by keras (tensorflow backend default)
    :param operator: mutation operator
    :param mutate_ratio: ratio of selected neurons
    :return: mutation model object
    """
    if operator == 'WS':
        mutate_indices = utils.ModelUtils.weighted_layer_indices(model)
        mylogger.info("Generating model using {}".format(operator))
        return WS_mut(model=model,mutation_ratio=mutate_ratio,mutated_layer_indices=mutate_indices)
    elif operator == 'GF':
        mylogger.info("Generating model using {}".format(operator))
        return GF_mut(model=model,mutation_ratio=mutate_ratio)
    elif operator == 'NEB':
        mylogger.info("Generating model using {}".format(operator))
        return NEB_mut(model=model, mutation_ratio=mutate_ratio)
    elif operator == 'NAI':
        mylogger.info("Generating model using {}".format(operator))
        return NAI_mut(model=model, mutation_ratio=mutate_ratio)
    elif operator == 'NS':
        mylogger.info("Generating model using {}".format(operator))
        return NS_mut(model=model)
    elif operator == 'ARem':
        mylogger.info("Generating model using {}".format(operator))
        return ARem_mut(model=model)
    elif operator == 'ARep':
        mylogger.info("Generating model using {}".format(operator))
        return ARep_mut(model=model)
    elif operator == 'LA':
        mylogger.info("Generating model using {}".format(operator))
        return LA_mut(model=model)
    elif operator == 'LC':
        mylogger.info("Generating model using {}".format(operator))
        return LC_mut(model=model)
    elif operator == 'LR':
        mylogger.info("Generating model using {}".format(operator))
        return LR_mut(model=model)
    elif operator == 'LS':
        mylogger.info("Generating model using {}".format(operator))
        return LS_mut(model=model)
    elif operator == 'MLA':
        mylogger.info("Generating model using {}".format(operator))
        return MLA_mut(model=model)
    else:
        mylogger.info("No such Mutation operator {}".format(operator))
        return None


def all_mutate_ops():
    return ['WS','GF','NEB','NAI','NS','ARem','ARep','LA','LC','LR','LS','MLA']


if __name__ == '__main__':


    """Parser of command args"""
    parse = argparse.ArgumentParser()
    parse.add_argument("--model", type=str, help="model path")
    parse.add_argument("--mutate_op", type=str, help="model mutation operator")
    parse.add_argument("--save_path", type=str, help="model save path")
    parse.add_argument("--mutate_ratio", type=float, help="mutate ratio")
    flags, unparsed = parse.parse_known_args(sys.argv[1:])

    import keras
    model_path = flags.model
    mutate_ratio = flags.mutate_ratio
    print("Current {}; Mutate ratio {}".format(flags.mutate_op,mutate_ratio))
    origin_model = keras.models.load_model(model_path, custom_objects=utils.ModelUtils.custom_objects())
    mutated_model = generate_model_by_model_mutation(model=origin_model,operator=flags.mutate_op,mutate_ratio=mutate_ratio)


    if mutated_model is None:
        raise Exception("Error: Model mutation using {} failed".format(flags.mutate_op))
    else:
        mutated_model.save(flags.save_path)






