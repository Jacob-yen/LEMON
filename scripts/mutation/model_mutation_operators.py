from scripts.tools import utils
import math
from typing import *
from scripts.mutation.mutation_utils import *
from scripts.mutation.layer_matching import LayerMatching
import random
import os
import warnings
from scripts.logger.lemon_logger import Logger
import datetime
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

mylogger = Logger()


def _assert_indices(mutated_layer_indices: List[int] , depth_layer: int):

    assert max(mutated_layer_indices) < depth_layer,"Max index should be less than layer depth"
    assert min(mutated_layer_indices) >= 0,"Min index should be greater than or equal to zero"


def _shuffle_conv2d(weights, mutate_ratio):
    new_weights = []
    for val in weights:
        # val is bias if len(val.shape) == 1
        if len(val.shape) > 1:
            val_shape = val.shape
            filter_width, filter_height, num_of_input_channels, num_of_output_channels = val_shape
            mutate_output_channels = utils.ModelUtils.generate_permutation(num_of_output_channels, mutate_ratio)
            for output_channel in mutate_output_channels:
                copy_list = val.copy()
                copy_list = np.reshape(copy_list,(filter_width * filter_height * num_of_input_channels, num_of_output_channels))
                selected_list = copy_list[:,output_channel]
                shuffle_selected_list = utils.ModelUtils.shuffle(selected_list)
                copy_list[:, output_channel] = shuffle_selected_list
                val = np.reshape(copy_list,(filter_width, filter_height, num_of_input_channels, num_of_output_channels))
        new_weights.append(val)
    return new_weights


def _shuffle_dense(weights,mutate_ratio):
    new_weights = []
    for val in weights:
        # val is bias if len(val.shape) == 1
        if len(val.shape) > 1:
            val_shape = val.shape
            input_dim,output_dim = val_shape
            mutate_output_dims = utils.ModelUtils.generate_permutation(output_dim, mutate_ratio)
            copy_list = val.copy()
            for output_dim in mutate_output_dims:
                selected_list = copy_list[:, output_dim]
                shuffle_selected_list = utils.ModelUtils.shuffle(selected_list)
                copy_list[:, output_dim] = shuffle_selected_list
            val = copy_list
        new_weights .append(val)
    return new_weights


def _LA_model_scan(model, new_layers, mutated_layer_indices=None):

    layer_utils = LayerUtils()
    layers = model.layers
    # new layers can never be added after the last layer
    positions_to_add = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    _assert_indices(positions_to_add, len(layers))

    insertion_points = {}
    available_new_layers = [layer for layer in
                            layer_utils.available_model_level_layers.keys()] if new_layers is None else new_layers
    for i, layer in enumerate(layers):
        if hasattr(layer, 'activation') and 'softmax' in layer.activation.__name__.lower():
            break
        if i in positions_to_add:
            for available_new_layer in available_new_layers:
                if layer_utils.is_input_legal[available_new_layer](layer.output.shape):
                    if i not in insertion_points.keys():
                        insertion_points[i] = [available_new_layer]
                    else:
                        insertion_points[i].append(available_new_layer)
    return insertion_points


def _MLA_model_scan(model, new_layers, mutated_layer_indices=None):
    layer_matching = LayerMatching()
    layers = model.layers
    # new layers can never be added after the last layer
    positions_to_add = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    _assert_indices(positions_to_add, len(layers))

    insertion_points = {}
    available_new_layers = [layer for layer in layer_matching.layer_concats.keys()] if new_layers is None else new_layers
    for i, layer in enumerate(layers):
        if hasattr(layer, 'activation') and 'softmax' in layer.activation.__name__.lower():
            break
        if i in positions_to_add:
            for available_new_layer in available_new_layers:
                # print('{} test shape: {} as list: {}'.format(available_new_layer, layer.output.shape,
                #                                              layer.output.shape.as_list()))
                if layer_matching.input_legal[available_new_layer](layer.output.shape):
                    # print('shape {} can be inserted'. format(layer.output.shape))
                    if i not in insertion_points.keys():
                        insertion_points[i] = [available_new_layer]
                    else:
                        insertion_points[i].append(available_new_layer)
    return insertion_points


def _LC_and_LR_scan(model, mutated_layer_indices):
    layers = model.layers

    # the last layer should not be copied or removed
    mutated_layer_indices = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    _assert_indices(mutated_layer_indices, len(layers))

    available_layer_indices = []
    for i, layer in enumerate(layers):
        if hasattr(layer, 'activation') and 'softmax' in layer.activation.__name__.lower():
            break
        if i in mutated_layer_indices:
            # InputLayer should not be copied or removed
            from keras.engine.input_layer import InputLayer
            if isinstance(layer, InputLayer):
                continue
            # layers with multiple input tensors can't be copied or removed
            if isinstance(layer.input, list) and len(layer.input) > 1:
                continue
            layer_input_shape = layer.input.shape.as_list()
            layer_output_shape = layer.output.shape.as_list()
            if layer_input_shape == layer_output_shape:
                available_layer_indices.append(i)
    np.random.shuffle(available_layer_indices)
    return available_layer_indices


def _LS_scan(model):
    layers = model.layers
    shape_dict = {}
    for i,layer in enumerate(layers):
        if hasattr(layer, 'activation') and 'softmax' in layer.activation.__name__.lower():
            break
        if isinstance(layer.input, list) and len(layer.input) > 1:
            continue
        layer_input_shape = [str(i) for i in layer.input.shape.as_list()[1:]]
        layer_output_shape = [str(i) for i in layer.output.shape.as_list()[1:]]
        input_shape = "-".join(layer_input_shape)
        output_shape = "-".join(layer_output_shape)
        k = "+".join([input_shape,output_shape])
        if k not in shape_dict.keys():
            shape_dict[k] = [i]
        else:
            shape_dict[k].append(i)
    return shape_dict


def GF_mut(model, mutation_ratio, distribution='normal', STD=0.1, lower_bound=None, upper_bound=None):

    valid_distributions = ['normal', 'uniform']
    assert distribution in valid_distributions, 'Distribution %s is not support.' % distribution
    if distribution == 'uniform' and (lower_bound is None or upper_bound is None):
        mylogger.error('Lower bound and Upper bound is required for uniform distribution.')
        raise ValueError('Lower bound and Upper bound is required for uniform distribution.')

    mylogger.info('copying model...')

    GF_model = utils.ModelUtils.model_copy(model, 'GF')
    mylogger.info('model copied')
    chosed_index = np.random.randint(0, len(GF_model.layers))
    layer = GF_model.layers[chosed_index]
    mylogger.info('executing mutation of {}'.format(layer.name))
    weights = layer.get_weights()
    new_weights = []
    for weight in weights:
        weight_shape = weight.shape
        weight_flat = weight.flatten()
        permu_num = math.floor(len(weight_flat) * mutation_ratio)
        permutation = np.random.permutation(len(weight_flat))[:permu_num]
        STD = math.sqrt(weight_flat.var()) * STD
        weight_flat[permutation] += np.random.normal(scale=STD, size=len(permutation))
        weight = weight_flat.reshape(weight_shape)
        new_weights.append(weight)
    layer.set_weights(new_weights)

    return GF_model


def WS_mut(model, mutation_ratio, mutated_layer_indices=None):
    WS_model = utils.ModelUtils.model_copy(model, 'WS')
    layers = WS_model.layers
    depth_layer = len(layers)
    mutated_layer_indices = np.arange(depth_layer) if mutated_layer_indices is None else mutated_layer_indices
    if 0 < mutation_ratio <= 1.0:
        _assert_indices(mutated_layer_indices, depth_layer)
        np.random.shuffle(mutated_layer_indices)
        i = mutated_layer_indices[0]
        layer = layers[i]
        weights = layer.get_weights()
        layer_name = type(layer).__name__
        if layer_name == "Conv2D" and len(weights) != 0:
            layer.set_weights(_shuffle_conv2d(weights, mutation_ratio))
        elif layer_name == "Dense" and len(weights) != 0:
            layer.set_weights(_shuffle_dense(weights, mutation_ratio))
        else:
            pass
    else:
        mylogger.error("mutation_ratio or index are wrong")
        raise Exception("mutation_ratio or index are wrong")
    return WS_model


def NEB_mut(model, mutation_ratio, mutated_layer_indices=None):
    NEB_model = utils.ModelUtils.model_copy(model, 'NEB')
    layers = NEB_model.layers
    mutated_layer_indices = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    if 0 < mutation_ratio <= 1.0:
        _assert_indices(mutated_layer_indices, len(layers))
        layer_utils = LayerUtils()
        np.random.shuffle(mutated_layer_indices)
        for i in mutated_layer_indices:
            layer = layers[i]
            # skip if layer is not in white list
            if not layer_utils.is_layer_in_weight_change_white_list(layer):
                continue

            weights = layer.get_weights()
            if len(weights) > 0:
                if isinstance(weights, list):
                    # assert len(weights) == 2
                    if len(weights) != 2:
                        continue
                    else:
                        weights_w, weights_b = weights
                        weights_w = weights_w.transpose()
                        permutation = utils.ModelUtils.generate_permutation(weights_w.shape[0], mutation_ratio)
                        weights_w[permutation] = np.zeros(weights_w[0].shape)
                        weights_w = weights_w.transpose()
                        weights_b[permutation] = 0
                        weights = weights_w, weights_b
                        layer.set_weights(weights)
                else:
                    assert isinstance(weights, np.ndarray)
                    weights_w = weights
                    weights_w = weights_w.transpose()
                    permutation = utils.ModelUtils.generate_permutation(weights_w.shape[0], mutation_ratio)
                    weights_w[permutation] = np.zeros(weights_w[0].shape)
                    weights_w = weights_w.transpose()
                    weights = [weights_w]
                    layer.set_weights(weights)
                break
        return NEB_model
    else:
        mylogger.error("mutation_ratio or index are wrong")
        raise Exception("mutation_ratio or index are wrong")


def NAI_mut(model, mutation_ratio, mutated_layer_indices=None):
    NAI_model = utils.ModelUtils.model_copy(model, 'NAI')
    layers = NAI_model.layers
    mutated_layer_indices = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    if 0 < mutation_ratio <= 1.0:
        _assert_indices(mutated_layer_indices, len(layers))
        np.random.shuffle(mutated_layer_indices)
        layer_utils = LayerUtils()
        for i in mutated_layer_indices:
            layer = layers[i]
            if not layer_utils.is_layer_in_weight_change_white_list(layer):
                continue
            weights = layer.get_weights()
            if len(weights) > 0:
                if isinstance(weights, list):
                    if len(weights) != 2:
                        continue
                    else:
                        weights_w, weights_b = weights
                        weights_w = weights_w.transpose()
                        permutation = utils.ModelUtils.generate_permutation(weights_w.shape[0], mutation_ratio)
                        # print(permutation)
                        weights_w[permutation] *= -1
                        weights_w = weights_w.transpose()
                        weights_b[permutation] *= -1
                        weights = weights_w, weights_b
                        layer.set_weights(weights)
                else:
                    weights_w = weights[0]
                    weights_w = weights_w.transpose()
                    permutation = utils.ModelUtils.generate_permutation(weights_w.shape[0], mutation_ratio)
                    # print(permutation)
                    weights_w[permutation] *= -1
                    weights_w = weights_w.transpose()
                    weights = [weights_w]
                    layer.set_weights(weights)
                break
    else:
        mylogger.error("mutation_ratio or index are wrong")
        raise Exception("mutation_ratio or index are wrong")
    return NAI_model


def NS_mut(model, mutated_layer_indices=None):
    NS_model = utils.ModelUtils.model_copy(model, 'NS')
    layers = NS_model.layers
    mutated_layer_indices = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    _assert_indices(mutated_layer_indices, len(layers))
    layer_utils = LayerUtils()
    for i in mutated_layer_indices:
        layer = layers[i]
        if not layer_utils.is_layer_in_weight_change_white_list(layer):
            continue
        weights = layer.get_weights()
        if len(weights) > 0:
            if isinstance(weights, list):
                if len(weights) != 2:
                    continue
                weights_w, weights_b = weights
                weights_w = weights_w.transpose()
                if weights_w.shape[0] >= 2:
                    permutation = np.random.permutation(weights_w.shape[0])[:2]

                    weights_w[permutation[0]], weights_w[permutation[1]] = \
                        weights_w[permutation[1]].copy(), weights_w[permutation[0]].copy()
                    weights_w = weights_w.transpose()

                    weights_b[permutation[0]], weights_b[permutation[1]] = \
                        weights_b[permutation[1]].copy(), weights_b[permutation[0]].copy()

                    weights = weights_w, weights_b

                    layer.set_weights(weights)
                else:
                    mylogger.warning("NS not used! One neuron can't be shuffle!")
            else:
                assert isinstance(weights, np.ndarray)
                weights_w = weights
                weights_w = weights_w.transpose()
                if weights_w.shape[0] >= 2:
                    permutation = np.random.permutation(weights_w.shape[0])[:2]

                    weights_w[permutation[0]], weights_w[permutation[1]] = \
                        weights_w[permutation[1]].copy(), weights_w[permutation[0]].copy()
                    weights_w = weights_w.transpose()
                    weights = [weights_w]

                    layer.set_weights(weights)
                else:
                    mylogger.warning("NS not used! One neuron can't be shuffle!")
            break

    return NS_model


def ARem_mut(model, mutated_layer_indices=None):
    ARem_model = utils.ModelUtils.model_copy(model, 'ARem')
    layers = ARem_model.layers
    # the activation of last layer should not be removed
    mutated_layer_indices = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    np.random.shuffle(mutated_layer_indices)
    _assert_indices(mutated_layer_indices, len(layers))

    for i in mutated_layer_indices:
        layer = layers[i]
        if hasattr(layer, 'activation') and 'softmax' not in layer.activation.__name__.lower():
            layer.activation = ActivationUtils.no_activation
            break
    return ARem_model


def ARep_mut(model, new_activations=None, mutated_layer_indices=None):

    activation_utils = ActivationUtils()
    ARep_model = utils.ModelUtils.model_copy(model, 'ARep')
    layers = ARep_model.layers
    # the activation of last layer should not be replaced
    mutated_layer_indices = np.arange(len(layers) - 1) if mutated_layer_indices is None else mutated_layer_indices
    np.random.shuffle(mutated_layer_indices)
    _assert_indices(mutated_layer_indices, len(layers))
    for i in mutated_layer_indices:
        layer = layers[i]
        if hasattr(layer, 'activation') and 'softmax' not in layer.activation.__name__.lower():
            layer.activation = activation_utils.pick_activation_randomly(new_activations)
            break
    return ARep_model


def LA_mut(model, new_layers=None, mutated_layer_indices=None):
    layer_utils = LayerUtils()
    if new_layers is not None:
        for layer in new_layers:
            if layer not in layer_utils.available_model_level_layers.keys():
                mylogger.error('Layer {} is not supported.'.format(layer))
                raise Exception('Layer {} is not supported.'.format(layer))
    LA_model = utils.ModelUtils.model_copy(model, 'LA')

    insertion_points = _LA_model_scan(LA_model, new_layers, mutated_layer_indices)
    if len(insertion_points.keys()) == 0:
        mylogger.warning('no appropriate layer to insert')
        return None
    for key in insertion_points.keys():
        mylogger.info('{} can be added after layer {} ({})'
            .format(insertion_points[key], key, type(model.layers[key])))
    layers_index_avaliable = list(insertion_points.keys())
    layer_index_to_insert = layers_index_avaliable[np.random.randint(0, len(layers_index_avaliable))]
    available_new_layers = insertion_points[layer_index_to_insert]
    layer_name_to_insert = available_new_layers[np.random.randint(0, len(available_new_layers))]
    mylogger.info('insert {} after {}'.format(layer_name_to_insert, LA_model.layers[layer_index_to_insert].name))
    # insert new layer
    if model.__class__.__name__ == 'Sequential':
        import keras
        new_model = keras.models.Sequential()
        for i, layer in enumerate(LA_model.layers):
            new_layer = LayerUtils.clone(layer)
            new_model.add(new_layer)
            if i == layer_index_to_insert:
                output_shape = layer.output_shape
                new_model.add(layer_utils.available_model_level_layers[layer_name_to_insert](output_shape))
    else:

        def layer_addition(x, layer):
            x = layer(x)
            output_shape = layer.output_shape
            new_layer = layer_utils.available_model_level_layers[layer_name_to_insert](output_shape)
            x = new_layer(x)
            return x
        new_model = utils.ModelUtils.functional_model_operation(LA_model, operation={LA_model.layers[layer_index_to_insert].name: layer_addition})

    assert len(new_model.layers) == len(model.layers) + 1
    tuples = []
    import time
    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name

        if layer_name.endswith('_copy_LA'):
            key = layer_name[:-8]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()
        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            for i in range(len(shape_sw)):
                assert shape_sw[i] == shape_w[i], '{}'.format(layer_name)
            tuples.append((sw, w))

    import keras.backend as K
    K.batch_set_value(tuples)
    return new_model


def MLA_mut(model, new_layers = None, mutated_layer_indices=None):
    # mutiple layers addition
    layer_matching = LayerMatching()
    if new_layers is not None:
        for layer in new_layers:
            if layer not in layer_matching.layer_concats.keys():
                raise Exception('Layer {} is not supported.'.format(layer))
    MLA_model = utils.ModelUtils.model_copy(model, 'MLA')
    insertion_points = _MLA_model_scan(model, new_layers, mutated_layer_indices)
    mylogger.info(insertion_points)
    if len(insertion_points.keys()) == 0:
        mylogger.warning('no appropriate layer to insert')
        return None
    for key in insertion_points.keys():
        mylogger.info('{} can be added after layer {} ({})'
                             .format(insertion_points[key], key, type(model.layers[key])))

    # use logic: randomly select a new layer available to insert into the layer which can be inserted
    layers_index_avaliable = list(insertion_points.keys())
    # layer_index_to_insert = np.max([i for i in insertion_points.keys()])
    layer_index_to_insert = layers_index_avaliable[np.random.randint(0, len(layers_index_avaliable))]
    available_new_layers = insertion_points[layer_index_to_insert]
    layer_name_to_insert = available_new_layers[np.random.randint(0, len(available_new_layers))]
    mylogger.info('choose to insert {} after {}'.format(layer_name_to_insert, MLA_model.layers[layer_index_to_insert].name))
    # insert new layers
    if model.__class__.__name__ == 'Sequential':
        import keras
        new_model = keras.models.Sequential()
        for i, layer in enumerate(MLA_model.layers):
            new_layer = LayerUtils.clone(layer)
            # new_layer.name += "_copy"
            new_model.add(new_layer)
            if i == layer_index_to_insert:
                output_shape = layer.output.shape.as_list()
                layers_to_insert = layer_matching.layer_concats[layer_name_to_insert](output_shape)
                for layer_to_insert in layers_to_insert:
                    layer_to_insert.name += "_insert"
                    mylogger.info(layer_to_insert)
                    new_model.add(layer_to_insert)
        new_model.build(MLA_model.input_shape)
    else:
        def layer_addition(x, layer):
            x = layer(x)
            output_shape = layer.output.shape.as_list()
            new_layers = layer_matching.layer_concats[layer_name_to_insert](output_shape)
            for l in new_layers:
                l.name += "_insert"
                mylogger.info('insert layer {}'.format(str(l)))
                x = l(x)
            return x
        new_model = utils.ModelUtils.functional_model_operation(MLA_model, operation={MLA_model.layers[layer_index_to_insert].name: layer_addition})

    tuples = []
    import time
    start_time = time.time()

    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name
        if layer_name.endswith('_copy_MLA'):
            key = layer_name[:-9]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()

        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            assert shape_sw[0] == shape_w[0]
            tuples.append((sw, w))

    import keras.backend as K
    K.batch_set_value(tuples)
    end_time = time.time()
    print('set weight cost {}'.format(end_time - start_time))

    return new_model


def LC_mut(model, mutated_layer_indices=None):
    LC_model = utils.ModelUtils.model_copy(model, 'LC')
    available_layer_indices = _LC_and_LR_scan(LC_model, mutated_layer_indices)

    if len(available_layer_indices) == 0:
        mylogger.warning('no appropriate layer to copy (input and output shape should be same)')
        return None

    # use logic: copy the last available layer
    copy_layer_index = available_layer_indices[-1]
    copy_layer_name = LC_model.layers[copy_layer_index].name + '_repeat'

    mylogger.info('choose to copy layer {}'.format(LC_model.layers[copy_layer_index].name))

    if model.__class__.__name__ == 'Sequential':
        import keras
        new_model = keras.models.Sequential()
        for i, layer in enumerate(LC_model.layers):
            new_model.add(LayerUtils.clone(layer))
            if i == copy_layer_index:
                copy_layer = LayerUtils.clone(layer)
                copy_layer.name += '_repeat'
                new_model.add(copy_layer)
    else:
        def layer_repeat(x, layer):
            x = layer(x)
            copy_layer = LayerUtils.clone(layer)
            copy_layer.name += '_repeat'
            x = copy_layer(x)
            return x
        new_model = utils.ModelUtils.functional_model_operation(LC_model, operation={LC_model.layers[copy_layer_index].name: layer_repeat})

    # update weights
    assert len(new_model.layers) == len(model.layers) + 1
    tuples = []
    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name
        if layer_name.endswith('_copy_LC'):
            key = layer_name[:-8]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()

        if layer_name + '_copy_LC_repeat' == copy_layer_name:
            for sw, w in zip(new_model_layers[copy_layer_name].weights, layer_weights):
                shape_sw = np.shape(sw)
                shape_w = np.shape(w)
                assert len(shape_sw) == len(shape_w)
                assert shape_sw[0] == shape_w[0]
                tuples.append((sw, w))

        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            assert shape_sw[0] == shape_w[0]
            tuples.append((sw, w))

    import keras.backend as K
    K.batch_set_value(tuples)
    return new_model


def LR_mut(model, mutated_layer_indices=None):
    LR_model = utils.ModelUtils.model_copy(model, 'LR')
    available_layer_indices = _LC_and_LR_scan(LR_model, mutated_layer_indices)

    if len(available_layer_indices) == 0:
        mylogger.warning('no appropriate layer to remove (input and output shape should be same)')
        return None

    # use logic: remove the last available layer
    remove_layer_index = available_layer_indices[-1]
    mylogger.info('choose to remove layer {}'.format(LR_model.layers[remove_layer_index].name))
    if model.__class__.__name__ == 'Sequential':
        import keras
        new_model = keras.models.Sequential()
        for i, layer in enumerate(LR_model.layers):
            if i != remove_layer_index:
                new_layer = LayerUtils.clone(layer)
                # new_layer.name += '_copy'
                new_model.add(new_layer)
    else:
        new_model = utils.ModelUtils.functional_model_operation(LR_model, operation={LR_model.layers[remove_layer_index].name: lambda x, layer: x})

    # update weights
    assert len(new_model.layers) == len(model.layers) - 1
    tuples = []
    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name
        if layer_name.endswith('_copy_LR'):
            key = layer_name[:-8]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in new_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()

        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            assert shape_sw[0] == shape_w[0]
            tuples.append((sw, w))

    import keras.backend as K
    K.batch_set_value(tuples)
    return new_model


def LS_mut(model):
    LS_model = utils.ModelUtils.model_copy(model,"LS")
    shape_dict = _LS_scan(LS_model)
    layers = LS_model.layers

    swap_list = []
    for v in shape_dict.values():
        if len(v) > 1:
            swap_list.append(v)
    if len(swap_list) == 0:
        mylogger.warning("No layers to swap!")
        return None
    swap_list = swap_list[random.randint(0, len(swap_list)-1)]
    choose_index = random.sample(swap_list, 2)
    mylogger.info('choose to swap {} ({} - {}) and {} ({} - {})'.format(layers[choose_index[0]].name,
                                                                layers[choose_index[0]].input.shape,
                                                                layers[choose_index[0]].output.shape,
                                                                layers[choose_index[1]].name,
                                                                layers[choose_index[1]].input.shape,
                                                                layers[choose_index[1]].output.shape))
    if model.__class__.__name__ == 'Sequential':
        import keras
        new_model = keras.Sequential()
        for i, layer in enumerate(layers):
            if i == choose_index[0]:
                new_model.add(LayerUtils.clone(layers[choose_index[1]]))
            elif i == choose_index[1]:
                new_model.add(LayerUtils.clone(layers[choose_index[0]]))
            else:
                new_model.add(LayerUtils.clone(layer))
    else:
        layer_1 = layers[choose_index[0]]
        layer_2 = layers[choose_index[1]]
        new_model = utils.ModelUtils.functional_model_operation(LS_model, {layer_1.name: lambda x, layer: LayerUtils.clone(layer_2)(x),
                                                           layer_2.name: lambda x, layer: LayerUtils.clone(layer_1)(x)})

    # update weights
    assert len(new_model.layers) == len(model.layers)
    tuples = []
    old_model_layers = {}
    for layer in model.layers:
        old_model_layers[layer.name] = layer

    new_model_layers = {}
    for layer in new_model.layers:
        layer_name = layer.name
        if layer_name.endswith('_copy_LS'):
            key = layer_name[:-8]
        else:
            key = layer_name
        new_model_layers[key] = layer

    for layer_name in old_model_layers.keys():
        layer_weights = old_model_layers[layer_name].get_weights()

        for sw, w in zip(new_model_layers[layer_name].weights, layer_weights):
            shape_sw = np.shape(sw)
            shape_w = np.shape(w)
            assert len(shape_sw) == len(shape_w)
            assert shape_sw[0] == shape_w[0]
            tuples.append((sw, w))

    import keras.backend as K
    K.batch_set_value(tuples)
    return new_model


if __name__ == '__main__':
    pass