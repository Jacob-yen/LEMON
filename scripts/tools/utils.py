import os
import pickle
import math
from PIL import Image
import warnings
import datetime
import configparser
import numpy as np

np.random.seed(20200501)
warnings.filterwarnings("ignore")
"""Set seed and Init cuda"""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""



class ModelUtils:
    def __init__(self):
        pass

    @staticmethod
    def model_copy(model, mode=''):
        from scripts.mutation.mutation_utils import LayerUtils
        import keras
        suffix = '_copy_' + mode
        if model.__class__.__name__ == 'Sequential':
            new_layers = []
            for layer in model.layers:
                new_layer = LayerUtils.clone(layer)
                new_layer.name += suffix
                new_layers.append(new_layer)
            new_model = keras.Sequential(layers=new_layers, name=model.name + suffix)
        else:
            new_model = ModelUtils.functional_model_operation(model, suffix=suffix)

        s = datetime.datetime.now()
        new_model.set_weights(model.get_weights())
        e1 = datetime.datetime.now()
        td1 = e1 - s
        h, m, s = ToolUtils.get_HH_mm_ss(td1)
        print("Set model weights! {} hour,{} min,{} sec".format(h, m, s))
        del model
        return new_model

    @staticmethod
    def functional_model_operation(model, operation=None, suffix=None):
        from scripts.mutation.mutation_utils import LayerUtils
        input_layers = {}
        output_tensors = {}
        model_output = None
        for layer in model.layers:
            for node in layer._outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name not in input_layers.keys():
                    input_layers[layer_name] = [layer.name]
                else:
                    input_layers[layer_name].append(layer.name)

        output_tensors[model.layers[0].name] = model.input

        for layer in model.layers[1:]:
            layer_input_tensors = [output_tensors[l] for l in input_layers[layer.name]]
            if len(layer_input_tensors) == 1:
                layer_input_tensors = layer_input_tensors[0]

            if operation is not None and layer.name in operation.keys():
                x = layer_input_tensors
                cloned_layer = LayerUtils.clone(layer)
                if suffix is not None:
                    cloned_layer.name += suffix
                x = operation[layer.name](x, cloned_layer)
            else:
                cloned_layer = LayerUtils.clone(layer)
                if suffix is not None:
                    cloned_layer.name += suffix
                x = cloned_layer(layer_input_tensors)

            output_tensors[layer.name] = x
            model_output = x

        import keras
        return keras.Model(inputs=model.inputs, outputs=model_output)

    @staticmethod
    def save_initial_weights(model):
        weights = model.get_weights()
        np.save('initial_weights.npy', weights)

    @staticmethod
    def load_initial_weights(model):
        weights = np.load('initial_weights.npy')
        model.set_weights(weights)
        return model

    @staticmethod
    def save_layers_output(path, layers_output):

        dirname = os.path.dirname(path)
        if len(dirname)>0 and (not os.path.exists(dirname)):
            os.makedirs(dirname)
        with open(path,'wb') as f:
            pickle.dump(layers_output,f)

    @staticmethod
    def load_layers_output(path):
        if not os.path.exists(path):
            return None
        with open(path,'rb') as f:
            layers_output = pickle.load(f)
        return layers_output

    @staticmethod
    def layer_divation(model, model_nodes, layer_index, layers_output_1, layers_output_2, epsilon=1e-7):
        layer = model.layers[layer_index]
        # get all of its input layers
        input_layers_index = []
        for node in layer._inbound_nodes:
            if node not in model_nodes:
                continue
            for l in node.inbound_layers:
                from keras.engine.input_layer import InputLayer
                if isinstance(l, InputLayer):
                    continue
                # find the index of l in model
                for i, model_layer in enumerate(model.layers):
                    if l == model_layer:
                        input_layers_index.append(i)
                        break
                else:
                    raise Exception('can not find the layer in model')
        # calculate the divation of current layer
        cur_output_1 = layers_output_1[layer_index]
        cur_output_2 = layers_output_2[layer_index]
        delta_cur = MetricsUtils.delta(cur_output_1, cur_output_2)[0] # the second value of delta is sum()

        if len(input_layers_index) == 0:
            delta_pre = 0
        else:
            delta_pre_list = []
            for i in input_layers_index:
                pre_output_1 = layers_output_1[i]
                pre_output_2 = layers_output_2[i]
                delta_pre_list.append(MetricsUtils.delta(pre_output_1, pre_output_2)[0])
            delta_pre = np.max(delta_pre_list, axis=0)
        return delta_cur, (delta_cur - delta_pre) / (delta_pre + epsilon), [model.layers[i].name for i in input_layers_index]

    @staticmethod
    def layers_divation(model, layers_output_1, layers_output_2):
        relevant_nodes = []
        for v in model._nodes_by_depth.values():
            relevant_nodes += v
        layers_divation = []
        for i in range(len(model.layers)):
            layers_divation.append(ModelUtils.layer_divation(model, relevant_nodes, i, layers_output_1, layers_output_2))
        return layers_divation

    @staticmethod
    def layers_output(model, input):
        from keras import backend as K
        # print(K.backend()+" in loadmodel")
        from keras.engine.input_layer import InputLayer
        get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                      [l.output for l in
                                       (model.layers[1:]
                                        if isinstance(model.layers[0], InputLayer)
                                        else model.layers)])
        if isinstance(model.layers[0], InputLayer):
            layers_output = [input]
            layers_output.extend(get_layer_output([input, 0]))
        else:
            layers_output = get_layer_output([input, 0])
        return layers_output

    @staticmethod
    def layers_input(model, input):
        inputs = [[input]]
        from keras import backend as K
        from keras.engine.input_layer import InputLayer
        for i, layer in enumerate(model.layers):
            if i == 0:
                continue
            if i == 1 and isinstance(model.layers[0], InputLayer):
                continue
            get_layer_input = K.function([model.layers[0].input, K.learning_phase()],
                                         layer.input if isinstance(layer.input, list) else [layer.input])
            inputs.append(get_layer_input([input, 0]))
        return inputs

    @staticmethod
    def generate_permutation(size_of_permutation, extract_portion):
        assert extract_portion <= 1
        num_of_extraction = math.floor(size_of_permutation * extract_portion)
        permutation = np.random.permutation(size_of_permutation)
        permutation = permutation[:num_of_extraction]
        return permutation

    @staticmethod
    def shuffle(a):
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        length = len(a)
        permutation = np.random.permutation(length)
        index_permutation = np.arange(length)
        shuffled_a[permutation] = a[index_permutation]
        return shuffled_a

    @staticmethod
    def compile_model(model, optimer, loss, metric:list):
        model.compile(optimizer=optimer,
                      loss=loss,
                      metrics=metric)
        return model

    @staticmethod
    def custom_objects():
        from scripts.mutation.mutation_utils import ActivationUtils
        objects = {}
        objects['no_activation'] = ActivationUtils.no_activation
        objects['leakyrelu'] = ActivationUtils.leakyrelu
        return objects



    @staticmethod
    def weighted_layer_indices(model):
        indices = []
        for i, layer in enumerate(model.layers):
            weight_count = layer.count_params()
            if weight_count > 0:
                indices.append(i)
        return indices

    @staticmethod
    def is_valid_model(inputs_backends, threshold=0.95):
        invalid_status_num = 0
        inputs_values = list(inputs_backends.values())
        # results like (1500,1) is valid
        if inputs_values[0].shape[1] == 1:
            return True
        else:
            for inputs in inputs_backends.values():
                indice_map = {}
                for input in inputs:
                    max_indice = np.argmax(input)
                    if max_indice not in indice_map.keys():
                        indice_map[max_indice] = 1
                    else:
                        indice_map[max_indice] += 1
                for indice in indice_map.keys():
                    if indice_map[indice] > len(inputs) * threshold:
                        invalid_status_num += 1

            return False if invalid_status_num == 3 else True


class DataUtils:

    @staticmethod
    def image_resize(x, shape):
        x_return = []
        for x_test in x:
            tmp = np.copy(x_test)
            img = Image.fromarray(tmp.astype('uint8')).convert('RGB')
            img = img.resize(shape, Image.ANTIALIAS)
            x_return.append(np.array(img))
        return np.array(x_return)

    @staticmethod
    def get_data_by_exp(exp):
        import keras
        import keras.backend as K
        K.set_image_data_format("channels_last")

        lemon_cfg = configparser.ConfigParser()
        lemon_cfg.read("./config/experiments.conf")
        dataset_dir = lemon_cfg['parameters']['dataset_dir']
        x_test = y_test = []
        if 'fashion-mnist' in exp:
            _, (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
            x_test = DataUtils.get_fashion_mnist_data(x_test)
            y_test = keras.utils.to_categorical(y_test, num_classes=10)
        elif 'mnist' in exp:
            _, (x_test, y_test) = keras.datasets.mnist.load_data()
            x_test = DataUtils.get_mnist_data(x_test)
            y_test = keras.utils.to_categorical(y_test, num_classes=10)
        elif 'cifar10' in exp:
            _, (x_test, y_test) = keras.datasets.cifar10.load_data()
            x_test = DataUtils.get_cifar10_data(x_test)
            y_test = keras.utils.to_categorical(y_test, num_classes=10)
        elif 'imagenet' in exp:
            input_precessor = DataUtils.imagenet_preprocess_dict()
            input_shapes_dict = DataUtils.imagenet_shape_dict()
            model_name = exp.split("-")[0]
            shape = input_shapes_dict[model_name]
            data_path = os.path.join(dataset_dir,"sampled_imagenet-1500.npz")
            data = np.load(data_path)
            x, y = data['x_test'], data['y_test']
            x_resize = DataUtils.image_resize(np.copy(x),shape)
            x_test = input_precessor[model_name](x_resize)
            y_test = keras.utils.to_categorical(y, num_classes=1000)
        elif 'sinewave' in exp:
            """
            see more details in
            https://github.com/StevenZxy/CIS400/tree/f69489c0624157ae86b5d8ddb1fa99c89a927256/code/LSTM-Neural-Network-for-Time-Series-Prediction-master
            """
            import pandas as pd
            dataframe = pd.read_csv(f"{dataset_dir}/sinewave.csv")
            test_size,seq_len = 1500, 50
            data_test = dataframe.get("sinewave").values[-(test_size + 50):]
            data_windows = []
            for i in range(test_size):
                data_windows.append(data_test[i:i + seq_len])
            data_windows = np.array(data_windows).astype(float).reshape((test_size,seq_len,1))
            data_windows = np.array(data_windows).astype(float)
            x_test = data_windows[:, :-1]
            y_test = data_windows[:, -1, [0]]

        elif 'price' in exp:
            """see more details in https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo/tree/master/StockPricesPredictionProject"""
            x_test, y_test = DataUtils.get_price_data(dataset_dir)

        # TODO: Add your own data preprocessing here
        # Note: The returned inputs should be preprocessed and labels should decoded as one-hot vector which could be directly feed in model.
        # Both of them should be returned in batch, e.g. shape like (1500,28,28,1) and (1500,10)
        # elif 'xxx' in exp:
        #     x_test, y_test = get_your_data(dataset_dir)

        return x_test, y_test

    @staticmethod
    def save_img_from_array(path,array,index,exp):
        im = Image.fromarray(array)
        #path = path.rstrip("/")
        #save_path = "{}/{}_{}.png".format(path,exp,index)
        save_path = os.path.join(path,"{}_{}.png".format(exp, index))
        im.save(save_path)
        return save_path

    @staticmethod
    def shuffled_data(x, y, bs=None):
        ds = x.shape[0]
        all_idx = np.arange(ds)
        np.random.shuffle(all_idx)
        shuffle_idx = all_idx
        # shuffle_idx = all_idx[:bs]
        return x[shuffle_idx], y[shuffle_idx]

    @staticmethod
    def get_mnist_data(x_test):
        x_test = x_test.astype('float32') / 255.0
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        return x_test

    @staticmethod
    def get_fashion_mnist_data(x_test):
        x_test = x_test.astype('float32') / 255.0
        w, h = 28, 28
        x_test = x_test.reshape(x_test.shape[0], w, h, 1)
        return x_test

    @staticmethod
    def get_cifar10_data(x_test):
        x_test = x_test.astype('float32') / 255.0
        w, h = 32, 32
        x_test = x_test.reshape(x_test.shape[0], w, h, 3)
        return x_test

    @staticmethod
    def get_price_data(data_dir):
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler

        def create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back - 1):
                a = dataset[i:(i + look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)

        input_file = os.path.join(data_dir,"DIS.csv")
        df = pd.read_csv(input_file, header=None, index_col=None, delimiter=',')
        all_y = df[5].values
        dataset = all_y.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        train_size = int(len(dataset) * 0.5)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        # reshape into X=t and Y=t+1, timestep 240
        look_back = 240
        trainX, trainY = create_dataset(train, look_back)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        return trainX,trainY

    @staticmethod
    def imagenet_preprocess_dict():
        import keras
        keras_preprocess_dict = dict()
        keras_preprocess_dict['resnet50'] = keras.applications.resnet50.preprocess_input
        keras_preprocess_dict['densenet121'] = keras.applications.densenet.preprocess_input
        keras_preprocess_dict['mobilenet.1.00.224'] = keras.applications.mobilenet.preprocess_input
        keras_preprocess_dict['vgg16'] = keras.applications.vgg16.preprocess_input
        keras_preprocess_dict['vgg19'] = keras.applications.vgg19.preprocess_input
        keras_preprocess_dict['inception.v3'] = keras.applications.inception_v3.preprocess_input
        keras_preprocess_dict['inception.v2'] = keras.applications.inception_resnet_v2.preprocess_input
        keras_preprocess_dict['xception'] = keras.applications.xception.preprocess_input
        return keras_preprocess_dict

    @staticmethod
    def imagenet_shape_dict():
        image_shapes = dict()
        image_shapes['resnet50'] = (224,224)
        image_shapes['densenet121'] = (224,224)
        image_shapes['mobilenet.1.00.224'] = (224,224)
        image_shapes['vgg16'] = (224,224)
        image_shapes['vgg19'] = (224, 224)
        image_shapes['inception.v3'] = (299,299)
        image_shapes['inception.v2'] = (299, 299)
        image_shapes['xception'] = (299,299)
        return image_shapes


class ToolUtils:

    @staticmethod
    def select_mutant(roulette,**kwargs):
        return roulette.choose_mutant()

    @staticmethod
    def select_mutator(logic, **kwargs):
        # import numpy as np
        # return np.random.permutation(mutate_ops)[0]
        last_used_mutator = kwargs['last_used_mutator']
        return logic.choose_mutator(last_used_mutator)

    @staticmethod
    def get_HH_mm_ss(td):
        days, seconds = td.days, td.seconds
        hours = days * 24 + seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return hours, minutes, secs


class MetricsUtils:

    @staticmethod
    def delta(y1_pred, y2_pred,y_true=None):
        y1_pred = np.reshape(y1_pred, [np.shape(y1_pred)[0], -1])
        y2_pred = np.reshape(y2_pred, [np.shape(y2_pred)[0], -1])
        return np.mean(np.abs(y1_pred - y2_pred), axis=1), np.sum(np.abs(y1_pred - y2_pred), axis=1)

    @staticmethod
    def D_MAD_metrics(y1_pred, y2_pred,y_true, epsilon=1e-7):
        # sum could be remove and use mean in branch.
        theta_y1,sum_y1 = MetricsUtils.delta(y1_pred, y_true)
        theta_y2,sum_y2 = MetricsUtils.delta(y2_pred, y_true)
        return [
            0
            if (sum_y1[i] == 0 and sum_y2[i] == 0)
            else
            np.abs(theta_y1[i] - theta_y2[i]) / (theta_y1[i] + theta_y2[i])
            for i in range(len(y_true))
        ]

    @staticmethod
    def wilcoxon_metric(y1_pred_list, y2_pred_list,y_true=None):
        import scipy.stats
        max_prob_indice = [np.argsort(y1_pred)[-1] for y1_pred in y1_pred_list]
        y1 = [y1_pred_list[i][max_prob_indice[i]] for i in range(len(max_prob_indice))]
        y2 = [y2_pred_list[i][max_prob_indice[i]] for i in range(len(max_prob_indice))]
        return scipy.stats.ranksums(y1, y2)

    @staticmethod
    def get_all_metrics():
        metrics_dict = {}
        metrics_dict['D_MAD'] = MetricsUtils.D_MAD_metrics
        metrics_dict['MAD'] = MetricsUtils.delta
        metrics_dict['Wilcoxon'] = MetricsUtils.wilcoxon_metric
        return metrics_dict

    @staticmethod
    def get_metrics_by_name(name):
        metrics = MetricsUtils.get_all_metrics()
        return metrics[name]

    @staticmethod
    def generate_result_by_metrics(metrics_list,lemon_results,save_dir,exp):

        for metrics_name in metrics_list:
            file_name = "{}/{}_{}_result.csv".format(save_dir,exp,metrics_name)
            metrics_result_dict = lemon_results[metrics_name]
            with open(file_name, "w") as writer:
                if metrics_name == 'Wilcoxon':
                    for dm_k, dm_v in metrics_result_dict.items():
                        writer.write("{}\n".format(dm_k))
                        for bk_pair, metrics_values in dm_v.items():
                            writer.write("{},{}\n".format(bk_pair, metrics_values))
                else:
                    writer.write("Mutation-Backend-Pair,Inconsistency Score\n")
                    for dm_k,dm_v in metrics_result_dict.items():
                        writer.write("{},{}\n".format(dm_k,dm_v))

if __name__ == '__main__':
    pass






