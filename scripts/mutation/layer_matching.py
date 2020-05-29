import os
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class LayerMatching:
    concat_size_limit = 1e4

    def __init__(self):
        self.layers = {}
        self.constraints = {}

        self.layers['flatten'] = LayerMatching.flatten
        self.constraints['flatten'] = LayerMatching.flatten_constraints

        self.layer_concats = {}
        self.input_legal = {}
        self.layer_concats['flatten'] = LayerMatching.flatten_dense
        self.input_legal['flatten'] = LayerMatching.flatten_dense_input_legal
        self.layer_concats['repeat_vector'] = LayerMatching.repeat_vector_dense
        self.input_legal['repeat_vector'] = LayerMatching.repeat_vector_dense_input_legal
        self.layer_concats['cropping1d'] = LayerMatching.cropping1d_dense
        self.input_legal['cropping1d'] = LayerMatching.cropping1d_dense_input_legal
        self.layer_concats['cropping2d'] = LayerMatching.cropping2d_dense
        self.input_legal['cropping2d'] = LayerMatching.cropping2d_dense_input_legal
        self.layer_concats['cropping3d'] = LayerMatching.cropping3d_dense
        self.input_legal['cropping3d'] = LayerMatching.cropping3d_dense_input_legal
        self.layer_concats['upsampling_1d'] = LayerMatching.upsampling_1d_dense
        self.input_legal['upsampling_1d'] = LayerMatching.upsampling_1d_dense_input_legal
        self.layer_concats['upsampling_2d'] = LayerMatching.upsampling_2d_dense
        self.input_legal['upsampling_2d'] = LayerMatching.upsampling_2d_dense_input_legal
        self.layer_concats['upsampling_3d'] = LayerMatching.upsampling_3d_dense
        self.input_legal['upsampling_3d'] = LayerMatching.upsampling_3d_dense_input_legal
        self.layer_concats['zeropadding_1d'] = LayerMatching.zeropadding_1d_conv
        self.input_legal['zeropadding_1d'] = LayerMatching.zeropadding_1d_conv_input_legal
        self.layer_concats['zeropadding_2d'] = LayerMatching.zeropadding_2d_conv
        self.input_legal['zeropadding_2d'] = LayerMatching.zeropadding_2d_conv_input_legal
        self.layer_concats['zeropadding_3d'] = LayerMatching.zeropadding_3d_conv
        self.input_legal['zeropadding_3d'] = LayerMatching.zeropadding_3d_conv_input_legal
        self.layer_concats['global_max_pooling_1d'] = LayerMatching.global_max_pooling_1d_dense
        self.input_legal['global_max_pooling_1d'] = LayerMatching.global_pooling_1d_dense_input_legal
        self.layer_concats['global_average_pooling_1d'] = LayerMatching.global_average_pooling_1d_dense
        self.input_legal['global_average_pooling_1d'] = LayerMatching.global_pooling_1d_dense_input_legal
        self.layer_concats['global_max_pooling_2d'] = LayerMatching.global_max_pooling_2d_dense
        self.input_legal['global_max_pooling_2d'] = LayerMatching.global_pooling_2d_dense_input_legal
        self.layer_concats['global_average_pooling_2d'] = LayerMatching.global_average_pooling_2d_dense
        self.input_legal['global_average_pooling_2d'] = LayerMatching.global_pooling_2d_dense_input_legal
        self.layer_concats['global_max_pooling_3d'] = LayerMatching.global_max_pooling_3d_dense
        self.input_legal['global_max_pooling_3d'] = LayerMatching.global_pooling_3d_dense_input_legal
        self.layer_concats['global_average_pooling_3d'] = LayerMatching.global_average_pooling_3d_dense
        self.input_legal['global_average_pooling_3d'] = LayerMatching.global_pooling_3d_dense_input_legal
        self.layer_concats['simple_rnn'] = LayerMatching.simple_rnn_dense
        self.input_legal['simple_rnn'] = LayerMatching.simple_rnn_dense_input_legal
        self.layer_concats['gru'] = LayerMatching.gru_dense
        self.input_legal['gru'] = LayerMatching.gru_dense_input_legal
        self.layer_concats['lstm'] = LayerMatching.lstm_dense
        self.input_legal['lstm'] = LayerMatching.lstm_dense_input_legal
        self.layer_concats['conv_lstm_2d'] = LayerMatching.conv_lstm_2d_dense
        self.input_legal['conv_lstm_2d'] = LayerMatching.conv_lstm_2d_dense_input_legal

    @staticmethod
    def flatten(input_shape):
        import keras
        return keras.layers.Flatten()

    @staticmethod
    def flatten_constraints(input_shape):
        input_shape = input_shape.as_list()
        input_shape_len = len(input_shape)
        constraints = []
        if input_shape_len < 2:
            return None
        constraints = []
        dim_size = 1
        for i in range(input_shape_len):
            if i == 0:
                continue
            constraints.append('= input_{} {}'.format(i, input_shape[i]))
            dim_size *= input_shape[i]
        constraint_str = '= output_{} {}'.format(1, dim_size)
        constraints.append(constraint_str)
        return constraints

    # --------------------------------------------

    @staticmethod
    def flatten_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.Flatten())
        units = 1
        for i in range(len(input_shape)):
            if i == 0:
                continue
            units *= input_shape[i]
        layer_concat.append(keras.layers.Dense(units))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def flatten_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        is_legal = len(input_shape) > 3 and input_shape[0] is None
        concat_size = 1
        for i, dim in enumerate(input_shape):
            if i == 0:
                continue
            is_legal = is_legal and dim is not None
            if dim is not None:
                concat_size *= dim
        return is_legal and concat_size <= LayerMatching.concat_size_limit

    @staticmethod
    def repeat_vector_dense(input_shape):
        n = 3
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.RepeatVector(n))
        layer_concat.append(keras.layers.Reshape((input_shape[1] * n,)))
        layer_concat.append(keras.layers.Dense(input_shape[1]))
        return layer_concat

    @staticmethod
    def repeat_vector_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 2 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[1] <= LayerMatching.concat_size_limit

    @staticmethod
    def cropping1d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.Cropping1D(cropping=(1, 1)))
        layer_concat.append(keras.layers.Dense(input_shape[1]))
        return layer_concat

    @staticmethod
    def cropping1d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None and input_shape[1] > 2 \
               and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def cropping2d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.Cropping2D(cropping=((1, 1), (1, 1))))
        layer_concat.append(keras.layers.Reshape(((input_shape[1] - 2) * (input_shape[2] - 2) * input_shape[3],)))
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def cropping2d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[1] > 2 \
               and input_shape[2] is not None and input_shape[2] > 2 \
               and input_shape[3] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] <= LayerMatching.concat_size_limit

    @staticmethod
    def cropping3d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.Cropping3D(cropping=((1, 1), (1, 1), (1, 1))))
        layer_concat.append(keras.layers.Reshape(((input_shape[1] - 2) * (input_shape[2] - 2) * (input_shape[3] - 2) * input_shape[4],)))
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def cropping3d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[1] > 2 \
               and input_shape[2] is not None and input_shape[2] > 2 \
               and input_shape[3] is not None and input_shape[3] > 2 \
               and input_shape[4] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit

    @staticmethod
    def upsampling_1d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.UpSampling1D(size=2))
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2]))
        return layer_concat

    @staticmethod
    def upsampling_1d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def upsampling_2d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.UpSampling2D(size=(2, 2)))
        layer_concat.append(keras.layers.Flatten())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def upsampling_2d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[2] is not None and input_shape[3] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] <= LayerMatching.concat_size_limit

    @staticmethod
    def upsampling_3d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.UpSampling3D(size=(2, 2, 2)))
        layer_concat.append(keras.layers.Flatten())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def upsampling_3d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None \
               and input_shape[4] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit

    @staticmethod
    def zeropadding_1d_conv(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.ZeroPadding1D(padding=1))
        layer_concat.append(keras.layers.Conv1D(input_shape[-1], 3))
        return layer_concat

    @staticmethod
    def zeropadding_1d_conv_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None \
               and input_shape[1] is not None and input_shape[2] is not None \
               and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def zeropadding_2d_conv(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.ZeroPadding2D(padding=(1, 1)))
        layer_concat.append(keras.layers.Conv2D(input_shape[-1], 3))
        return layer_concat

    @staticmethod
    def zeropadding_2d_conv_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] <= LayerMatching.concat_size_limit

    @staticmethod
    def zeropadding_3d_conv(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.ZeroPadding3D(padding=(1, 1, 1)))
        layer_concat.append(keras.layers.Conv3D(input_shape[-1], 3))
        return layer_concat

    @staticmethod
    def zeropadding_3d_conv_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None \
               and input_shape[4] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit

    @staticmethod
    def global_max_pooling_1d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.GlobalMaxPooling1D())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def global_average_pooling_1d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.GlobalAveragePooling1D())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def global_pooling_1d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def global_max_pooling_2d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.GlobalMaxPooling2D())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def global_average_pooling_2d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.GlobalAveragePooling2D())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def global_pooling_2d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 4 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] <= LayerMatching.concat_size_limit

    @staticmethod
    def global_max_pooling_3d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.GlobalMaxPooling3D())
        layer_concat.append(keras.layers.Flatten())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def global_average_pooling_3d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.GlobalAveragePooling3D())
        layer_concat.append(keras.layers.Flatten())
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def global_pooling_3d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[3] is not None \
               and input_shape[4] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit

    @staticmethod
    def simple_rnn_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.SimpleRNN(50))
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def simple_rnn_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None \
               and input_shape[1] is not None \
               and input_shape[2] is not None \
               and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def gru_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.GRU(50))
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def gru_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def lstm_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.LSTM(50))
        layer_concat.append(keras.layers.Dense(input_shape[1] * input_shape[2]))
        layer_concat.append(keras.layers.Reshape(input_shape[1:]))
        return layer_concat

    @staticmethod
    def lstm_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 3 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[1] * input_shape[2] <= LayerMatching.concat_size_limit

    @staticmethod
    def conv_lstm_2d_dense(input_shape):
        import keras
        layer_concat = []
        layer_concat.append(keras.layers.ConvLSTM2D(input_shape[-1], kernel_size=(1, 1), strides=(1, 1), padding='same', return_sequences=True))
        return layer_concat

    @staticmethod
    def conv_lstm_2d_dense_input_legal(input_shape):
        input_shape = input_shape.as_list()
        return len(input_shape) == 5 and input_shape[0] is None and input_shape[1] is not None \
               and input_shape[2] is not None and input_shape[2] > 3 \
               and input_shape[3] is not None and input_shape[3] > 3 \
               and input_shape[4] is not None \
               and input_shape[1] * input_shape[2] * input_shape[3] * input_shape[4] <= LayerMatching.concat_size_limit

if __name__ == '__main__':
    pass