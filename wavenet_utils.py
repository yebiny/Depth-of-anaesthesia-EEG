import tensorflow as tf
from tensorflow.keras.layers import Conv1D
import tensorflow.keras.utils as conv_utils

def asymmetric_temporal_padding(x, left_pad=1, right_pad=1):
    '''Pad the middle dimension of a 3D tensor
    with "left_pad" zeros left and "right_pad" right.
    '''
    pattern = [[0, 0], [left_pad, right_pad], [0, 0]]
    return tf.pad(x, pattern)


class CausalAtrousConvolution1D(Conv1D):
    def __init__(self, 
                 filters, 
                 kernel_size, 
                 strides=1, 
                 dilation_rate=1, 
                 init='glorot_uniform', 
                 padding='valid', 
                 activation=None,
                 bias_regularizer=None,
                 activity_regularizer=None, 
                 kernel_constraint=None,
                 bias_constraint=None, 
                 use_bias=True, 
                 causal=False, 
                 **kwargs):
        
        super(CausalAtrousConvolution1D, self).__init__(filters,
                                                        kernel_size=kernel_size,
                                                        strides=strides,
                                                        dilation_rate=dilation_rate,
                                                        padding=padding,
                                                        activation=activation,
                                                        use_bias=use_bias,
                                                        kernel_initializer=init,
                                                        activity_regularizer=activity_regularizer,
                                                        bias_regularizer=bias_regularizer,
                                                        kernel_constraint=kernel_constraint,
                                                        bias_constraint=bias_constraint,
                                                        **kwargs)

        self.causal = causal
        if self.causal and padding != 'valid':
            raise ValueError("Causal mode dictates border_mode=valid.")

    def compute_output_shape(self, input_shape):
        input_length = input_shape[1]

        if self.causal:
            input_length += self.dilation_rate[0] * (self.kernel_size[0] - 1)

        length = conv_output_length(input_length,
                                    self.kernel_size[0],
                                    self.padding,
                                    self.strides[0],
                                    dilation=self.dilation_rate[0])

        return (input_shape[0], length, self.filters)

    def call(self, x):
        if self.causal:
            x = asymmetric_temporal_padding(x, self.dilation_rate[0] * (self.kernel_size[0] - 1), 0)
        return super(CausalAtrousConvolution1D, self).call(x)
