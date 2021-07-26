import os, sys, glob
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from wavenet_utils import *

class WaveNet(tf.keras.Model):
    def __init__(self,
                 x_shape,
                 n_filters, 
                 n_categories,
                 n_residual_blocks,
                 mode='g',
                 use_skip_connections=True,
                 res_l2=0,
                 final_l2=0,
                 name='WaveNet',
                 **kwargs
                ):
        
        super(WaveNet, self).__init__(name=name, **kwargs)
        self.x_shape = x_shape
        self.n_filters = n_filters
        self.n_categories = n_categories
        self.n_residual_blocks = n_residual_blocks
        self.mode = mode,
        self.use_skip_connections=use_skip_connections,
        self.res_l2=res_l2
        self.final_l2=final_l2,

        self.blocks=[]
        for i in range(self.n_residual_blocks):
            block = self.residual_block(i, use_bias=False)
            self.blocks.append(block)
            
        self.model = self.build_model()
        
        
    def residual_block(self, i, use_bias=False):

        x = layers.Input(shape=(self.x_shape[0], self.n_filters), name='x_residual_%i'%i)
        f = CausalAtrousConvolution1D(self.n_filters, 2, 
                                      dilation_rate=2**i, 
                                      causal=True, 
                                      use_bias=use_bias,
                                      kernel_regularizer=l2(self.res_l2),
                                      name='f_%i'%i, 
                                      activation='tanh',
                                      )(x)

        g = CausalAtrousConvolution1D(self.n_filters, 2, 
                                      dilation_rate=2**i, 
                                      causal=True, 
                                      use_bias=use_bias,
                                      kernel_regularizer=l2(self.res_l2),
                                      name='g_%i'%i, 
                                      activation='sigmoid',
                                      )(x)

        z = layers.Multiply(name='multiply_%i'%i)([f,g])
        z = layers.Convolution1D(self.n_filters, 1, 
                                 padding='same', 
                                 use_bias=use_bias,
                                 kernel_regularizer=l2(self.res_l2),
                                 name = '1x1_conv_%i'%i,
                                 )(z)

        y = layers.Add(name='res_out_%i'%i)([x, z])
        block = models.Model(x, [y, z], name='Residual_Block_%i'%i)
        return block
    
    def generator_block(self, x):
                
        y = layers.Activation('relu', name='relu_1')(x)
        y = layers.Convolution1D(self.n_categories, 1, 
                                 padding='same',
                                 kernel_regularizer=l2(self.final_l2))(y)
        y = layers.Activation('relu', name='relu_2')(y)
        y = layers.Convolution1D(self.n_categories, 1, 
                                 padding='same')(y)
        y_out = layers.Activation('softmax', name='softmax')(y)

        return y_out
    
    def classifier_block(self, x):
                        
        y = layers.Activation('relu', name='relu_1')(x)
        
        y = layers.Convolution1D(self.n_categories, 1, padding='same',
                                 kernel_regularizer=l2(self.final_l2))(y)
        y = layers.AveragePooling1D(50, padding='same')(y)
        
        y = layers.Convolution1D(self.n_categories, 1, padding='same',
                                 kernel_regularizer=l2(self.final_l2))(y)
        y = layers.AveragePooling1D(int(self.x_shape[0]/50), padding='same')(y)

        y = layers.Reshape((self.n_categories, ))(y)
        y_out = layers.Activation('softmax', name='softmax')(y)

        return y_out
    
        
    def build_model(self):
        
        x_in = layers.Input(shape=self.x_shape)
        skip_connections = []


        y = CausalAtrousConvolution1D(self.n_filters, 2, 
                                      dilation_rate=1, 
                                      padding='valid',
                                      causal=True,
                                      name='init_conv', 
                                      )(x_in)        
        
        for ResidualBlock in self.blocks:
            y, z = ResidualBlock(y)
            skip_connections.append(z)

        if self.use_skip_connections==True:
            y = layers.Add(name='Skip_Connection')(skip_connections)

        if self.mode == 'g':
            y_out = self.generator_block(y)
        else:
            y_out = self.classifier_block(y)
        
        model = models.Model(x_in, y_out)
        
        return model
    
    def call(self, inputs):
        y = self.model(inputs)
        return y
