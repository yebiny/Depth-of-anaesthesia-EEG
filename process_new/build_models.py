import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K
   
   
class Conv1DTranspose(tf.keras.layers.Layer):
     def __init__(self, filters, kernel_size, dilation_rate=1,  strides=1, padding='valid'):
         super().__init__()
         self.conv2dtranspose = tf.keras.layers.Conv2DTranspose(
           filters, (kernel_size, 1), (strides, 1), padding, dilation_rate =(dilation_rate,1)
         )

     def call(self, x):
         x = tf.expand_dims(x, axis=2)
         x = self.conv2dtranspose(x)
         x = tf.squeeze(x, axis=2)
         return x

class BuildModel():
    def __init__(self, img_shape, z_dim, n_conv_layers=3, dense_dim=256):
        self.img_shape = img_shape
        self.z_dim = z_dim
        self.n_conv_layers = n_conv_layers
        self.dense_dim = dense_dim

    def _sampling(self, args):
        """
        Reparameterization function by sampling from an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    def build_encoder(self):

        inputs = layers.Input(shape=self.img_shape)
        y = inputs 
        for rate in (1,2,4,8):
            y = layers.Conv1D( filters=20, kernel_size=2 
                             , padding='causal', activation='relu'
                             , dilation_rate=rate)(y)
            
        y = layers.Conv1D(20, 100, padding='same', activation = 'relu')(y)
        y = layers.AveragePooling1D(50, padding='same')(y)
        y = layers.Conv1D(1, 100, padding='same', activation = 'relu')(y)
        y = layers.AveragePooling1D(10, padding='same')(y)
        y = layers.Conv1D(self.z_dim, 10, padding='same')(y)
        y = layers.AveragePooling1D(10, padding='same')(y)
        
        y = layers.Reshape((self.z_dim, ))(y)

        z_mean = layers.Dense(self.z_dim, name="z_mean")(y)
        z_log_var = layers.Dense(self.z_dim, name="z_log_var")(y)
        z = layers.Lambda(self._sampling)([z_mean, z_log_var])
        
        encoder = models.Model(inputs, [z_mean, z_log_var, z], name ='encoder')
        
        return encoder
   


    def build_decoder(self):
        decoder_input = layers.Input(shape=(self.z_dim,))
        
        y = layers.Reshape((1, self.z_dim))(decoder_input)
        
        y = layers.AveragePooling1D(10, padding='same')(y)
        y = Conv1DTranspose(self.z_dim, 10, padding='same')(y)
        y = layers.Activation('relu')(y)
        y = layers.AveragePooling1D(10, padding='same')(y)
        y = Conv1DTranspose(1, 100, padding='same')(y)
        y = layers.Activation('relu')(y)
        y = layers.AveragePooling1D(50, padding='same')(y)
        y = Conv1DTranspose(20, 10, padding='same')(y)
       

        for rate in (8,4,2,1):
            y = Conv1DTranspose( filters=20, kernel_size=2 
                             , padding='causal'
                             , dilation_rate=rate)(y)
            y = layers.Activation('relu')(y)


        decoder = models.Model(decoder_input, y, name='decoder')
    
        return decoder

def main():
    
    builder = BuildModel((28,28,1), 100)
    encoder = builder.build_encoder()
    decoder = builder.build_decoder()
    
    encoder.summary()
    decoder.summary()

if __name__ == '__main__':
    main()
