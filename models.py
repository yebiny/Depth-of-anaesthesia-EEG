from tensorflow.keras import layers, models


def build_wavenet(x, nclass, optimizer, loss):
    
    xshape= x.shape

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=[xshape[1], xshape[2]]))
    
    for rate in (1,2,4,8):
        model.add(layers.Conv1D(filters=20, kernel_size=2, 
                                padding='causal', activation='relu', dilation_rate=rate))
    
    model.add(layers.AveragePooling1D(50, padding='same'))
    model.add(layers.Conv1D(20, 100, padding='same', activation='relu'))
    
    model.add(layers.Conv1D(1, 100, padding='same', activation='relu'))
    model.add(layers.AveragePooling1D(10, padding='same'))
    
   
    # Last layer - for label
    model.add(layers.Conv1D(nclass, 10, padding='same'))
    model.add(layers.AveragePooling1D(10, padding='same'))
    model.add(layers.Reshape((nclass, )))
    model.add(layers.Activation('softmax'))
    
    model.compile(optimizer, loss, metrics=['accuracy'])    
    
    return model


MODELS = {
    'wavenet': build_wavenet,
}

def get_model(model):
    return MODELS[model]
