from keras.models import Sequential, Model

from keras.layers import (Dense,
                          BatchNormalization,
                          MaxPooling1D,
                          AveragePooling1D,
                          Convolution1D,
                          Merge,
                          Dropout,
                          BatchNormalization,
                          Flatten,
                          Convolution2D,
                          MaxPooling2D,
                          Activation,
                          merge,
                          Input,
                          Lambda,
                          Reshape)

from keras.layers.advanced_activations import LeakyReLU
from keras.activations import softmax

from sklearn.cross_validation import StratifiedShuffleSplit

def res_unit(inputs, out, length, leak=0.0):
    
    if inputs._keras_shape[2] != out:
        inputs = Convolution1D(out, length, border_mode='same')(inputs)
    
    x = BatchNormalization()(inputs)
    x = Activation(LeakyReLU(alpha=leak))(x)
    x = Convolution1D(out, length, border_mode='same')(x)
    
    x = BatchNormalization()(x)
    x = Activation(LeakyReLU(alpha=leak))(x)
    x = Convolution1D(out, length, border_mode='same')(x)
    
    x = merge([inputs, x], mode='sum')
    
    return x

def new_resnet(inputs, outputs, layers):
    inputs = Input(shape=inputs)
    
    print inputs._keras_shape
    
    x = Convolution1D(nb_filter=64, filter_length=7, border_mode='same', subsample_length=2)(inputs)
    
    x = BatchNormalization()(x)
    x = Activation(LeakyReLU(alpha=0.0))(x)
    
    print x._keras_shape
    
    #x = MaxPooling1D(pool_length=4, stride=2, border_mode='valid')(x)
    
    print x._keras_shape
    
    nb_filters = 64
    for i in range(layers):
        x = res_unit(x, nb_filters, 1)
        
        print x._keras_shape
        
        nb_filters *= 2
    
    print x._keras_shape
    
    x = AveragePooling1D(pool_length=x._keras_shape[1], stride=32, border_mode='valid')(x)
    x = Flatten()(x)
    
    print x._keras_shape
    
    output = Dense(outputs, activation='softmax')(x)
    model = Model(inputs, output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
    
    
def res_block_1d(inputs, kernels, leak=0.0, shortcut=False):
    
    for i, k in enumerate(kernels):
        if i == 0:
            x = Convolution1D(k, 1, border_mode='same')(inputs)
        else:
            x = Convolution1D(k, 1, border_mode='same')(x)
            
        x = BatchNormalization()(x)
        if i == len(kernels):
            if shortcut:
                short = Convolution1D(k, 1, border_mode='same')
                short = BatchNormalization(short)
                x = merge([x, short], mode='sum')
            else:
                x = merge([x, input], mode='sum')
                
        x = Activation(LeakyReLU(alpha=leak))(x)
    return x

def ResNet(input_shape, output_shape, optimizer='adam', leak=0.0):
    
    inputs = Input(shape=input_shape)
    
    x = Convolution1D(64, 3, border_mode='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = MaxPooling1D(pool_length=2, border_mode='valid')(x)
    
    x = res_block_1d(x, [64, 64, 256], leak=leak, shortcut=True)
    x = res_block_1d(x, [64, 64, 256], leak=leak, shortcut=False)
    x = res_block_1d(x, [64, 64, 256], leak=leak, shortcut=False)
    
    x = res_block_1d(x, [128, 128, 512], leak=leak, shortcut=True)
    x = res_block_1d(x, [128, 128, 512], leak=leak, shortcut=False)
    x = res_block_1d(x, [128, 128, 512], leak=leak, shortcut=False)
    x = res_block_1d(x, [128, 128, 512], leak=leak, shortcut=False)
    
    x = res_block_1d(x, [256, 256, 1024], leak=leak, shortcut=True)
    x = res_block_1d(x, [256, 256, 1024], leak=leak, shortcut=False)
    x = res_block_1d(x, [256, 256, 1024], leak=leak, shortcut=False)
    x = res_block_1d(x, [256, 256, 1024], leak=leak, shortcut=False)
#     x = res_block_1d(x, [256, 256, 1024], leak=leak, shortcut=False)
#     x = res_block_1d(x, [256, 256, 1024], leak=leak, shortcut=False)
    
#     x = res_block_1d(x, [512, 512, 2048], leak=leak, shortcut=True)
#     x = res_block_1d(x, [512, 512, 2048], leak=leak, shortcut=False)
#     x = res_block_1d(x, [512, 512, 2048], leak=leak, shortcut=False)
    
    x = AveragePooling1D(pool_length=4)(x)
    
    x = Flatten()(x)
    outputs = Dense(output_shape, activation='softmax')(x)
    
    model = Model(inputs, outputs)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model
    
def muti_time_steps(steps, conv_shape, timesteps, input, pool=2, leak=0.3, drop=0.5, time_leak=0.2, time_drop=0.1):
    
    for i in range(steps):
    
        c2 = Convolution1D(conv_shape[0], conv_shape[1], border_mode='same')(input)
        b2 = BatchNormalization()(c2)
        a2 = Activation(LeakyReLU(alpha=leak))(b2)

        inner = shared_weights_steps(steps=timesteps,
                                     conv_shape=conv_shape,
                                     input=a2, 
                                     conv=c2,
                                     leak=time_leak,
                                     drop=time_drop)

        p2 = MaxPooling1D(pool_length=pool, stride=None, border_mode='valid')(inner)
        d2 = Dropout(p=drop)(p2)
        
        input = d2
        
    return input

def shared_weights_steps(steps, conv_shape, input, conv, leak, drop):

    for i in range(steps):
        if i == 0:
            _c2 = Convolution1D(conv_shape[0], conv_shape[1], border_mode='same')
    
        c2a = _c2(input)
        s2a = merge([conv, c2a], mode='sum')
        b2a = BatchNormalization()(s2a)
        a2a = Activation(LeakyReLU(alpha=leak))(b2a)
        d2a = Dropout(p=drop)(a2a)
        
        input = d2a
    
    return input

def RCL(layers=3,
        conv_shape=(256, 9),
        timesteps=3,
        time_conv_shape=(256, 9),
        input_shape=(1024, 124),
        targets=5, 
        optimizer='adam',
        pool=2,
        leak=0.3,
        drop=0.5,
        time_leak=0.2,
        time_drop=0.1):
    
    inputs = Input(shape=input_shape)
    
    c1 = Convolution1D(conv_shape[0], conv_shape[1], border_mode='same')(inputs)
    b1 = BatchNormalization()(c1)
    a1 = Activation(LeakyReLU(alpha=leak))(b1)
    p1 = MaxPooling1D(pool_length=pool, stride=None, border_mode='valid')(a1)
    d1 = Dropout(p=drop)(p1)
    
    inner = muti_time_steps(steps=layers,
                            conv_shape=conv_shape,
                            timesteps=timesteps,
                            input=d1,
                            leak=leak,
                            drop=drop,
                            time_leak=time_leak,
                            time_drop=time_drop)
    
    flat = Flatten()(inner)
    
    output = Dense(targets, activation='softmax')(flat)
    
    model = Model(input=inputs, output=output)
        
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


def SIMPLE(layers, input_shape, targets, optimizer='adam', pool=1, leak=0.0, drop=0.0):
    
    inputs = Input(shape=input_shape)
    
    for i in range(layers):
        if i == 0:
            input = inputs
    
        conv1 = Convolution1D(256, 9, border_mode='same')(input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(LeakyReLU(alpha=leak))(conv1)
        conv1 = MaxPooling1D(pool_length=pool, stride=None, border_mode='valid')(conv1)
        conv1 = Dropout(p=0.0)(conv1)
        
        input = conv1
    
    flat = Flatten()(input)
    flat = Dense(512, activation='relu')(flat)
    flat = Dropout(p=drop)(flat)
    
    output = Dense(targets, activation='softmax')(flat)
    
    model = Model(input=inputs, output=output)
        
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model