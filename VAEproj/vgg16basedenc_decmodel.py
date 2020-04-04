# vgg 16 based artitecture encoder-decoder
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,Concatenate
import numpy as np

# inputdim = _
inputshape = (224,224,)


inp1 = Input(shape=input_shape,name="1")
# input2 = Input(shape=input_shape,name="1")

x2 = Conv2D(64,(3,3),activation = 'relu',padding = 'same',name = '2')(input1)
x3 = Conv2D(64,(3,3),activation = 'relu',padding = 'same',name = '2')(x2)
x4 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x3)
# 

y13 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x4)
y12 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(y13)
y11 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(y12)


# x = Input(shape=input_shape,name="1")
x5 = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = '2')(x4)
x6 = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = '2')(x5)
x7 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x6)

# 

y22 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x7)
y21 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(y22)


# x = Input(shape=input_shape,name="1")


# 3
# layer for concatenation
c1 = Concatenate([x7,y13])
x8 = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = '2')(c1)
x9 = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = '2')(x8)
x10 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x9)


y31 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x10)
# y21 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(y22)

# 2
# layer for concatenation
c2 = Concatenate([x10,y12,y22])

x11 = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = '2')(c2)
x12 = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = '2')(x11)
x13 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x12)

# 1
# layer for concatenation

c3 = Concatenate([x13,y11,y21,y31])
x14 = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = '2')(x13)
x15 = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = '2')(x14)
x16 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x15)

x17 = Flatten()(x16)

featureextractor = Model(inp1,x17,name = "featureextractor")



# input1 = featureextractor(input1)


input1 = Input(shape=(224,224,), name='encoder_input_im')
input2 = Input(shape=(224,224,), name='encoder_input_imsum')


input3 = Input(shape=(classifieroutputdim,),name='encoder_input_labels')

x = Dense(intermediate_dim, activation='relu')(featureextractor(input1))


x = Dense(intermediate_dim1, activation='relu')(x)
# drop1 = Dropout(0.2)(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
inps = [input1,input2,input3]
# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
# instantiate encoder model
# encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
# encoder = Model(inputs = [input1,input2], [z_mean, z_log_var, z], name='encoder')
encoder = Model([featureextractor(input1),featureextractor(input2),input3], [z_mean, z_log_var, z], name='encoder')
encoder.summary()
# plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_samples')

x = Dense(intermediate_dim, activation='relu')(latent_inputs)
# drop2 = Dropout(0.2)(x)
# x = Dense(intermediate_dim1, activation='relu')(x)
x = Dense(intermediate_dim1, activation='relu')(x)
# x = Dense(intermediate_dim1, activation='relu')(x)

outputs = Dense(original_dim, activation='sigmoid')(x)


# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
# plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

#building classifier models
classifier_inputs = Input(shape=(latent_dim,), name="class1")

x = Dense(classifierdim,activation = 'relu')(classifier_inputs)
# x = Dense(classifierdim,activation = 'relu')(x)  # additional layer
classifier_output = Dense(classifieroutputdim,activation = 'softmax')(x)

classifier = Model(classifier_inputs,classifier_output,name="classifier")

classifier.summary()

# instantiate VAE model
output1 = decoder(encoder(inps)[2])
output2 = classifier(encoder(inps)[0]) # classifing on smaple values
# output2 = classifier(encoder(inps)[0]) # classifing on z_means
outputs = [output1,output2]

vae = Model(inps, outputs, name='vae_mlp + classifier + vgg16')





