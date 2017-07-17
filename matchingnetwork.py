"""
Implementation of "Matching network for one short learning" in Keras
__author__ = Chetan Nichkawde
"""

import datanway as dataset
from keras.models import Model
from keras.layers import Flatten, Input, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from matchnn import MatchCosine

bsize = 32 # batch size
classes_per_set = 5 # classes per set or 5-way
samples_per_class = 1 # samples per class 1-short

data = dataset.OmniglotNShotDataset(batch_size=bsize,classes_per_set=classes_per_set,samples_per_class=samples_per_class,trainsize=64000,valsize=20000)

# Image embedding using Deep Convolutional Network
conv1 = Conv2D(64,(3,3),padding='same',activation='relu')
bnorm1 = BatchNormalization()
mpool1 = MaxPooling2D((2,2),padding='same')
conv2 = Conv2D(64,(3,3),padding='same',activation='relu')
bnorm2 = BatchNormalization()
mpool2 = MaxPooling2D((2,2),padding='same')
conv3 = Conv2D(64,(3,3),padding='same',activation='relu')
bnorm3 = BatchNormalization()
mpool3 = MaxPooling2D((2,2),padding='same')
conv4 = Conv2D(64,(3,3),padding='same',activation='relu')
bnorm4 = BatchNormalization()
mpool4 = MaxPooling2D((2,2),padding='same')
fltn = Flatten()

# Function that generarates Deep CNN embedding given the input image x
def convembedding(x):
    x = conv1(x)
    x = bnorm1(x)
    x = mpool1(x)
    x = conv2(x)
    x = bnorm2(x)
    x = mpool2(x)
    x = conv3(x)
    x = bnorm3(x)
    x = mpool3(x)
    x = conv4(x)
    x = bnorm4(x)
    x = mpool4(x)
    x = fltn(x)
    
    return x

numsupportset = samples_per_class*classes_per_set
input1 = Input((numsupportset+1,28,28,1))

modelinputs = []
for lidx in range(numsupportset):
    modelinputs.append(convembedding(Lambda(lambda x: x[:,lidx,:,:,:])(input1)))
targetembedding = convembedding(Lambda(lambda x: x[:,-1,:,:,:])(input1))
modelinputs.append(targetembedding)
supportlabels = Input((numsupportset,classes_per_set))
modelinputs.append(supportlabels)

knnsimilarity = MatchCosine(nway=classes_per_set)(modelinputs)

model = Model(inputs=[input1,supportlabels],outputs=knnsimilarity)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit([data.datasets_cache["train"][0],data.datasets_cache["train"][1]],data.datasets_cache["train"][2],
          validation_data=[[data.datasets_cache["val"][0],data.datasets_cache["val"][1]],data.datasets_cache["val"][2]],
          epochs=10,batch_size=32,verbose=1)



