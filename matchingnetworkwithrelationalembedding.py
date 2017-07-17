"""
Implementation of "Matching network for one short learning" in Keras.
A new kind of Full Context Embedding is define here which uses 
Siamese like pairwise interaction and does a max pooling on these interaction.
The pooled out is then forwarded to a multi layer perceptron.

__author__ = Chetan Nichkawde
"""

import datanway as dataset
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Lambda
from keras.layers.merge import Maximum
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from matchnn import Siamify, MatchEuclidean
from itertools import combinations
from collections import defaultdict

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

# Relational embedding comprising a 4 layer MLP
d1 = Dense(64,activation='relu')
dbnrm1 = BatchNormalization()
d2 = Dense(64,activation='relu')
dbnrm2 = BatchNormalization()
d3 = Dense(64,activation='relu')
dbnrm3 = BatchNormalization()
d4 = Dense(64,activation='relu')
dbnrm4 = BatchNormalization()

def relationalembedding(x):
    x = d1(x)
    x = dbnrm1(x)
    x = d2(x)
    x = dbnrm2(x)
    x = d3(x)
    x = dbnrm3(x)
    x = d4(x)
    x = dbnrm4(x)
    
    return x

numsupportset = samples_per_class*classes_per_set
input1 = Input((numsupportset+1,28,28,1))

# CNN embedding support set and query image
convolutionlayers = []
for lidx in range(numsupportset):
    convolutionlayers.append(convembedding(Lambda(lambda x: x[:,lidx,:,:,:])(input1)))
targetembedding = convembedding(Lambda(lambda x: x[:,-1,:,:,:])(input1))

siam = Siamify()
pairwiseinteractions = defaultdict(list)

# Get all pairwise Siamese interactions in the support set and generate of list of interactions
# for each member of support set 
for tt in combinations(range(numsupportset),2):
    aa = siam([convolutionlayers[tt[0]],convolutionlayers[tt[1]]])
    pairwiseinteractions[tt[0]].append(aa)
    pairwiseinteractions[tt[1]].append(aa)

# Get Siamese interactions for query image
targetinteractions = []
for i in range(numsupportset):
    aa = siam([targetembedding,convolutionlayers[i]])
    targetinteractions.append(aa)  
    pairwiseinteractions[i].append(aa) # add this interaction to the set of interaction for this member

# Take 4 layer MLP transform on Mak pooling of interactions to serve as Full Context Embedding (FCE)
maxi = Maximum()
modelinputs = []
for i in range(numsupportset):
    modelinputs.append(relationalembedding(maxi(pairwiseinteractions[i])))
modelinputs.append(relationalembedding(maxi(targetinteractions)))

supportlabels = Input((numsupportset,classes_per_set))
modelinputs.append(supportlabels)

knnsimilarity = MatchEuclidean(nway=classes_per_set)(modelinputs)

model = Model(inputs=[input1,supportlabels],outputs=knnsimilarity)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit([data.datasets_cache["train"][0],data.datasets_cache["train"][1]],data.datasets_cache["train"][2],
          validation_data=[[data.datasets_cache["val"][0],data.datasets_cache["val"][1]],data.datasets_cache["val"][2]],
          epochs=50,batch_size=32,verbose=1)
