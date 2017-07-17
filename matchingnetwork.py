"""
Implementation of "Matching network for one short learning" in Keras
__author__ = Chetan Nichkawde
"""

import datanway as dataset
from keras.models import Model
from keras.layers import Flatten, Input, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras.layers.merge import _Merge

bsize = 32 # batch size
classes_per_set = 5 # classes per set or 5-way
samples_per_class = 1 # samples per class 1-short

data = dataset.OmniglotNShotDataset(batch_size=bsize,classes_per_set=classes_per_set,samples_per_class=samples_per_class,
                                    trainsize=64000,valsize=20000)

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

class MatchCosine(_Merge):
    """
        Matching network with cosine similarity metric
    """
    def __init__(self,nway=5,**kwargs):
        super(MatchCosine,self).__init__(**kwargs)
        self.eps = 1e-10
        self.nway = nway

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != self.nway+2:
            raise ValueError('A ModelEuclidean layer should be called on a list of inputs of length %d'%(self.nway+2))

    def call(self,inputs):
        """
        inputs in as array which contains the support set the embeddings, the target embedding as the second last value in the array, and true class of target embedding as the last value in the array
        """ 
        similarities = []

        targetembedding = inputs[-2] # embedding of the query image
        numsupportset = len(inputs)-2
        for ii in range(numsupportset):
            supportembedding = inputs[ii] # embedding for i^{th} member in the support set

            sum_support = tf.reduce_sum(tf.square(supportembedding), 1, keep_dims=True)
            supportmagnitude = tf.rsqrt(tf.clip_by_value(sum_support, self.eps, float("inf"))) #reciprocal of the magnitude of the member of the support 

            sum_query = tf.reduce_sum(tf.square(targetembedding), 1, keep_dims=True)
            querymagnitude = tf.rsqrt(tf.clip_by_value(sum_query, self.eps, float("inf"))) #reciprocal of the magnitude of the query image

            dot_product = tf.matmul(tf.expand_dims(targetembedding,1),tf.expand_dims(supportembedding,2))
            dot_product = tf.squeeze(dot_product,[1])

            cosine_similarity = dot_product*supportmagnitude*querymagnitude
            similarities.append(cosine_similarity)

        similarities = tf.concat(axis=1,values=similarities)
        softmax_similarities = tf.nn.softmax(similarities)
        preds = tf.squeeze(tf.matmul(tf.expand_dims(softmax_similarities,1),inputs[-1]))
        
        preds.set_shape((inputs[0].shape[0],self.nway))

        return preds

    def compute_output_shape(self,input_shape):
        input_shapes = input_shape
        return (input_shapes[0][0],self.nway)

# Bonus: Matching network with Euclidean metrtic
class ModelEuclidean(_Merge):
    """
        Matching network with Euclidean metric
    """
    def __init__(self,nway=5,**kwargs):
        super(ModelEuclidean,self).__init__(**kwargs)
        self.eps = 1e-10
        self.nway = nway

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != self.nway+2:
            raise ValueError('A ModelEuclidean layer should be called on a list of inputs of length %d'%(self.nway+2))

    def call(self,inputs):
        """
        inputs in as array which contains the support set the embeddings, the target embedding as the second last value in the array, and true class of target embedding as the last value in the array
        """ 
        similarities = []

        targetembedding = inputs[-2]
        numsupportset = len(inputs)-2
        for ii in range(numsupportset):
            supportembedding = inputs[ii]
            dd = tf.negative(tf.sqrt(tf.reduce_sum(tf.square(supportembedding-targetembedding),1,keep_dims=True)))

            similarities.append(dd)

        similarities = tf.concat(axis=1,values=similarities)
        softmax_similarities = tf.nn.softmax(similarities)
        preds = tf.squeeze(tf.matmul(tf.expand_dims(softmax_similarities,1),inputs[-1]))
        
        preds.set_shape((inputs[0].shape[0],self.nway))

        return preds

    def compute_output_shape(self,input_shape):
        input_shapes = input_shape
        return (input_shapes[0][0],self.nway)

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



