import numpy as np
import keras

np.random.seed(2191)  # for reproducibility

class OmniglotNShotDataset():
    def __init__(self,batch_size,classes_per_set=5,samples_per_class=1,trainsize=32000,valsize=10000):

        """
        Constructs an N-Shot omniglot Dataset
        :param batch_size: Experiment batch_size
        :param classes_per_set: Integer indicating the number of classes per set
        :param samples_per_class: Integer indicating samples per class
        e.g. For a 20-way, 1-shot learning task, use classes_per_set=20 and samples_per_class=1
             For a 5-way, 10-shot learning task, use classes_per_set=5 and samples_per_class=10
        """
        self.x = np.load("data.npy")
        self.x = np.reshape(self.x, [-1, 20, 28, 28, 1])
        shuffle_classes = np.arange(self.x.shape[0])
        np.random.shuffle(shuffle_classes)
        self.x = self.x[shuffle_classes]
        self.x_train, self.x_val  = self.x[:1200], self.x[1200:]
        self.normalization()

        self.batch_size = batch_size
        self.n_classes = self.x.shape[0]
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class

        self.indexes = {"train": 0, "val": 0}
        self.datasets = {"train": self.x_train, "val": self.x_val}
        self.datasets_cache = {"train": self.packslice(self.datasets["train"],trainsize),
                               "val": self.packslice(self.datasets["val"],valsize)}

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sd of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        print("train_shape", self.x_train.shape, "val_shape", self.x_val.shape)
        print("before_normalization", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_val = (self.x_val - self.mean) / self.std
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        print("after_normalization", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        
    def packslice(self, data_pack, numsamples):
        """
        Collects 1000 batches data for N-shot learning
        :param data_pack: Data pack to use (any one of train, val, test)
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        n_samples = self.samples_per_class * self.classes_per_set
        support_cacheX = []
        support_cacheY = []
        target_cacheY = []
        
        for iiii in range(numsamples):
            slice_x = np.zeros((n_samples+1,28,28,1))
            slice_y = np.zeros((n_samples,))
            
            ind = 0
            pinds = np.random.permutation(n_samples)
            classes = np.random.choice(data_pack.shape[0],self.classes_per_set,False) # chosen classes
            
            x_hat_class = np.random.randint(self.classes_per_set) # target class
            
            for j, cur_class in enumerate(classes):  # each class
                example_inds = np.random.choice(data_pack.shape[1],self.samples_per_class,False)
                
                for eind in example_inds:
                    slice_x[pinds[ind],:,:,:] = data_pack[cur_class][eind]
                    slice_y[pinds[ind]] = j
                    ind += 1
                
                if j == x_hat_class:
                    slice_x[n_samples,:,:,:] = data_pack[cur_class][np.random.choice(data_pack.shape[1])]
                    target_y = j

            support_cacheX.append(slice_x)
            support_cacheY.append(keras.utils.to_categorical(slice_y,self.classes_per_set))
            target_cacheY.append(keras.utils.to_categorical(target_y,self.classes_per_set))
            
        return np.array(support_cacheX), np.array(support_cacheY), np.array(target_cacheY)
