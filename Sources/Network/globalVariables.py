import  numpy as np

image_size = (3, 250, 250)

# HyperParams
beta1 = 0.9 # for decay of first-momentum
beta2 = 0.999 # for decay of second-momentum
alpha = 0.001 # initial learning rate
t = 0.5 # tolerance for how close images should be to be called "same"
feature_vector_size = 128
mini_batch_size = 5
amount_of_epochs = 3

eps = 1e-8 # to prevent dividing by 0

he_init = np.vectorize(lambda x, size: np.random.normal(loc=0, scale=np.sqrt(2/size))) # He initialization

ReLU = np.vectorize(lambda x: max(0.0, x))
dReLU = np.vectorize(lambda x: 1 if x>0 else 0)

linear = np.vectorize(lambda x: x)
dlinear = np.vectorize(lambda x: 1)