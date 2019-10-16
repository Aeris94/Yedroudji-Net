# train data folder path
train_data_folder = 'data/train'

# test data folder path
test_data_folder = 'data/val'

# btach-size - number of images that are loaded at once to GPU
batch_size = 10
num_workers = 0

shuffle = True

# max number of epochs
epochs = 10000

# optimizer learining rate
lr = 0.01

# optimizer momentum
momentum = 0.95

# optimizer weight decay
weight_decay = 0.0001

# number of iterations in epoch after with validation on test set will be performed
validation_freq = 2

# lr scheduler gamma
gamma = 0.1
