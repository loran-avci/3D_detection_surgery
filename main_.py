import numpy as np
import tensorflow as tf
from os.path import join
from glob import glob
from Model import StereoNeuralNetwork
from StereoDataGenerator import StereoDataGenerator
from Evaluation import Evaluator
import imgaug.augmenters as iaa

# Directories
data_root = r'../marco_data'
log_path = r'../logs'
checkpoints_path = r'../checkpoints'

# Train/Test-Split
train_filenames = glob(join(data_root,'train','img','*.png'))
train_filenames = [filename[:-5] for filename in train_filenames]
train_filenames = np.unique(train_filenames)
test_filenames = glob(join(data_root,'test','img','*.png'))
test_filenames = [filename[:-5] for filename in test_filenames]
test_filenames = np.unique(test_filenames)
eval_filenames = glob(join(data_root,'eval','img','*.png'))
eval_filenames = [filename[:-5] for filename in eval_filenames]
eval_filenames = np.unique(eval_filenames)

# Augmentation
seqs = []
seqs.append(iaa.Sequential([iaa.Resize(416)]).to_deterministic())
seqs.append(iaa.Sequential([iaa.Resize(416),iaa.Flipud()]).to_deterministic())
seqs.append(iaa.Sequential([iaa.Resize(416),iaa.Fliplr()]).to_deterministic())

# Data Generator
train_gen = StereoDataGenerator(train_filenames,shuffle=True,batch_size=8,resize_only=False,seqs=seqs,im_dim=416,output_dim=13)
test_gen = StereoDataGenerator(test_filenames,shuffle=False,batch_size=8,resize_only=True,seqs=seqs,im_dim=416,output_dim=13)
eval_gen = StereoDataGenerator(eval_filenames,shuffle=False,batch_size=8,resize_only=False,seqs=seqs,im_dim=416,output_dim=13)

# Model
model = StereoNeuralNetwork(im_size=416,channels=3,output_dim=13)

TB = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=0,write_graph=True, write_images=True)
CP = tf.keras.callbacks.ModelCheckpoint(join(checkpoints_path,'weights_{epoch:02d}.hdf5'), monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=25)

model.compile(optimizer='Adam',loss='mse')

# Train
model.fit(train_gen,epochs=150,callbacks=[TB,CP],validation_data=test_gen)

# Infer
inferred = np.array(model.predict(test_gen))

# Evaluate
#left_imgs,right_imgs,labels = eval_gen.get_data(eval_filenames)
#evaluator = Evaluator(inferred=labels,test_filenames=eval_filenames,im_width=416,im_height=416)
#evaluator.compareAfterAug(left_imgs,right_imgs)
evaluator = Evaluator(inferred=inferred,test_filenames=test_filenames,im_width=1280,im_height=720)
evaluator.run()

print('done')