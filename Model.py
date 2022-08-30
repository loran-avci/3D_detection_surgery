import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Concatenate, BatchNormalization, Reshape, Dropout, Input,LeakyReLU


def StereoNeuralNetwork3(im_size, channels, output_dim, rate=0.3):
	### SETUP MODEL ###
	left_input = Input(shape=(im_size,im_size,channels))
	right_input = Input(shape=(im_size,im_size,channels))

	### LEFT ###
	model = Sequential()
	 
	model.add(BatchNormalization(input_shape=(im_size,im_size,channels)))
	model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D((4,4)))
	 
	model.add(BatchNormalization())
	model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D((4,4)))

	model.add(BatchNormalization())
	model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D((4,4)))

	model.add(BatchNormalization())
	model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D(4,4))

	stereo = Concatenate()([model(left_input), model(right_input)])

	stereo = BatchNormalization()(stereo)   

	union_branch = Conv2D(128,(1,1),activation='relu')(stereo)
	union_branch = Flatten()(union_branch)

	union_branch = Dense(1024)(union_branch) 
	union_branch = LeakyReLU(0.1)(union_branch)
	union_branch = Dropout(rate)(union_branch)

	union_branch = Dense(1024)(union_branch) 
	union_branch = LeakyReLU(0.1)(union_branch)
	union_branch = Dropout(rate)(union_branch)
	
	union_branch = Dense(1024)(union_branch) 
	union_branch = LeakyReLU(0.1)(union_branch)
	union_branch = Dropout(rate)(union_branch)
	
	union_branch = Dense(1024)(union_branch) 
	union_branch = LeakyReLU(0.1)(union_branch)
	union_branch = Dropout(rate)(union_branch)
	
	out_1 = Dense(output_dim*output_dim*1,activation='sigmoid')(union_branch)
	out_1 = Reshape((output_dim,output_dim,1),name = 'detection')(out_1)
	
	out_2 = Dense(output_dim*output_dim*6,activation='linear')(union_branch)
	out_2 = Reshape((output_dim,output_dim,6),name = 'params')(out_2)
	
	stereo_model = Model(inputs=[left_input,right_input], outputs=[out_1, out_2])
	#stereo_model.summary()
	
	return stereo_model
def StereoNeuralNetwork2(im_size, channels, output_dim):
	### SETUP MODEL ###
	left_input = Input(shape=(im_size,im_size,channels))
	right_input = Input(shape=(im_size,im_size,channels))

	### LEFT ###
	model = Sequential()
	model.add(BatchNormalization(input_shape=(im_size,im_size,channels)))
	model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D((4,4)))

	model.add(BatchNormalization())
	model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D((4,4)))

	model.add(BatchNormalization())
	model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D())

	stereo = Concatenate()([model(left_input), model(right_input)])

	stereo = BatchNormalization()(stereo)   

	union_branch = Conv2D(64,(1,1),activation='relu')(stereo)
	union_branch = Flatten()(union_branch)
	union_branch = Dense(1024,activation='relu')(union_branch)
	union_branch = Dense(1024,activation='relu')(union_branch)
	union_branch = Dense(1024,activation='relu')(union_branch)
	union_branch = Dense(1024,activation='relu')(union_branch)
	
	out_1 = Dense(output_dim*output_dim*1,activation='sigmoid')(union_branch)
	out_1 = Reshape((output_dim,output_dim,1),name = 'detection')(out_1)
	
	out_2 = Dense(output_dim*output_dim*6,activation='linear')(union_branch)
	out_2 = Reshape((output_dim,output_dim,6),name = 'params')(out_2)
	
	
	stereo_model = Model(inputs=[left_input,right_input], outputs=[out_1, out_2])
	
	stereo_model.summary()
	
	return stereo_model


def StereoNeuralNetwork(im_size, channels, output_dim):
	### SETUP MODEL ###
	left_input = Input(shape=(im_size,im_size,channels))
	right_input = Input(shape=(im_size,im_size,channels))

	### LEFT ###
	model = Sequential()
	model.add(BatchNormalization(input_shape=(im_size,im_size,channels)))
	model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D((4,4)))

	model.add(BatchNormalization())
	model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D((4,4)))

	model.add(BatchNormalization())
	model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D())

	stereo = Concatenate()([model(left_input), model(right_input)])

	stereo = BatchNormalization()(stereo)   

	union_branch = Conv2D(64,(1,1),activation='relu')(stereo)
	union_branch = Flatten()(union_branch)
	union_branch = Dense(1024,activation='relu')(union_branch)
	union_branch = Dense(1024,activation='relu')(union_branch)
	union_branch = Dense(1024,activation='relu')(union_branch)
	union_branch = Dense(1024,activation='relu')(union_branch)
	union_branch = Dense(output_dim*output_dim*7,activation='linear')(union_branch)
	union_branch = Reshape((output_dim,output_dim,7))(union_branch)

	stereo_model = Model(inputs=[left_input,right_input], outputs=[union_branch])

	stereo_model.summary()

	return stereo_model

def StereoNeuralNetwork_new(im_size, channels, output_dim):
	### SETUP MODEL ###
	left_input = Input(shape=(im_size,im_size,channels))
	right_input = Input(shape=(im_size,im_size,channels))

	### LEFT ###
	model = Sequential()
	model.add(BatchNormalization(input_shape=(im_size,im_size,channels)))
	model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D((2,2)))

	model.add(BatchNormalization())
	model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D((2,2)))

	model.add(BatchNormalization())
	model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D((2,2)))

	model.add(BatchNormalization())
	model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(256,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D((2,2)))

	model.add(BatchNormalization())
	model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
	model.add(Conv2D(512,(3,3),padding='same',activation='relu'))
	model.add(MaxPooling2D())

	stereo = Concatenate()([model(left_input), model(right_input)])

	stereo = BatchNormalization()(stereo)   

	union_branch = Conv2D(128,(1,1),activation='relu')(stereo)
	union_branch = Flatten()(union_branch)
	union_branch = Dense(1024,activation='relu')(union_branch)
	union_branch = Dense(1024,activation='relu')(union_branch)
	union_branch = Dense(1024,activation='relu')(union_branch)
	union_branch = Dense(1024,activation='relu')(union_branch)
	union_branch = Dense(output_dim*output_dim*6,activation='linear')(union_branch)
	union_branch = Reshape((output_dim,output_dim,6))(union_branch)

	stereo_model = Model(inputs=[left_input,right_input], outputs=[union_branch])

	stereo_model.summary()

	return stereo_model
