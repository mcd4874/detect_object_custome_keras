from model import detectModel
from shapes import create_training_data
from tensorflow import keras as keras
import tensorflow as tf
n_epochs = 2
img_size = 200
max_radius = 50
noise_level = 2
n_train_sample = 30
n_test_sample = 10
n_valid_sample = 10
train_data,train_label = create_training_data(n_train_sample,img_size,max_radius,noise_level)
test_data,test_label = create_training_data(n_test_sample,img_size,max_radius,noise_level)
valid_data,valid_label = create_training_data(n_valid_sample,img_size,max_radius,noise_level)


#custome optimizer
# cus_optimizer = keras.optimizers.SGD(lr=0.08)
# cus_optimizer = tf.train.GradientDescentOptimizer(learning_rate=.08)
model = detectModel(img_size)

#train
model.train(train_data,train_label,epochs=n_epochs)

#predict
predict_test_label = model.predictions(test_data)
print(test_label)
print(predict_test_label)
# def train()

#evaluate
# print(model.evaluate(valid_data,valid_label))