import tensorflow as tf
import numpy as np
from tensorflow import keras as keras
from shapely.geometry.point import Point
import matplotlib.pyplot as plt
import os
class detectModel:
    def __init__(self, image_size,save_path = None,output_size = 3, model = None, optimizer = None,loss_function = None):
        self.size = image_size
        self.model = model
        self.optimizer = optimizer
        self.loss = loss_function
        self.output_size = output_size
        self.checkpoint_path = save_path
        if self.model == None:
            self.model = self.create_model()
        if self.optimizer == None:
            self.optimizer = self.create_optimizer()
        if self.loss == None:
            self.loss = self.create_loss_function()

        if self.checkpoint_path == None:
            self.checkpoint_path = "training_1/cp.ckpt"
            # os.path.dirname(self.save_path)

    def create_model(self):
        model = keras.Sequential()

        # Must define the input shape in the first layer of the neural network
        model.add(keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(self.size, self.size, 1)))
        # model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(pool_size=2,strides=2))
        # model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'))
        # model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(pool_size=2,strides=2))
        # model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(self.output_size))

        # Take a look at the model summary
        model.summary()

        return model
    def create_optimizer(self):
        return keras.optimizers.Adam()
    def create_loss_function(self):
        return keras.losses.MeanSquaredError()


    def train(self,data,label,batch_size = 32, epochs = 10):
        print(data.shape)
        # print(data[])
        data = data.reshape(data.shape[0],data.shape[1],data.shape[2],1)
        self.model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['accuracy'])

        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        # Create a callback that saves the model's weights
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        history = self.model.fit(data,
                  label,
                  batch_size=batch_size,
                  epochs=epochs,
                    callbacks=[cp_callback])
        # print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        # plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    def predictions (self,predict_data):
        predict_data = predict_data.reshape(predict_data.shape[0],predict_data.shape[1],predict_data.shape[2],1)
        self.model.load_weights(self.checkpoint_path)
        return self.model.predict(predict_data)

    def iou(self,params0, params1):
        row0, col0, rad0 = params0
        row1, col1, rad1 = params1

        shape0 = Point(row0, col0).buffer(rad0)
        shape1 = Point(row1, col1).buffer(rad1)

        return (
                shape0.intersection(shape1).area /
                shape0.union(shape1).area
        )

    def evaluate(self,eval_data,eval_label):
        predict_label = self.predictions(eval_data)
        # print(predict_label)
        results = []
        for i in range(eval_data.shape[0]):
            results.append(self.iou(eval_label[i],predict_label[i]))

        results = np.array(results)
        return (results > 0.7).mean()


