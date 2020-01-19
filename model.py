import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras as keras
from shapely.geometry.point import Point
import math as m
import matplotlib.pyplot as plt
import os
class detectModel:
    def __init__(self, image_size,save_path = None,output_size = 3, model = None, optimizer = None,loss_function = None):
        """
        initialie the detectModel classes.
        :param image_size: size of image
        :param save_path:  path to save weight
        :param output_size: size of output for prediction
        :param model: defaul = None. Train new model if there is no given model
        :param optimizer: default = None. Use default Adam optimizer
        :param loss_function: defaul = None. Use mean square error function
        """


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

    def create_model(self):
        model = keras.Sequential()
        # Must define the input shape in the first layer of the neural network
        model.add(keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(self.size, self.size, 1)))
        model.add(keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(pool_size=2,strides=2))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(pool_size=2,strides=2))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1024, activation='relu'))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(self.output_size))

        # Take a look at the model summary
        model.summary()

        return model

    def create_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=0.001)
    def create_loss_function(self):
        return keras.losses.MeanSquaredError()

    def save_model(self,path = None):
        if path is None:
            if not os.path.isdir("save_model"):
                os.makedirs("save_model")
            path = "save_model/model.h5"
        self.model.save(path)
    def load_model(self,path):
        return keras.models.load_model(path)

    def train(self,data, label, test_data, test_label,batch_size = 32, epochs = 10):
        """

        :param data:
        :param label:
        :param test_data:
        :param test_label:
        :param batch_size:
        :param epochs:
        :return:
        """
        data = data.reshape(data.shape[0],data.shape[1],data.shape[2],1)
        test_data = test_data.reshape(test_data.shape[0],test_data.shape[1],test_data.shape[2],1)
        loss = self.create_loss_function()
        self.model.compile(loss=loss,
                      optimizer=self.optimizer,
                      metrics=[self.mean_ious])

        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)


        cp_callback = keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      monitor='val_mean_ious',
                                                      mode='max',
                                                      verbose=1)

        history = self.model.fit(data,
                  label,
                  batch_size=batch_size,
                  epochs=epochs,
                                 validation_data=(test_data,test_label),
                    callbacks=[cp_callback])
        hist_df = pd.DataFrame(history.history)
        tfile = open('output.txt', 'a')
        tfile.write(hist_df.to_string())
        tfile.close()




        self.create_observe_graph(history)







    def predictions (self,predict_data):
        """
        make a prediction on the given image
        :param predict_data: (m,n,n) where m is # of sample and n is image size
        :return: (m,3) return predict row,column and radius
        """
        predict_data = predict_data.reshape(predict_data.shape[0],predict_data.shape[1],predict_data.shape[2],1)
        self.model.load_weights(self.checkpoint_path)
        return self.model.predict(predict_data)

    def evaluate(self,eval_data,eval_label):
        """
        evaluate the performance of the model
        :param eval_data: (m,n,n) where m is # of sample and n is image size
        :param eval_label: (m,3)
        :return:
        """
        predict_label = self.predictions(eval_data)
        results = []
        for i in range(eval_data.shape[0]):
            results.append(self.iou(eval_label[i],predict_label[i]))
        results = np.array(results)
        return (results > 0.7).mean()

    def iou(self,params0, params1):
        row0, col0, rad0 = params0
        row1, col1, rad1 = params1

        shape0 = Point(row0, col0).buffer(rad0)
        shape1 = Point(row1, col1).buffer(rad1)

        return (
                shape0.intersection(shape1).area /
                shape0.union(shape1).area
        )

    def iou_2(self,params0, params1):
        """
        A tensorflow implementation version of  iou between 2 circle

        :param params0: (m,3)
        :param params1: (m,3
        :return:
        """
        r0 = params0[...,2]
        x0 = params0[...,0]
        y0 = params0[...,1]
        r1 = params1[...,2]
        x1 = params1[...,0]
        y1 = params1[...,1]

        R = tf.math.maximum(r0, r1)
        r = tf.math.minimum(r0, r1)
        d = tf.sqrt(tf.add(tf.square(x0-x1),tf.square(y0-y1)))


        area1 = r0 * r0 * tf.constant(m.pi)
        area2 = r1 * r1 * tf.constant(m.pi)

        part1 = r * r * tf.math.acos((d * d + r * r - R * R) / (2 * d * r))
        part2 = R * R * tf.math.acos((d * d + R * R - r * r) / (2 * d * R))
        part3 = 0.5 * tf.sqrt((-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R))
        intersectionArea = part1 + part2 - part3

        union = area1 + area2 - intersectionArea
        ious = tf.math.divide(intersectionArea, union)
        return tf.where(tf.is_nan(ious),tf.zeros_like(ious),ious)


    def mean_ious(self,params0,params1):
        """
        This function will calculate the mean of IOUS such that
        IOU between 2 circle must be more than 70%
        :param params0: (m,3)
        :param params1: (m,3)
        :return:
        """
        result = self.iou_2(params0, params1)
        compare = tf.less(tf.constant(0.7), result)
        return tf.reduce_mean(tf.cast(compare, tf.float32))


    def create_observe_graph(self,history):
        plt.plot(history.history['mean_ious'])
        plt.title('model mean_ious')
        plt.ylabel('mean_ious')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('mean_ious.png')
        plt.show()

        plt.plot(history.history['val_mean_ious'])
        # plt.plot(history.history['val_accuracy'])
        plt.title('model val_mean_ious')
        plt.ylabel('val_mean_ious')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('val_mean_ious.png')
        plt.show()

        # summarize history for loss
        plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('loss_after_train.png')
        plt.show()

        numpy_loss_history = np.array(history.history['loss'])
        np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")

        numpy_mean_ious_history = np.array(history.history['mean_ious'])
        np.savetxt("mean_ious_history.txt", numpy_mean_ious_history, delimiter=",")

        numpy_mean_ious_history = np.array(history.history['val_mean_ious'])
        np.savetxt("val_mean_ious_history.txt", numpy_mean_ious_history, delimiter=",")