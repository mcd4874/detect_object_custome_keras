from model import detectModel
from create_train_set import create_dataset as create_train_set
from create_train_set import save_data_into_h5,load_h5_data
import numpy as np

def create_data(n_train_sample,n_test_sample,n_valid_sample,img_size,max_radius,noise_level,constant_seed = True,save_data = True):
    test_data, test_label = create_train_set(n_test_sample, img_size, max_radius, noise_level)
    valid_data, valid_label = create_train_set(n_valid_sample, img_size, max_radius, noise_level)
    if constant_seed == True:
        np.random.seed(0)
    train_data, train_label = create_train_set(n_train_sample, img_size, max_radius, noise_level)
    return [train_data,train_label,test_data,test_label,valid_data,valid_label]

if __name__== "__main__":
    n_epochs = 300
    img_size = 200
    max_radius = 50
    noise_level = 2
    n_train_sample = 20000
    n_test_sample = 2000
    n_valid_sample = 1000

    create_new_data = True

    checkpoint_path = "training_1/cp.ckpt"

    train_model = False

    eval_model = True

    save_model = False

    if create_new_data == True:
        train_data,train_label,test_data,test_label,valid_data,valid_label = create_data(n_train_sample,n_test_sample,n_valid_sample,img_size,max_radius,noise_level)
        #save to dataset/data.hdf5 if no path is given
        save_data_into_h5(train_data,train_label,test_data,test_label,valid_data,valid_label)
    else:
        #if no path is given, load from dataset/data.hdf5
        train_data, train_label, test_data, test_label, valid_data, valid_label = load_h5_data()

    model = detectModel(img_size, save_path=checkpoint_path)

    # train
    if train_model == True:
        model.train(train_data, train_label, test_data, test_label, epochs=n_epochs)

    # evaluate
    if eval_model == True:
        print(model.evaluate(valid_data,valid_label))

    if save_model == True:
        #if path is none then save whole model to save_model/model.h5
        path = None
        model.save_model()


