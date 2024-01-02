import os
import json
import tensorflow as tf

from module import activation_function as afwrapper

def directory_checker(parent_directory='', save_directory_name=''):
    if save_directory_name == '':
        if not os.path.exists(parent_directory):
            os.makedirs(parent_directory)
    elif save_directory_name != '':
        save_dir = os.path.join(parent_directory, save_directory_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

def save_model(save_name:str, saving_folder:str, model, training_conf):
    directory_checker(saving_folder)
    savefile_path = os.path.join(saving_folder, save_name)

    model.save(savefile_path + ".keras")
    with open(savefile_path + ".json", "w") as f:
        json.dump(training_conf, f, indent=4)


def load_model(model_path:str, configuration_path:str):
    with open(configuration_path, 'r') as f:
        configuration_json = json.read(f)

    serialization_key = configuration_json['serialization_key']
    selected_activation_func = None
    if 'relu' in serialization_key:
        selected_activation_func = afwrapper.relu
    if 'tanh' in serialization_key:
        selected_activation_func = afwrapper.tanh
    if 'sigmoid' in activation_key:
        selected_activation_func = afwrapper.sigmoid


    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"serialization_key"}
    )

    return model
