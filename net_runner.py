import os
import sys

# Make a hyper parameter dict
param_dict = {
        "name": "test",
        "learning_rate": 0.0001, #0.001
        "beta1": 0.5, #0.9
        "beta2": 0.9, #0.999
        "epsilon": 1e-08, #1e-08
        "training_epochs": 1,
        "display_step": 50,
        "batch_size": 5000,
        "train_test_ratio": 0.85,
        "activation_function": "relu",
        "layers": [200],
        "output_layer_biases": False,
        "n_input": 60,
        "n_classes": 2
}

param_dict_keylist = [
        "name",
        "learning_rate",
        "beta1",
        "beta2",
        "epsilon",
        "training_epochs",
        "display_step",
        "batch_size",
        "train_test_ratio",
        "activation_function",
        "layers",
        "output_layer_biases",
        "n_input",
        "n_classes"
]

def param_dict_to_str(dict, keylist):
    # Convert dict into list
    param_list = []
    for x in keylist:
        param_list.append(param_dict[x])

    param_str = str(param_list).strip('[]')
    param_str = param_str.replace(', ', ' ')
    param_str = param_str.replace('\'', '')

    return param_str

def param_str_decoder(string):
    pass

argv = param_dict_to_str(param_dict, param_dict_keylist)
os.system("net.py " + argv)
