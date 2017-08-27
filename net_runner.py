import os
import sys
import time

# Make a hyper parameter dict
param_dict = {
        "name": "test",
        "learning_rate": 0.00006, #0.001
        "beta1": 0.7, #0.9
        "beta2": 0.9, #0.999
        "epsilon": 1e-08, #1e-08
        "training_epochs": 5,
        "display_step": 50,
        "batch_size": 5000,
        "train_test_ratio": 0.85,
        "activation_function": "relu",
        "layers": "[400]",
        "output_layer_biases": True,
        "n_input": 20,
        "n_classes": 2,
        "window_length": 50
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
        "n_classes",
        "window_length"
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

def run_net(name, argv):
    for x in range(1):
        print("Starting {:s}, run {:d}.".format(name, x))
        os.system("python sequence_net.py " + argv)
        time.sleep(1)

def main():

    # Run 1 - Control
    param_dict["name"] = "testaroo"
    param_dict["learning_rate"] = 0.00006
    param_dict["epsilon"] = 1e-08
    run_net("Run 1", param_dict_to_str(param_dict, param_dict_keylist))

    # # Run 2
    # param_dict["name"] = "r5_sequence_2_"
    # param_dict["epsilon"] = 1e-00
    # run_net("Run 2", param_dict_to_str(param_dict, param_dict_keylist))
    #
    # # Run 3
    # param_dict["name"] = "r5_sequence_3_"
    # param_dict["beta1"] = 0.7
    # param_dict["beta2"] = 0.3
    # param_dict["epsilon"] = 1e-01
    # run_net("Run 3", param_dict_to_str(param_dict, param_dict_keylist))
    #
    # # Run 4
    # param_dict["name"] = "r5_sequence_4_"
    # param_dict["epsilon"] = 1e-00
    # run_net("Run 4", param_dict_to_str(param_dict, param_dict_keylist))
    #
    # # Run 5
    # param_dict["name"] = "r5_sequence_5_"
    # param_dict["beta1"] = 0.7
    # param_dict["beta2"] = 0.9
    # param_dict["epsilon"] = 1e-08
    # param_dict["layers"] = "[100]"
    # run_net("Run 5", param_dict_to_str(param_dict, param_dict_keylist))
    #
    # # Run 6
    # param_dict["name"] = "r5_sequence_6_"
    # param_dict["layers"] = "[10]"
    # run_net("Run 6", param_dict_to_str(param_dict, param_dict_keylist))
    #
    # # Run 7
    # param_dict["name"] = "r5_sequence_7_"
    # param_dict["layers"] = "[400-20]"
    # run_net("Run 7", param_dict_to_str(param_dict, param_dict_keylist))
    #
    # # Run 8
    # param_dict["name"] = "r5_sequence_8_"
    # param_dict["learning_rate"] = 0.0001
    # param_dict["layers"] = "[400-20]"
    # run_net("Run 8", param_dict_to_str(param_dict, param_dict_keylist))
    #
    # # Run 9
    # param_dict["name"] = "r5_sequence_9_"
    # param_dict["learning_rate"] = 0.00001
    # param_dict["layers"] = "[400-20]"
    # run_net("Run 9", param_dict_to_str(param_dict, param_dict_keylist))

if __name__ == "__main__":
    main()
