import os
import sys
import time

# Make a hyper parameter dict
param_dict = {
        "name": "test",
        "learning_rate": 0.001, #0.001
        "beta1": 0.9, #0.9
        "beta2": 0.999, #0.999
        "epsilon": 1e-08, #1e-08
        "training_epochs": 20,
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

def run_net(name, argv):
    for x in range(3):
        print("Starting {:s}, run {:d}.".format(name, x))
        os.system("net.py " + argv)
        time.sleep(1)

def main():

    # Run 1
    param_dict["name"] = "hyper_param_testing_relu_run1"
    run_net("Run 1", param_dict_to_str(param_dict, param_dict_keylist))

    # Run 2
    param_dict["name"] = "hyper_param_testing_relu_run2"
    param_dict["learning_rate"] = 0.0001
    run_net("Run 2", param_dict_to_str(param_dict, param_dict_keylist))

    # Run 3
    param_dict["name"] = "hyper_param_testing_relu_run3"
    param_dict["learning_rate"] = 0.01
    run_net("Run 3", param_dict_to_str(param_dict, param_dict_keylist))

    # Run 4
    param_dict["name"] = "hyper_param_testing_relu_run4"
    param_dict["learning_rate"] = 0.001
    param_dict["beta1"] = 0.6
    param_dict["beta2"] = 0.9
    run_net("Run 4", param_dict_to_str(param_dict, param_dict_keylist))

    # Run 5
    param_dict["name"] = "hyper_param_testing_relu_run5"
    param_dict["beta1"] = 0.3
    param_dict["beta2"] = 0.8
    run_net("Run 5", param_dict_to_str(param_dict, param_dict_keylist))

    # Run 6
    param_dict["name"] = "hyper_param_testing_relu_run6"
    param_dict["beta1"] = 0.9
    param_dict["beta2"] = 0.999
    param_dict["epsilon"] = 1e-01
    run_net("Run 6", param_dict_to_str(param_dict, param_dict_keylist))

    # Run 7
    param_dict["name"] = "hyper_param_testing_relu_run7"
    run_net("Run 7", param_dict_to_str(param_dict, param_dict_keylist))
    param_dict["epsilon"] = 1.0

    # Run 8
    param_dict["name"] = "hyper_param_testing_sigmoid_run8"
    param_dict["activation_function"] = "sigmoid"
    param_dict["epsilon"] = 1e-08
    run_net("Run 8", param_dict_to_str(param_dict, param_dict_keylist))

    # Run 9
    param_dict["name"] = "hyper_param_testing_sigmoid_run9"
    param_dict["learning_rate"] = 0.0001
    run_net("Run 9", param_dict_to_str(param_dict, param_dict_keylist))

    # Run 10
    param_dict["name"] = "hyper_param_testing_sigmoid_run10"
    param_dict["learning_rate"] = 0.01
    run_net("Run 10", param_dict_to_str(param_dict, param_dict_keylist))

    # Run 11
    param_dict["name"] = "hyper_param_testing_sigmoid_run11"
    param_dict["learning_rate"] = 0.01
    param_dict["beta1"] = 0.6
    param_dict["beta2"] = 0.9
    run_net("Run 11", param_dict_to_str(param_dict, param_dict_keylist))

    # Run 12
    param_dict["name"] = "hyper_param_testing_sigmoid_run12"
    param_dict["beta1"] = 0.3
    param_dict["beta2"] = 0.8
    run_net("Run 12", param_dict_to_str(param_dict, param_dict_keylist))

    # Run 13
    param_dict["name"] = "hyper_param_testing_sigmoid_run13"
    param_dict["beta1"] = 0.9
    param_dict["beta2"] = 0.999
    param_dict["epsilon"] = 1e-01
    run_net("Run 13", param_dict_to_str(param_dict, param_dict_keylist))

    # Run 14
    param_dict["name"] = "hyper_param_testing_sigmoid_run14"
    param_dict["epsilon"] = 1.0
    run_net("Run 14", param_dict_to_str(param_dict, param_dict_keylist))

if __name__ == "__main__":
    main()
