import numpy as np
from HebbianNetwork import HebbianNetwork
from VectorsFactory import VectorsFactory

RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
RESET = "\033[0m"

def standard_deviation(var):
    """
    This function returns the standard deviation of the results
    :param var:
    :return:
    """
    return np.sqrt(var)

def variance(net,outputs,avg,percent_of_mistake,number_of_rounds):
    """
    This function returns the variance of the results
    :param net: the Hebbian Network object
    :param outputs: the output vectors
    :param avg: the average results of the specific
    :return:
    """
    var=0
    for i in range(number_of_rounds):
        inputs = VectorsFactory.create_letters_vector()
        vecs = VectorsFactory.create_vectors_with_mistakes(inputs, percent=percent_of_mistake)
        accuracy=net.calculate_accuracy(vecs,outputs)
        var += ((accuracy-avg)**2) / number_of_rounds
    return var

def cal_currency_var_std(net,outputs,number_of_rounds=500):
    """
    this function prints all the results for the mistake groups
    :param net: the Hebbian Network object
    :param outputs: the output vectors
    :param number_of_rounds: the number of rounds
    """
    print(f"{GREEN}calculating of accuracy in avarage on {number_of_rounds} times of running {RESET}")
    for i in range(5, 21, 5):
        total_accuracy = 0
        for j in range(number_of_rounds):
            inputs = VectorsFactory.create_letters_vector()
            vecs = VectorsFactory.create_vectors_with_mistakes(inputs, percent=(i / 100))
            total_accuracy += net.calculate_accuracy(vecs, outputs) / number_of_rounds


#printing the results for the 3 group options
def result_for_group(description_of_group,net,inputs,outputs):
    """
    This function returns the results of the group
    :param description_of_group: the name of the group
    :param net: the Hebbian Network object
    :param inputs: the input vectors
    :param outputs: the output vectors
    """
    accuracy = net.calculate_accuracy(inputs, outputs)
    print(f"\n{PURPLE}the accuracy with {description_of_group}: {accuracy:.2%}{RESET}")
    print(f"{GREEN}Results for this group: {RESET}")
    for j, input_vector in enumerate(inputs):
        prediction = net.predict(input_vector)
        if np.array_equal(prediction, [1, 0, 0]):
            ans = "A-I"
        elif np.array_equal(prediction, [0, 1, 0]):
            ans = "J-R"
        else:
            ans = "S-Z"

        expected = "A-I" if j <= 8 else "J-R" if j <= 17 else "S-Z"
        color = BLUE if ans == expected else RED
        print(f"\t{color}The letter {chr(65 + j)} Suppose to be in: {expected},and the answer is: {ans}{RESET}")
    print()

#printing the results for the Letters prediction
def result2_for_group(description_of_group,net,inputs,outputs):
    accuracy = net.calculate_accuracy(inputs, outputs)
    print(f"\n{PURPLE}the accuracy with {description_of_group}: {accuracy:.2%}{RESET}")
    print(f"{GREEN}Results for this group: {RESET}")
    for j, input_vector in enumerate(inputs):
        prediction = net.predict(input_vector)
        for i in range(0,len(prediction)):
            if prediction[i] == 1:
                color = BLUE if i == j else RED
                print(f"\t{color}The letter {chr(65 + j)} got answer: {chr(65 + i)}{RESET}")



def project_option_1_groups():
    # crating the network
    net = HebbianNetwork(64, 3, learning_rate=1)

    # creating the inputs
    inputs = VectorsFactory.create_letters_vector()
    outputs = VectorsFactory.create_groups_mat()

    # training the network
    print(f"{GREEN}start training...{RESET}")
    errors = net.train(inputs, outputs, epochs=5000)
    print(f"{GREEN}training finished!{RESET}")

    # creating all the vectors we want to check their accuracy on this net
    train_inputs = VectorsFactory.create_letters_vector()
    vecs5per = VectorsFactory.create_vectors_with_mistakes(VectorsFactory.create_letters_vector(), percent=0.05)
    vecs10per = VectorsFactory.create_vectors_with_mistakes(VectorsFactory.create_letters_vector(), percent=0.1)
    vecs15per = VectorsFactory.create_vectors_with_mistakes(VectorsFactory.create_letters_vector(), percent=0.15)
    vecs20per = VectorsFactory.create_vectors_with_mistakes(VectorsFactory.create_letters_vector(), percent=0.2)
    test2vecs = VectorsFactory.create_bold_letters_vector()
    test3vecs = VectorsFactory.create_letters_v3_vector()
    test4vecs = VectorsFactory.create_letters_v4_vector()

    #printing and calculating the results of each group with their decription
    result_for_group("Group of the train letters ",net,train_inputs,outputs)
    result_for_group("Group with 5% of mistakes ",net,vecs5per,outputs)
    result_for_group("Group with 10% of mistakes ",net,vecs10per,outputs)
    result_for_group("Group with 15% of mistakes ",net,vecs15per,outputs)
    result_for_group("Group with 20% of mistakes ",net,vecs20per,outputs)
    result_for_group("Group 1:Bold ", net, test2vecs, outputs)
    result_for_group("Group 2:Bold and circle ", net, test3vecs, outputs)
    result_for_group("Group 3:extra Bold and circle ",net,test4vecs,outputs)
    #calculating the accuracy, std, and variance
    cal_currency_var_std(net,outputs,number_of_rounds=500)


def project_option_2_groups():
    # crating the network
    net = HebbianNetwork(64, 26, learning_rate=1)

    # creating the inputs
    inputs = VectorsFactory.create_letters_vector()
    outputs = VectorsFactory.create_output_mat()

    # training the network
    print(f"{GREEN}start training...{RESET}")
    errors = net.train(inputs, outputs, epochs=5000)
    print(f"{GREEN}training finished!{RESET}")

    # crating all the vectors we want to check their accuracy on this net
    train_inputs = VectorsFactory.create_letters_vector()
    vecs5per = VectorsFactory.create_vectors_with_mistakes(VectorsFactory.create_letters_vector(), percent=0.05)
    vecs10per = VectorsFactory.create_vectors_with_mistakes(VectorsFactory.create_letters_vector(), percent=0.1)
    vecs15per = VectorsFactory.create_vectors_with_mistakes(VectorsFactory.create_letters_vector(), percent=0.15)
    vecs20per = VectorsFactory.create_vectors_with_mistakes(VectorsFactory.create_letters_vector(), percent=0.2)
    test2vecs = VectorsFactory.create_bold_letters_vector()
    test3vecs = VectorsFactory.create_letters_v3_vector()
    test4vecs = VectorsFactory.create_letters_v4_vector()

    # printing and calculating the results of each group with their decription for output 2 option
    result2_for_group("Group of the train letters ", net, train_inputs, outputs)
    result2_for_group("Group with 5% of mistakes ", net, vecs5per, outputs)
    result2_for_group("Group with 10% of mistakes ", net, vecs10per, outputs)
    result2_for_group("Group with 15% of mistakes ", net, vecs15per, outputs)
    result2_for_group("Group with 20% of mistakes ", net, vecs20per, outputs)
    result2_for_group("Group 1:Bold ", net, test2vecs, outputs)
    result2_for_group("Group 2:Bold and circle ", net, test3vecs, outputs)
    result2_for_group("Group 3:extra Bold and circle ", net, test4vecs, outputs)

    # calculating the accuracy, std, and variance
    cal_currency_var_std(net, outputs, number_of_rounds=500)

if __name__ == '__main__':
    a=""
    while a!="3":
        print("Enter a number for active the Hebbian Network: \n\t1.For categories for 3 groups (A-I),(J-R),(S-Z)"
              "\n\t2.For categories each letter for itself \n\t3.Exit")
        a = input("\tenter a number:")

        if a=="1":
            project_option_1_groups()
        elif a=="2":
            project_option_2_groups()
        elif a=="3":
            print("Exit...")




