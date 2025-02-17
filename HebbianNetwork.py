
import numpy as np

RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"


class HebbianNetwork:
    """
    This class implements the Hebbian Network
    """
    def __init__(self, input_size, output_size, learning_rate=0.001):
        """
        in this constructor we made the weight mat, and getting the size of the net, number of inputs and outputs
        and the learning rate of the net
        :param input_size: the size of one input layer, the number of neurons in the input layer
        :param output_size: the size of one output layer, the number of neurons in the output layer
        :param learning_rate: the learning rate
        """
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = np.zeros((output_size, input_size)) #set the weight mat to 0

    @staticmethod
    def sigmoid(x):
        """
        the sigmoid function is for normalization of the output to a number between 0 and 1
        :param x: the output we want to normalize
        :return: the normalized output
        """
        return 1 / (1 + np.exp(-x))

    def train(self, inputs_train, targets_train, epochs=100, regularization=0.00001):
        """
        In this function we will train our network
        :param inputs_train: all the inputs we want to train
        :param targets_train: the target that fit to each input
        :param epochs: the max number of epochs we want to do
        :param regularization: the regularization parameter
        :return: the best error that we can achieve in the last epoch we made
        """
        inputs_train = np.array([np.array(x, dtype=float) for x in inputs_train])
        targets_train = np.array([np.array(x, dtype=float) for x in targets_train])
        best_weights = None
        best_error = float('inf')
        for epoch in range(epochs):  # round for all the letters
            total_error = 0
            for inputs, target in zip(inputs_train, targets_train):  # loop for specific input and target

                # making the vectors to be with the rows and the columns swap
                inputs = inputs.reshape(-1, 1)
                target = target.reshape(-1, 1)
                curr_output = np.dot(self.weights, inputs)

                # for normalize the outputs
                curr_output = self.sigmoid(curr_output)

                # now we calculate the Error by MSE -Mean Squared Root, and then we will add it to the total error
                error = np.mean((curr_output - target) ** 2)
                total_error += error

                # Now we will update the weight by the hebbian formula Delta_w= learning rate X x X (y_t-y)
                delta_w = self.learning_rate * np.dot((target - curr_output), inputs.T)
                # Now we want to update the weights by giving more wight for the delta each time
                self.weights += delta_w - regularization * self.weights

            # End for the specific input vector, now we will calculate all the important values
            avg_error = total_error / len(inputs_train)
            if avg_error < best_error:
                best_error = avg_error
                best_weights = self.weights.copy()

            # Print each 10 epochs the error the best error mistake
            if (epoch + 1) % 10 == 0:
                print(f"{GREEN}\tFor epoch number: {epoch + 1} the best error mistake is {avg_error:.7f} {RESET}")

            #if the avarage error is under this level it pretty accurate and we can break the loop
            if avg_error < 0.000001:
                print(f"{GREEN}We succeeded for converge on epoch {epoch + 1}{RESET} ")
                break

        #saving the calculated weights
        self.weights = best_weights
        print(f"{GREEN}finished training updating weights...{RESET}")
        return best_error

    def predict(self, inputs):
        """
        Here we will calculate to predict, using sigmoid to normalize it and creating the last vector by out 1 were
        it is the max value and 0 else
        :param inputs: the input vector
        :return:
        """
        inputs = np.array(inputs, dtype=float).reshape(-1, 1)
        outputs = self.sigmoid(np.dot(self.weights, inputs))
        output = np.zeros(self.output_size)
        i = np.argmax(outputs)
        output[i] = 1
        return output

    def calculate_accuracy(self, inputs_test, targets_test):
        """
        For testing here we will calculate the accuracy of the network for other vectors
        :param inputs_test: the input vector
        :param targets_test: the target vector
        :return: the accuracy of the network
        """
        succeeded = 0
        input_len = len(inputs_test)
        for inputs, target in zip(inputs_test, targets_test):
            prediction = self.predict(inputs)
            if np.array_equal(prediction, target):
                succeeded += 1

        return succeeded / input_len
