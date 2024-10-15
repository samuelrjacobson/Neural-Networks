import numpy as np

#initialize weights and biases
input_neurons = 36
hidden_neurons = 3
output_neurons = 2
learning_rate = 0.1
epochs = 10000

#both team's stats
#This data is the past 4 years of Ohio State - Michigan State games (chosen as first example because OSU has consistently beaten MSU by quite a lot)
training_data = np.array([[19.6, 2, 1.5, 44.5, 0.5, 0.5, 15.2, 0.7, 0.5, 40.7, 0.3, 0.5, 80, 50, 43.5, 5, 5, 9, 18.4, 1.2, 0.6, 60.7, 0.8, 1.1, 18.8, 2, 1.2, 47.9, 0.6, 0.8, 72.2, 55, 46.8, 10, 42, -4],
                          [21.1,3.2,2.3,48,0.3,0.5,16.3,1.5,0.9,45.4,0.5,0.8,85,49,45.4,15,4,9,21.5,1.9,1.3,54.9,0.3,1,19.9,2.2,1.2,63.3,0.8,0.2,50,50,49,10,24,-2],
                          [26.8,3.5,1.8,59.2,0.2,0.6,21.8,1.6,1.3,38.2,0.6,0.9,95.2,31,42.3,27,4,9,18.7,2.1,1.8,63.9,0.5,0.8,29.2,2.1,0.8,46.3,0.8,0.8,63.2,59,48.4,13,23,9],
                          [19.8,2.8,2.4,54.4,0.4,0.8,26.1,2,1.3,51.4,1.5,0.9,71.4,27,45,15,2,6,18.9,1.6,0.3,63,1.1,1.7,21.6,1.1,2.9,50.4,0.9,0.7,75,37,43.6,30,46,-3]])
actual_scores = np.array([[38, 7], [38,3], [49,20], [56,7]])

w_hidden = np.random.randn(input_neurons, hidden_neurons) * np.sqrt(1.0 / input_neurons) # * see bottom
b_hidden = np.random.randn(1, hidden_neurons) * np.sqrt(1.0 / input_neurons)

w_output = np.random.randn(hidden_neurons, output_neurons) * np.sqrt(1.0 / hidden_neurons)
b_output = np.random.randn(1, output_neurons) * np.sqrt(1.0 / hidden_neurons)

#activation functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    return(x*(1-x))

for epoch in range(epochs):
    #forward pass
    #input layer to hidden layer
    input_to_hidden_out = sigmoid(np.dot(training_data, w_hidden) + b_hidden)
    #hidden layer to output
    output = sigmoid(np.dot(input_to_hidden_out, w_output) + b_output)

    #calculate the error
    error = output - actual_scores

    #backpropagation
    #gradient of the error with respect to the output
    g_out = error * sigmoid_derivative(output)
    w_output -= learning_rate * np.dot(input_to_hidden_out.T, g_out)
    b_output -= learning_rate * np.sum(g_out, axis=0, keepdims=True)
    #gradient of error with respect to hidden neurons
    g_hidden = np.dot(g_out, w_output.T) * sigmoid_derivative(input_to_hidden_out)
    w_hidden -= learning_rate * np.dot(training_data.T, g_hidden)
    b_hidden -= learning_rate * np.sum(g_hidden, axis=0)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

#predictions
for i in range(training_data.shape[0]):
    hidden_layer = sigmoid(np.dot(training_data[i].reshape(1, -1), w_hidden) + b_hidden) # * see bottom
    output = np.dot(hidden_layer, w_output) + b_output # no activation function
    print(f"Input: {training_data[i]} - Pred.: {np.round(output)} - Actual: {actual_scores[i]}")

# * These two parts of the code were written with assistance from generative AI. I asked it to reshape the data, as my starting
# code was meant for smaller numbers of data. It definitely helped in getting my program to run and produce answers, but the predictions
# still were not very good.