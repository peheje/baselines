import tensorflow as tf
import numpy as np

# Hyperparameters
learning_rate = 0.0001
training_epochs = 100000
display_step = 1000
batch_size = 128
beta = 0.01

count_up_to = 100

# Network parameters
n_hidden_1 = 128
n_hidden_2 = 128
n_hidden_3 = 128
n_input = 1                  # 0-9
n_output = count_up_to       # 0-1 but as 1-hot vector. 2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# Placeholders
x = tf.placeholder("float", [None, n_input], name="Input")
y = tf.placeholder("float", [None, n_output], name="Output")


def multilayer_perceptron(nn_input, weights, biases):
    layer_1 = tf.add(tf.matmul(nn_input, weights["h1"]), biases["b1"])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
    layer_2 = tf.nn.sigmoid(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2, weights["h3"]), biases["b3"])
    layer_3 = tf.nn.sigmoid(layer_3)
    out_layer = tf.add(tf.matmul(layer_3, weights["out"]), biases["out"])
    return out_layer


def create_batch(batch_size):
    # We want to learn it how to coint: 0->1, 1->2, 2->3 .. 9->0
    xs = np.random.randint(0, count_up_to, batch_size)
    ys = [(t + 1) % count_up_to for t in xs]

    # But we need to create batches of observation input variables and response.
    # Format it into something tensorflow understands (shit).
    # The shape of batch_x has to be (batch_size, 1) as there is 64 observations of 1 value
    # The shape of batch_y has to be (batch_size, 10) as there is 64 observations of 1-hot vectors corresponding
    # to correct answer e.g. 2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    batch_x = np.array([[t] for t in xs])

    # https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    targets_y = np.array([ys]).reshape(-1)
    batch_y = np.eye(n_output)[targets_y]

    return batch_x, batch_y

# Weights and bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Construct model
nn_out = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
regularizer = tf.nn.l2_loss(weights["h1"]) + tf.nn.l2_loss(weights["h2"])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nn_out, labels=y) + regularizer * beta)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Train
    for epoch in range(training_epochs):

        train_x, train_y = create_batch(batch_size)

        _, c = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})

        if epoch % display_step == 0:
            print("Epoch {}".format(epoch))

    print("Training finished")

    # Test model
    pairs = []
    for test_value in range(count_up_to):
        test_x = np.zeros(1)
        test_x[0] = test_value
        test_x = test_x.reshape(1, 1)

        nn_out_softmax = tf.nn.softmax(nn_out)
        classification = sess.run(nn_out_softmax, {x: test_x})
        # print("sum: {}".format(np.sum(classification, 1)))
        # print("classifications: {}".format(classification))
        predicted_number = np.argmax(classification)
        print("predicted next number from {} is {}".format(test_value, predicted_number))

        pairs.append((test_value, predicted_number))

    correct = len(list(filter(lambda t: t[0] != (t[1]+1) % count_up_to, pairs)))
    print("Correct: {} out of: {}, percentage: {}".format(correct, count_up_to, (correct/count_up_to)*100))
