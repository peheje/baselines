import random
import sys

sys.path.append('../')
import gym
import numpy as np
import tensorflow as tf


def do_it():
    n_hidden = 32
    env = gym.make('CartPole-v0')

    # Initialize placeholders
    state_pl = tf.placeholder(tf.float32, shape=(None, env.observation_space.shape[0]), name="state_pl")
    reward_pl = tf.placeholder(tf.float32, shape=(None,), name="reward_pl")
    action_pl = tf.placeholder(tf.float32, shape=(None, env.action_space.n), name="action_pl")

    # Initialize convolutional net
    w1 = tf.get_variable("w1", shape=[env.observation_space.shape[0], n_hidden])
    b1 = tf.get_variable("b1", shape=[1, n_hidden], initializer=tf.constant_initializer(0.0))

    w2 = tf.get_variable("w2", shape=[n_hidden, n_hidden])
    b2 = tf.get_variable("b2", shape=[1, n_hidden], initializer=tf.constant_initializer(0.0))

    w3 = tf.get_variable("w3", shape=[n_hidden, env.action_space.n])
    b3 = tf.get_variable("b3", shape=[1, env.action_space.n], initializer=tf.constant_initializer(0.0))

    l1 = tf.tanh(tf.add(tf.matmul(state_pl, w1), b1))
    l2 = tf.tanh(tf.add(tf.matmul(l1, w2), b2))
    nn_output = tf.tanh(tf.add(tf.matmul(l2, w3), b3))

    # Specify loss function
    regularization = (tf.reduce_sum(tf.square(w1)) + tf.reduce_sum(tf.square(w2)) + tf.reduce_sum(tf.square(w3)))
    q_vals = tf.reduce_sum(tf.multiply(nn_output, action_pl), 1)
    loss = tf.reduce_sum(tf.square(reward_pl - q_vals)) + regularization

    learning_rate = 0.1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Train
    eps = 0.4
    gamma = 0.5
    train_steps = 1000
    s_a_r_sn = []
    for i in range(train_steps):
        eps *= 0.999
        s = env.reset()

        if i % 100 == 0:
            print(train_steps-i)
            print("epsilon {}".format(eps))

        # Play for n steps
        n = 200
        for p in range(n):
            if random.random() < eps:
                a = np.random.randint(0, env.action_space.n)
            else:
                _, prediction_probs = sess.run(
                    [loss, nn_output],
                    feed_dict={state_pl: np.array([s]),
                               reward_pl: np.zeros(len(s)),
                               action_pl: np.zeros((len(s), env.action_space.n))})
                a = np.argmax(prediction_probs)

            sn, r, done, info = env.step(a)
            s_a_r_sn.append((s, a, r, sn, done))
            s = np.copy(sn)
            if done:
                env.reset()

        random.shuffle(s_a_r_sn)

        ss = []
        rs = []
        acs = []

        for (s, a, r, sn, done) in s_a_r_sn[:100]:
            ar = np.zeros(env.action_space.n)
            ar[a] = 1
            ar = np.array([ar])
            if not done:
                sn = np.array(sn)
                _, prediction_probs = sess.run(
                    [loss, nn_output],
                    feed_dict={state_pl: np.array([sn]),
                               reward_pl: np.zeros(len(sn)),
                               action_pl: np.zeros((len(sn), env.action_space.n))})
                rn = np.max(prediction_probs)
                r = r + gamma*rn

            ss.append(s)
            rs.append(r)
            acs.append(ar)

        sess.run(
            [loss, optimizer, nn_output, q_vals],
            feed_dict={state_pl: np.array(ss),
                       reward_pl: np.array(rs),
                       action_pl: ar
                       })

    # Play the game
    state = env.reset()
    while True:
        s = np.array([state])
        _, prediction_probs = sess.run(
            [loss, nn_output],
            feed_dict={state_pl: s,
                       reward_pl: np.zeros(len(s)),
                       action_pl: np.zeros((len(s), env.action_space.n))})
        print(prediction_probs)
        a = np.argmax(prediction_probs[0])
        print("action index {}".format(a))
        state, reward, done, info = env.step(a)
        env.render()
        if done:
            state = env.reset()

do_it()
