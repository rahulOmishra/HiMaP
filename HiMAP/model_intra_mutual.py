import tensorflow as tf
import numpy as np

class Model(object):

    def __init__(self, max_sen_len, class_num, embedding_dim, hidden_size):

        self.max_sen_len = max_sen_len
        self.embedding_dim = embedding_dim
        self.class_num = class_num
        self.hidden_size = hidden_size

        with tf.name_scope('input'):
            # two copies of the same propagation path 'U' as u1 and u2. 
            self.u1 = tf.placeholder(tf.float32, [None, self.max_sen_len, self.embedding_dim], name="u1")
            self.u2 = tf.placeholder(tf.float32, [None, self.max_sen_len, self.embedding_dim], name="u2")
            self.y = tf.placeholder(tf.float32, [None, self.class_num], name="y")
        with tf.name_scope('weights'):
            self.weights = {
                'q_1_to_2': tf.Variable(tf.random_uniform([2 * embedding_dim, self.hidden_size], -0.01, 0.01)),

                'p_1_to_2': tf.Variable(tf.random_uniform([self.hidden_size, 1], -0.01, 0.01)),

                'z': tf.Variable(tf.random_uniform([2*self.embedding_dim+self.hidden_size, self.hidden_size], -0.01, 0.01)),

                'f': tf.Variable(tf.random_uniform([self.hidden_size, self.class_num], -0.01, 0.01)),
            }

        with tf.name_scope('biases'):
            self.biases = {
                'q_1_to_2': tf.Variable(tf.random_uniform([self.hidden_size], -0.01, 0.01)),

                'p_1_to_2': tf.Variable(tf.random_uniform([1], -0.01, 0.01)),

                'z': tf.Variable(tf.random_uniform([self.hidden_size], -0.01, 0.01)),

                'f': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
            }

    def intra_mutual_attention(self):
        
        u1_shape = tf.shape(self.u1)
        u2_shape = tf.shape(self.u2)

        u1_reshape = tf.reshape(self.u1, [-1, self.embedding_dim, 1])
        ones = tf.ones([u1_shape[0]*self.max_sen_len, 1, self.max_sen_len])
        u1_increase = tf.matmul(u1_reshape, ones)
        u1_increase = tf.transpose(u1_increase, perm=[0, 2, 1])
        u1_increase = tf.reshape(u1_increase, [-1, self.max_sen_len*self.max_sen_len, self.embedding_dim])

        u2_reshape = tf.reshape(self.u2, [-1, self.embedding_dim, 1])
        ones = tf.ones([u2_shape[0]*self.max_sen_len, 1, self.max_sen_len])
        u2_increase = tf.matmul(u2_reshape, ones)
        u2_increase = tf.transpose(u2_increase, perm=[0, 2, 1])
        u2_increase = tf.reshape(u2_increase, [-1, self.max_sen_len, self.max_sen_len, self.embedding_dim])
        u2_increase = tf.transpose(u2_increase, perm=[0, 2, 1, 3])
        u2_increase = tf.reshape(u2_increase, [-1, self.max_sen_len*self.max_sen_len, self.embedding_dim])

        concat = tf.concat([u1_increase, u2_increase], axis=-1)

        '''
        This how intra mutual attention works: let's consider a case where the propagation path has only two Users and each learned user embedding has 3 dimensions
        therefore embedding matrix will be somewhat like:
        [[----1----]
        [----2----]].  (2*3)

        u1_increase =
        ----1----,----1----
        ----2----,----2----


        u2_increase =
        ----1----,----2----
        ----1----,----2----

        if we concatenate the both u1_increase and u2_increase, we get all possible combinations of user embedding pairs in a propagation path

        ----1--------1----,----1--------2----
        ----2--------1----,----2--------2----

        '''


        concat = tf.reshape(concat, [-1, 2*self.embedding_dim])

        s_1_to_2 = tf.nn.relu(tf.matmul(concat, self.weights['q_1_to_2']) + self.biases['q_1_to_2'])
        s_1_to_2 = tf.matmul(s_1_to_2, self.weights['p_1_to_2']) + self.biases['p_1_to_2']
        s_1_to_2 = tf.reshape(s_1_to_2, [-1, self.max_sen_len, self.max_sen_len])

        a_1 = tf.reshape(tf.nn.softmax(tf.reduce_max(s_1_to_2, axis=-1), axis=-1), [-1, 1, self.max_sen_len])

        self.v_a_1_to_2 = tf.reshape(tf.matmul(a_1, self.u1), [-1, self.embedding_dim])

        a_2 = tf.reshape(tf.nn.softmax(tf.reduce_max(tf.transpose(s_1_to_2, perm=[0, 2, 1]), axis=-1), axis=-1), [-1, 1, self.max_sen_len])

        self.v_a_2_to_1 = tf.reshape(tf.matmul(a_2, self.u2), [-1, self.embedding_dim])

        self.v_a = tf.concat([self.v_a_1_to_2, self.v_a_2_to_1], axis=-1)
        

    def long_short_memory_encoder(self):

        lstm_cell = tf.keras.layers.LSTMCell(self.hidden_size)
        LSTM_layer = tf.keras.layers.RNN(lstm_cell)
        self.v_c = LSTM_layer(tf.concat([self.u1, self.u2], axis=1))

    def prediction(self):

        v = tf.concat([self.v_a, self.v_c], -1)
        v = tf.nn.relu(tf.matmul(v, self.weights['z']) + self.biases['z'])

        self.scores = tf.nn.softmax((tf.matmul(v, self.weights['f']) + self.biases['f']), axis=-1)

        self.predictions = tf.argmax(self.scores, -1, name="predictions")

    def build_model(self):

        self.intra_mutual_attention()
        self.long_short_memory_encoder()
        self.prediction()
        
        with tf.name_scope("loss"):

            losses = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.argmax(self.y, -1),
                logits=self.scores)

            self.loss = tf.reduce_mean(losses)
            
        with tf.name_scope("metrics"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, -1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.c_matrix = tf.confusion_matrix(labels = tf.argmax(self.y, -1), predictions = self.predictions, name="c_matrix")
