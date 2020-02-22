import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

class SimpleNetWork(object):
    def __init__(self):
        self.layer_id = 0
        pass

    def init(self, model_conf):
        input_dim = model_conf["input_dim"]
        label_dim = model_conf["class_num"]
        self.learning_rate = model_conf["learning_rate"]
        self.layer_dim = model_conf["layer_dim"].split(",")
        self.l2_lambda = float(model_conf["l2_lambda"])
        assert len(self.layer_dim) > 0
        self.active_funcs = [tf.nn.tanh] * (len(self.layer_dim) - 1) + [tf.nn.softmax]

        self.input = tf.placeholder(tf.float32, [None, input_dim], name="input")
        self.label = tf.placeholder(tf.float32, [None, label_dim], name="label")

        last_input = self.input
        for i in xrange(len(self.layer_dim)):
            self.y = self._add_layer(last_input, self.layer_dim[i], self.active_funcs[i], l2_lambda=self.l2_lambda)
            last_input = self.y

        self.cross_entropy_loss = -tf.reduce_sum(self.label * tf.log(self.y + 1e-10)) 
        tf.add_to_collection('losses', self.cross_entropy_loss)

        self.loss = tf.add_n(tf.get_collection('losses'), name="loss")

        self.train = tf.train.AdamOptimizer(self.learning_rate, name="train_op").minimize(self.loss)
        self.predict = tf.identity(self.y, name="output")
        # self.correct_num = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.label, 1), tf.argmax(self.predict, 1)), tf.int32))

    def _add_layer(self, inputs, output_dim, active_func, l2_lambda=0.5, layer_name="layer"):
        input_dim = inputs.shape.as_list()[1]

        w_name = "%s_%d_w" % (layer_name, self.layer_id)
        b_name = "%s_%d_b" % (layer_name, self.layer_id)

        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.contrib.layers.xavier_initializer()

        w = tf.get_variable(w_name, [input_dim, output_dim], initializer = w_init, dtype=tf.float32)
        b = tf.get_variable(b_name, [1, output_dim], initializer = b_init, dtype=tf.float32)

        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(l2_lambda)(w))

        z = tf.add(tf.matmul(inputs, w), b)
        h = active_func(z)

        self.layer_id += 1

        return h

