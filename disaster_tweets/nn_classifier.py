import sys
import json
import argparse
import numpy as np
from collections import defaultdict

import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from data_util.data_reader import TweetLoader, DataSetIterator
from model.simple_nn import SimpleNetWork

tf.logging.set_verbosity(tf.logging.DEBUG)

def main(config, mode, restore_path):
    config = json.load(config)
    epoch = config["epoch"]
    batch_size = config["batch_size"]


    loader = TweetLoader(config["class_num"])
    if mode == "predict":
        predict_file = config["predict_data"]
        tf.logging.info("working in %s mode, loading data from file: %s ..." % (mode, predict_file))

        dataset = loader.load_data(predict_file, batch=batch_size, predict=True)
        tf.logging.info("load data complete")

        predict(config, dataset, restore_path)
    else:
        train_file = config["train_data"]
        tf.logging.info("working in %s mode, loading data from file: %s ..." % (mode, train_file))

        train_dataset, test_dataset = loader.load_data(train_file, batch=batch_size, repeat=epoch)
        tf.logging.info("load data complete")

        train(config, train_dataset, test_dataset, restore_path)


def train(config, train_dataset, test_dataset, restore_path=None):
    tf.logging.info("start train procedure")
    # choose model
    model_name = config["model_name"]
    if model_name == "CNN":
        # TODO
        pass
    else:
        net = SimpleNetWork()

    net.init(config)
    f, l = train_dataset.make_one_shot_iterator().get_next()
    saver = tf.train.Saver(max_to_keep=10)
    print_log_every = config["print_log_every"]
    save_model_every = config["save_model_every"]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        i = 1
        while True:
            try:
                feature, label = sess.run([f, l])
                _, loss = sess.run([net.train, net.loss], feed_dict={net.input: feature, net.label: label})

                if i % config["print_log_every"] == 0:
                    tf.logging.info("iter %d, current average loss: %.8f" % (i, loss / len(label)))

                if i % config["eval_model_every"] == 0:
                    predict = sess.run(net.predict, feed_dict={net.input: feature})
                    correct_num = np.sum(np.equal(np.argmax(label, 1), np.argmax(predict, 1)).astype(int))
                    tf.logging.info("iter %d, test %d train examples, %d correct, train accuracy: %.8f" % 
                            (i, len(label), correct_num, float(correct_num) / len(label)))
                    eval_model(net, sess, test_dataset)

                if i % config["save_model_every"] == 0:
                    tf.logging.info("iter %d, saving model ..." % i)
                    save_name = "%s-%d" % (config["model_save_path"], i)
                    saver.save(sess, save_name)
                    tf.logging.info("iter %d, model saved" % i)
                i += 1
            except tf.errors.OutOfRangeError:
                eval_model(net, sess, test_dataset)
                save_name = "%s-final" % (config["model_save_path"])
                saver.save(sess, save_name)
                tf.logging.info("train complete!")
                break

def eval_model(net, sess, test_dataset):
    stat_dict = defaultdict(int)
    f, l = test_dataset.make_one_shot_iterator().get_next()
    while True:
        try:
            feature, label = sess.run([f, l])
            predict = sess.run(net.predict, feed_dict={net.input: feature})

            correct_num = np.sum(np.equal(np.argmax(label, 1), np.argmax(predict, 1)).astype(int))

            stat_dict["total"] += len(predict)
            stat_dict["correct"] += correct_num
        except tf.errors.OutOfRangeError:
            break

    total = stat_dict["total"]
    correct = stat_dict["correct"]
    accuracy = float(correct) / total
    tf.logging.info("predict %d sample, correct %d, accuracy: %.8f" % (total, correct, accuracy))

def predict(config, dataset, restore_path):
    tf.logging.info("start predict procedure")
    if restore_path == "":
        tf.logging.error("restore path is empty when predict!")
        return

    graph_path = "%s.meta" % restore_path

    print "id,target"
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(graph_path)
        saver.restore(sess, restore_path)
        tf.logging.info("restore model from %s" % restore_path)

        graph = tf.get_default_graph()
        model_input = graph.get_tensor_by_name("input:0")
        model_output = graph.get_tensor_by_name("output:0")

        f, i = dataset.make_one_shot_iterator().get_next()
        while True:
            try:
                feature, idx = sess.run([f, i])
                output = sess.run(model_output, feed_dict={model_input: feature})
                label = np.argmax(output, 1)
                for idx, label in zip(idx, label):
                    print ",".join(map(str, [idx, label]))
            except tf.errors.OutOfRangeError:
                tf.logging.info("predict complete!")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=file, required=True, help="config json file")
    parser.add_argument("--mode", "-m", required=True, choices=['train', 'predict'], help="work mode for model")
    parser.add_argument("--restore_path", "-r", default="", help="path of model need to restore")
    args = parser.parse_args()

    main(args.config, args.mode, args.restore_path)
