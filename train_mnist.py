import tensorflow as tf
import numpy as np
import cPickle as pkl
from flip_gradient import flip_gradient
from utils import *

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Process MNIST
mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)

# Load MNIST-M
mnistm = pkl.load(open('mnistm_data.pkl'))
mnistm_train = mnistm['train']
mnistm_test = mnistm['test']
mnistm_valid = mnistm['valid']
# Compute pixel mean for normalizing data
pixel_mean = np.vstack([mnist_train, mnistm_train]).mean((0, 1, 2))

# Create a mixed dataset for TSNE visualization
num_test = 500
combined_test_imgs = np.vstack([mnist_test[:num_test], mnistm_test[:num_test]])
combined_test_labels = np.vstack([mnist.test.labels[:num_test], mnist.test.labels[:num_test]])
combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
        np.tile([0., 1.], [num_test, 1])])
#imshow_grid(mnist_train)
#imshow_grid(mnistm_train)
batch_size = 128
class MNISTModel(object):
    """Simple MNIST domain adaptation model."""
    def __init__(self):
        self._build_model()
    
    def _build_model(self):        
        self.X = tf.placeholder(tf.uint8, [None, 28, 28, 3])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.domain = tf.placeholder(tf.float32, [None, 2])
        self.l = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])
        self.keep_prob = tf.placeholder(tf.float32)
        X_input = (tf.cast(self.X, tf.float32) - pixel_mean) / 255.
    
        # CNN model for feature extraction
        #with tf.device('/gpu:0'):
        with tf.variable_scope('feature_extractor'):
            W_conv0 = weight_variable([5, 5, 3, 32])
            b_conv0 = bias_variable([32])
            h_conv0 = tf.nn.relu(conv2d(X_input, W_conv0) + b_conv0)
            h_pool0 = max_pool_2x2(h_conv0)

            W_conv1 = weight_variable([5, 5, 32, 48])
            b_conv1 = bias_variable([48])
            h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)
            print self.keep_prob
            h_fc1_drop = tf.nn.dropout(h_pool1, self.keep_prob)
            # The domain-invariant feature
            self.feature = tf.reshape(h_fc1_drop, [-1, 7*7*48])

        # MLP for class prediction
        with tf.variable_scope('label_predictor'):

            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            all_features = lambda: self.feature
            source_features = lambda: tf.slice(self.feature, [0, 0], [batch_size / 2, -1])
            classify_feats = tf.cond(self.train, source_features, all_features)

            all_labels = lambda: self.y
            source_labels = lambda: tf.slice(self.y, [0, 0], [batch_size / 2, -1])
            self.classify_labels = tf.cond(self.train, source_labels, all_labels)

            W_fc0 = weight_variable([7 * 7 * 48, 100])
            b_fc0 = bias_variable([100])
            h_fc0 = tf.nn.relu(tf.matmul(classify_feats, W_fc0) + b_fc0)

            W_fc1 = weight_variable([100, 100])
            b_fc1 = bias_variable([100])
            h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)

            W_fc2 = weight_variable([100, 10])
            b_fc2 = bias_variable([10])
            logits = tf.matmul(h_fc1, W_fc2) + b_fc2

            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits, self.classify_labels)

        # Small MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):

            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(self.feature, self.l)

            d_W_fc0 = weight_variable([7 * 7 * 48, 100])
            d_b_fc0 = bias_variable([100])
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)

            d_W_fc1 = weight_variable([100, 2])
            d_b_fc1 = bias_variable([2])
            d_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1

            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(d_logits, self.domain)
# Build the model graph
graph = tf.get_default_graph()
with graph.as_default():
    model = MNISTModel()    
    learning_rate = tf.placeholder(tf.float32, [])    
    #keep_prob = tf.placeholder(tf.float32)
    pred_loss = tf.reduce_mean(model.pred_loss)
    domain_loss = tf.reduce_mean(model.domain_loss)
    total_loss = pred_loss + domain_loss
    regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
    dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)    
    # Evaluation
    correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
    domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))
# Params
num_steps = 10000

def train_and_evaluate(training_mode, graph, model, verbose=True):
    """Helper to run the model with different training modes."""
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with tf.Session(graph=graph,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.initialize_all_variables().run()
        # Batch generators
        gen_source_batch = batch_generator(
            [mnist_train, mnist.train.labels], batch_size / 2)
        gen_target_batch = batch_generator(
            [mnistm_train, mnist.train.labels], batch_size / 2)
        gen_source_only_batch = batch_generator(
            [mnist_train, mnist.train.labels], batch_size)
        gen_target_only_batch = batch_generator(
            [mnistm_train, mnist.train.labels], batch_size)

        domain_labels = np.vstack([np.tile([1., 0.], [batch_size / 2, 1]),
                                   np.tile([0., 1.], [batch_size / 2, 1])])

        # Training loop
        for i in range(num_steps):
            print i
            # Adaptation param and learning rate schedule as described in the paper
            p = float(i) / num_steps
            l = 2. / (1. + np.exp(-10. * p)) - 1
            lr = 0.01 / (1. + 10 * p)**0.75
            dropout = 0.50
            # Training step
            if training_mode == 'dann':

                X0, y0 = gen_source_batch.next()
                X1, y1 = gen_target_batch.next()
                X = np.vstack([X0, X1])
                y = np.vstack([y0, y1])

                _, batch_loss, dloss, ploss, d_acc, p_acc = \
                    sess.run([dann_train_op, total_loss, domain_loss, pred_loss, domain_acc, label_acc],
                             feed_dict={model.X: X, model.y: y, model.domain: domain_labels,
                                        model.train: True, model.l: l, learning_rate: lr,model.keep_prob:dropout})

                if verbose and i % 100 == 0:
                    print 'loss: %f  d_acc: %f  p_acc: %f  p: %f  l: %f  lr: %f' % \
                            (batch_loss, d_acc, p_acc, p, l, lr)

            elif training_mode == 'source':
                X, y = gen_source_only_batch.next()
                _, batch_loss = sess.run([regular_train_op, pred_loss],
                                     feed_dict={model.X: X, model.y: y, model.train: False,
                                                model.l: l, learning_rate: lr,model.keep_prob:dropout})

            elif training_mode == 'target':
                X, y = gen_target_only_batch.next()
                _, batch_loss = sess.run([regular_train_op, pred_loss],
                                     feed_dict={model.X: X, model.y: y, model.train: False,
                                                model.l: l, learning_rate: lr,model.keep_prob:dropout})

        # Compute final evaluation on test data
        #with tf.device('/gpu:0'):
        
        gen_source_batch = batch_generator(
            [mnist_test, mnist.test.labels], batch_size)
        gen_target_batch = batch_generator(
            [mnistm_test, mnist.test.labels], batch_size)
        domain_labels = np.vstack([np.tile([1., 0.], [batch_size, 1])])
        num_iter = int(mnist_test.shape[0]/batch_size)
        step = 0
        total_source = 0
        total_target = 0
        while step < num_iter:
             X0, y0 = gen_source_batch.next()
             X1, y1 = gen_target_batch.next()           
             source_acc = sess.run(label_acc,feed_dict={model.X: X0, model.y: y0,model.train: False,model.keep_prob:1})
            
             target_acc = sess.run(label_acc,
                                   feed_dict={model.X: X1, model.y: y1,model.train: False,model.keep_prob:1})
             total_source += source_acc
             total_target += target_acc
             step += 1

    return total_source/num_iter,total_target/num_iter

print '\nSource only training'
source_acc, target_acc, = train_and_evaluate('source', graph, model)
print 'Source (MNIST) accuracy:', source_acc
print 'Target (MNIST-M) accuracy:', target_acc
print '\nDomain adaptation training'
source_acc, target_acc = train_and_evaluate('dann', graph, model)
print 'Source (MNIST) accuracy:', source_acc
print 'Target (MNIST-M) accuracy:', target_acc
print 'Domain accuracy:', d_acc
