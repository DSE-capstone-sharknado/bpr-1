# coding: utf-8
import tensorflow as tf
import os
import cPickle as pickle
import numpy
import random
from multiprocessing import Process, Queue

import sys
from utils import load_data_hybrid, load_image_features

data_path = os.path.join('', 'review_Women.csv')
user_count, item_count, users, items, user_ratings, brands, prices = load_data_hybrid(data_path)


train_queue = Queue(4)
test_queue = Queue(4)


def generate_train_batch(user_ratings, user_ratings_test, item_count, batch_size=512, batch_count=400):
    '''
    uniform sampling (user, item_rated, item_not_rated)
    '''
    for _ in range(batch_count):
        t = []
        iv = [1, 2]
        jv = [1, 2]
        for b in xrange(batch_size):
            u = random.sample(user_ratings.keys(), 1)[0]  # random user u

            i = random.sample(user_ratings[u], 1)[0]  # random item i from u
            while i == user_ratings_test[u]:  # make sure it's not in the test set
                i = random.sample(user_ratings[u], 1)[0]

            j = random.randint(1, item_count)  # random item j
            while j in user_ratings[u]:  # make sure it's not in u's items
                j = random.randint(1, item_count)

            t.append([u, i, j])

        # block if queue is full
        train_queue.put((numpy.asarray(t), numpy.vstack(tuple(iv)), numpy.vstack(tuple(jv))), True)
        # print train_queue.qsize()
    print "end batch"
    train_queue.put(None)
    print "Epoch Training Generation Complete..."


def train_data_process(batch_size=512, batch_count=400):
    p = Process(target=generate_train_batch, args=(user_ratings, user_ratings_test, item_count, batch_size, batch_count))
    return p



def test_data_process():
    p = Process(target=test_batch_generator_by_user, args=(user_ratings, user_ratings_test, item_count))
    return p


def generate_test(user_ratings):
    '''
    for each user, random select one rating into test set
    '''
    user_test = dict()
    for u, i_list in user_ratings.items():
        user_test[u] = random.sample(user_ratings[u], 1)[0]
    return user_test


user_ratings_test = generate_test(user_ratings)


def test_batch_generator_by_user(train_ratings, test_ratings, item_count):
    # using leave one cv
    for u in random.sample(test_ratings.keys(), 3000):
        i = test_ratings[u]
        t = []
        ilist = [1,2]
        jlist = [1,2]
        count = 0
        for j in range(item_count):
            # find item not in test[u] and train[u]
            if j != test_ratings[u] and not (j in train_ratings[u]):
                count += 1
                t.append([u, i, j])

        # print numpy.asarray(t).shape
        # print numpy.vstack(tuple(ilist)).shape
        # print numpy.vstack(tuple(jlist)).shape
        if (len(ilist) == 0):
            print "if count ==0, could not find neg item for user, count: ", count, u
            continue
        test_queue.put((numpy.asarray(t), numpy.vstack(tuple(ilist)), numpy.vstack(tuple(jlist))), True)
    test_queue.put(None)


def vbpr(user_count, item_count, hidden_dim=20, hidden_img_dim=1, learning_rate=0.01, bias_regulization=0.1):
    """
    user_count: total number of users
    item_count: total number of items
    hidden_dim: hidden feature size of MF
    hidden_img_dim: [4096, hidden_img_dim]
    """
    image_feat_dim = 1
    u = tf.placeholder(tf.int32, [None])
    i = tf.placeholder(tf.int32, [None])
    j = tf.placeholder(tf.int32, [None])
    iv = tf.placeholder(tf.float32, [None, image_feat_dim])
    jv = tf.placeholder(tf.float32, [None, image_feat_dim])

    # model parameters -- LEARN THESE
    # latent factors
    user_emb_w = tf.get_variable("user_emb_w", [user_count + 1, hidden_dim],
                                 initializer=tf.random_normal_initializer(0, 0.1))
    item_emb_w = tf.get_variable("item_emb_w", [item_count + 1, hidden_dim],
                                 initializer=tf.random_normal_initializer(0, 0.1))

    # biases
    item_b = tf.get_variable("item_b", [item_count + 1, 1], initializer=tf.constant_initializer(0.0))
    # user bias just cancels out it seems
    # missing visual bias?

    # pull out the respective latent factor vectors for a given user u and items i & j
    u_emb = tf.nn.embedding_lookup(user_emb_w, u)
    i_emb = tf.nn.embedding_lookup(item_emb_w, i)
    j_emb = tf.nn.embedding_lookup(item_emb_w, j)
    # get the respective biases for items i & j
    i_b = tf.nn.embedding_lookup(item_b, i)
    j_b = tf.nn.embedding_lookup(item_b, j)

    # MF predict: u_i > u_j
    xui = i_b + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1, keep_dims=True)
    xuj = j_b + tf.reduce_sum(tf.multiply(u_emb, j_emb), 1, keep_dims=True)
    xuij = xui - xuj

    # auc score is used in test/cv
    # reduce_mean is reasonable BECAUSE
    # all test (i, j) pairs of one user is in ONE batch
    auc = tf.reduce_mean(tf.to_float(xuij > 0))

    l2_norm = tf.add_n([
        tf.reduce_sum(tf.multiply(u_emb, u_emb)),
        tf.reduce_sum(tf.multiply(i_emb, i_emb)),
        tf.reduce_sum(tf.multiply(j_emb, j_emb)),
        bias_regulization * tf.reduce_sum(tf.multiply(i_b, i_b)),
        bias_regulization * tf.reduce_sum(tf.multiply(j_b, j_b))
    ])

    loss = l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(xuij)))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return u, i, j, iv, jv, loss, auc, train_op


# In[17]:

# user_count = len(user_id_mapping)
# item_count = len(item_id_mapping)

with tf.Graph().as_default(), tf.Session() as session:
    with tf.variable_scope('vbpr'):
        u, i, j, iv, jv, loss, auc, train_op = vbpr(user_count, item_count)
    
    session.run(tf.global_variables_initializer())
    
    for epoch in range(1, 11):
        print "epoch ", epoch
        _loss_train = 0.0
        sample_count = 400
        batch_size = 512
        p = train_data_process(sample_count, batch_size)
        p.start()
        data = train_queue.get(True) #block if queue is empty
        while data:
            d, _iv, _jv = data
            _loss, _ = session.run([loss, train_op], feed_dict={u:d[:,0], i:d[:,1], j:d[:,2]})
            _loss_train += _loss
            data = train_queue.get(True)
        p.join()
        print "train_loss:", _loss_train/sample_count
        
        p2 = test_data_process()
        p2.start()
        auc_values=[]
        _loss_test = 0.0
        user_count = 0
        data = test_queue.get(True) #block if queue is empty
        while data:
            d, _iv, _jv = data
            user_count += 1
            _loss, _auc = session.run([loss, auc], feed_dict={u:d[:,0], i:d[:,1], j:d[:,2]})
            _loss_test += _loss
            auc_values.append(_auc)
            data = test_queue.get(True)
        p2.join()
        print "test_loss: ", _loss_test/user_count, " auc: ", numpy.mean(auc_values)
        print ""

