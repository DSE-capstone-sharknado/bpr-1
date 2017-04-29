# coding: utf-8

import tensorflow as tf
import os
import cPickle as pickle
import numpy
import random
from multiprocessing import Process, Queue


import sys
from utils import load_data, load_image_features

data_path = os.path.join('data/amzn/', 'review_Women.csv')
user_count, item_count, users, items, user_ratings = load_data(data_path)

#items: asin -> iid

  
images_path = "data/amzn/image_features_Women.b"
image_features = load_image_features(images_path, items)    
    
print "extracted image feature count: ",len(image_features)





train_queue = Queue(4)
def uniform_sample_batch(train_ratings, item_count, image_features, sample_count=20000, batch_size=512):
    for i in range(sample_count):
        t = []
        iv = []
        jv = []
        for b in xrange(batch_size):
            u = random.sample(train_ratings.keys(), 1)[0]
            i = random.sample(train_ratings[u], 1)[0]
            j = random.randint(0, item_count-1)
            while j in train_ratings[u]:
                j = random.randint(0, item_count-1)
            
            
            #sometimes there will not be an image for given product
            try:
              image_features[i]
              image_features[j]
            except KeyError:
              continue
            iv.append(image_features[i])
            jv.append(image_features[j])
            t.append([u, i, j])
                
        # block if queue is full
        train_queue.put( (numpy.asarray(t), numpy.vstack(tuple(iv)), numpy.vstack(tuple(jv))), True )
    train_queue.put(None)

def train_data_process(sample_count=20000, batch_size=512):
    p = Process(target=uniform_sample_batch, args=(user_ratings, item_count, image_features, sample_count, batch_size))
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


def test_batch_generator_by_user(train_ratings, test_ratings, item_count, image_features):
    # using leave one cv
    for u in random.sample(test_ratings.keys(), 3000):
        i = test_ratings[u]
        t = []
        ilist = []
        jlist = []
        count=0
        for j in range(item_count):
            # find item not in test[u] and train[u]
            if j != test_ratings[u] and not (j in train_ratings[u]):
                try:
                  image_features[i]
                  image_features[j]
                except KeyError:
                  continue
                  
                count+=1
                t.append([u, i, j])
                ilist.append(image_features[i])
                jlist.append(image_features[j])
        
        # print numpy.asarray(t).shape
        # print numpy.vstack(tuple(ilist)).shape
        # print numpy.vstack(tuple(jlist)).shape
        if(len(ilist)==0):
          print "if count ==0, could not find neg item for user, count: ",count,u
          continue
        yield numpy.asarray(t), numpy.vstack(tuple(ilist)), numpy.vstack(tuple(jlist))


def vbpr(user_count, item_count, hidden_dim=20, hidden_img_dim=128, 
         learning_rate = 0.001,
         l2_regulization = 0.01, 
         bias_regulization=1.0):
    """
    user_count: total number of users
    item_count: total number of items
    hidden_dim: hidden feature size of MF
    hidden_img_dim: [4096, hidden_img_dim]
    """
    u = tf.placeholder(tf.int32, [None])
    i = tf.placeholder(tf.int32, [None])
    j = tf.placeholder(tf.int32, [None])
    iv = tf.placeholder(tf.float32, [None, 4096])
    jv = tf.placeholder(tf.float32, [None, 4096])
    
    #model parameters -- LEARN THESE
    #latent factors
    user_emb_w = tf.get_variable("user_emb_w", [user_count+1, hidden_dim], initializer=tf.random_normal_initializer(0, 0.1))
    item_emb_w = tf.get_variable("item_emb_w", [item_count+1, hidden_dim], initializer=tf.random_normal_initializer(0, 0.1))
    
    #UxD visual factors for users
    user_img_w = tf.get_variable("user_img_w", [user_count+1, hidden_img_dim],initializer=tf.random_normal_initializer(0, 0.1))
    #this is E, the embedding matrix
    img_emb_w = tf.get_variable("image_embedding_weights", [4096, hidden_img_dim], initializer=tf.random_normal_initializer(0, 0.1))
    
    #biases
    item_b = tf.get_variable("item_b", [item_count+1, 1], initializer=tf.constant_initializer(0.0))
    #user bias just cancels out it seems
    #missing visual bias?
    
    #pull out the respective latent factor vectors for a given user u and items i & j
    u_emb = tf.nn.embedding_lookup(user_emb_w, u)
    i_emb = tf.nn.embedding_lookup(item_emb_w, i)
    j_emb = tf.nn.embedding_lookup(item_emb_w, j)
    #pull out the visual factor, 1 X D for user u
    u_img = tf.nn.embedding_lookup(user_img_w, u)
    #get the respective biases for items i & j
    i_b = tf.nn.embedding_lookup(item_b, i)
    j_b = tf.nn.embedding_lookup(item_b, j)

    # MF predict: u_i > u_j
    theta_i = tf.matmul(iv, img_emb_w) # (f_i * E), eq. 3
    theta_j = tf.matmul(jv, img_emb_w) # (f_j * E), eq. 3
    xui = i_b + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1, keep_dims=True) + tf.reduce_sum(tf.multiply(u_img, theta_i), 1, keep_dims=True)
    xuj = j_b + tf.reduce_sum(tf.multiply(u_emb, j_emb), 1, keep_dims=True) + tf.reduce_sum(tf.multiply(u_img, theta_j), 1, keep_dims=True)
    xuij = xui - xuj

    # auc score is used in test/cv
    # reduce_mean is reasonable BECAUSE
    # all test (i, j) pairs of one user is in ONE batch
    auc = tf.reduce_mean(tf.to_float(xuij > 0))

    l2_norm = tf.add_n([
            tf.reduce_sum(tf.multiply(u_emb, u_emb)), 
            tf.reduce_sum(tf.multiply(u_img, u_img)),
            tf.reduce_sum(tf.multiply(i_emb, i_emb)),
            tf.reduce_sum(tf.multiply(j_emb, j_emb)),
            tf.reduce_sum(tf.multiply(img_emb_w, img_emb_w)),
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
    
    for epoch in range(1, 20):
        print "epoch ", epoch
        _loss_train = 0.0
        sample_count = 1
        batch_size = 512
        p = train_data_process(sample_count, batch_size)
        p.start()
        data = train_queue.get(True) #block if queue is empty
        while data:
            d, _iv, _jv = data
            _loss, _ = session.run([loss, train_op], feed_dict={u:d[:,0], i:d[:,1], j:d[:,2], iv:_iv, jv:_jv})
            _loss_train += _loss
            data = train_queue.get(True)
        p.join()
        print "train_loss:", _loss_train/sample_count

        # if epoch % 10 != 0:
        #     continue
        #
        # auc_values=[]
        # _loss_test = 0.0
        # user_count = 0
        # for d, _iv, _jv in test_batch_generator_by_user(user_ratings, user_ratings_test, item_count, image_features):
        #     user_count += 1
        #     _loss, _auc = session.run([loss, auc], feed_dict={u:d[:,0], i:d[:,1], j:d[:,2], iv:_iv, jv:_jv})
        #     _loss_test += _loss
        #     auc_values.append(_auc)
        # print "test_loss: ", _loss_test/user_count, " auc: ", numpy.mean(auc_values)
        # print ""

