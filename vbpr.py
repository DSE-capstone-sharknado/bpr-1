

import tensorflow as tf
import os
import cPickle as pickle
import numpy
import random
import time
import numpy as np

from utils import load_data, load_image_features, load_data_simple



def generate_test(user_ratings):
    '''
    for each user, random select one rating into test set
    '''
    user_test = dict()
    for u, i_list in user_ratings.items():
        user_test[u] = random.sample(user_ratings[u], 1)[0]
    return user_test
def uniform_sample_batch(train_ratings, item_count, image_features, sample_count=20000, batch_size=5):
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
            try: #sometimes there will not be an image for given product
              image_features[i]
              image_features[j]
            except KeyError:
              continue  #skipt this item
            t.append([u, i, j])
            iv.append(image_features[i])
            jv.append(image_features[j])
        yield numpy.asarray(t), numpy.vstack(tuple(iv)), numpy.vstack(tuple(jv))

def test_batch_generator_by_user(train_ratings, test_ratings, item_count, image_features):
    # using leave one cv
    for u in test_ratings.keys():
        i = test_ratings[u]
        t = []
        ilist = []
        jlist = []
        for j in range(item_count):
            if j != test_ratings[u] and not (j in train_ratings[u]):
                # find item not in test[u] and train[u]
                try: #sometimes there will not be an image for given product
                  image_features[i]
                  image_features[j]
                except KeyError:
                  continue  #skipt this item
                
                t.append([u, i, j])
                ilist.append(image_features[i])
                jlist.append(image_features[j])
        if len(ilist)==0: #edge case where no images are found in user test set (bad luck/low probability)
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
    
    user_emb_w = tf.get_variable("user_emb_w", [user_count+1, hidden_dim], 
                                initializer=tf.random_normal_initializer(0, 0.1))
    user_img_w = tf.get_variable("user_img_w", [user_count+1, hidden_img_dim],
                                initializer=tf.random_normal_initializer(0, 0.1))
    item_emb_w = tf.get_variable("item_emb_w", [item_count+1, hidden_dim], 
                                initializer=tf.random_normal_initializer(0, 0.1))
    item_b = tf.get_variable("item_b", [item_count+1, 1], 
                                initializer=tf.constant_initializer(0.0))
    
    u_emb = tf.nn.embedding_lookup(user_emb_w, u)
    u_img = tf.nn.embedding_lookup(user_img_w, u)
    
    i_emb = tf.nn.embedding_lookup(item_emb_w, i)
    i_b = tf.nn.embedding_lookup(item_b, i)
    j_emb = tf.nn.embedding_lookup(item_emb_w, j)
    j_b = tf.nn.embedding_lookup(item_b, j)
    
    img_emb_w = tf.get_variable("image_embedding_weights", [4096, hidden_img_dim], 
                               initializer=tf.random_normal_initializer(0, 0.1))

    img_i_j = tf.matmul(iv - jv,  img_emb_w)

    # MF predict: u_i > u_j
    x = i_b - j_b + tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1, keep_dims=True) + \
        tf.reduce_sum(tf.multiply(u_img, img_i_j),1, keep_dims=True)

    # auc score is used in test/cv
    # reduce_mean is reasonable BECAUSE
    # all test (i, j) pairs of one user is in ONE batch
    auc = tf.reduce_mean(tf.to_float(x > 0))

    l2_norm = tf.add_n([
            tf.reduce_sum(tf.multiply(u_emb, u_emb)), 
            tf.reduce_sum(tf.multiply(u_img, u_img)),
            tf.reduce_sum(tf.multiply(i_emb, i_emb)),
            tf.reduce_sum(tf.multiply(j_emb, j_emb)),
            tf.reduce_sum(tf.multiply(img_emb_w, img_emb_w)),
            bias_regulization * tf.reduce_sum(tf.multiply(i_b, i_b)),
            bias_regulization * tf.reduce_sum(tf.multiply(j_b, j_b))
        ])

    loss = l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(x)))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return u, i, j, iv, jv, loss, auc, train_op
    


# data_path = os.path.join('data/amzn/', 'review_Women.csv')
# user_count, item_count, users, items, train_ratings = load_data(data_path)
simple_path = os.path.join('data', 'amzn', 'reviews_Women_5.txt')
users, items, reviews_count, train_ratings = load_data_simple(simple_path, min_items=5)
user_count = len(users)
item_count = len(items)
print user_count,item_count,reviews_count

#items: asin -> iid

  
images_path = "data/amzn/image_features_Women.b"
image_features = load_image_features(images_path, items)    
    
print "extracted image feature count: ",len(image_features)

test_ratings = generate_test(train_ratings)

sample_count = 1600
batch_size = 128
epochs =21 # ideally we should not hard code this. GD should terminate when loss converges.
        

with tf.Graph().as_default(), tf.Session() as session:
    with tf.variable_scope('vbpr'):
        u, i, j, iv, jv, loss, auc, train_op = vbpr(user_count, item_count)
    
    session.run(tf.global_variables_initializer())
    
    epoch_durations = []
    for epoch in range(1, epochs):
        print "epoch ", epoch
        epoch_start_time = time.time()
        _loss_train = 0.0

        for d, _iv, _jv in uniform_sample_batch(train_ratings, item_count, image_features, batch_size=batch_size, sample_count=sample_count):
            _loss, _ = session.run([loss, train_op], feed_dict={ u:d[:,0], i:d[:,1], j:d[:,2], iv:_iv, jv:_jv})
            _loss_train += _loss
        print "train_loss:", _loss_train/sample_count
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_durations.append(epoch_duration)
        print "epoch time: ",epoch_duration,", avg: ",np.mean(epoch_durations)
        
        if epoch % 20 != 0:
            continue
        
        _auc_all = 0
        _loss_test = 0.0
        user_count=0
        dur_sum=0
        _test_user_count = len(test_ratings)
        for d, fi, fj in test_batch_generator_by_user(train_ratings, test_ratings, item_count, image_features):
            s = time.time()    
            _loss, _auc = session.run([loss, auc], feed_dict={u:d[:,0], i:d[:,1], j:d[:,2], iv:fi, jv:fj})
            _loss_test += _loss
            _auc_all += _auc
            user_count+=1
            e=time.time()
            dur=e-s
            dur_sum+=dur
            print "avg,user: ",dur_sum/user_count,dur #seems like it takes about 10s per user
        print "test_loss: ", _loss_test/_test_user_count, " auc: ", _auc_all/_test_user_count
        print ""
        
        