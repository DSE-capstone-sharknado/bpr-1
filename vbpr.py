import tensorflow as tf
import os
import cPickle as pickle
import numpy
import random
import time
import numpy as np

from utils import load_image_features, load_simple, stats



def generate_val_and_test(user_ratings):
    '''
    for each user, random select one rating into test set
    '''
    user_test = dict()
    user_val = dict()
    for u, i_list in user_ratings.items():
        samples = random.sample(i_list, 2)
        user_test[u] = samples[0]
        user_val[u] = samples[1]
    return user_val, user_test

def uniform_sample_batch(train_ratings, val_ratings, test_ratings, item_count, image_features, sample_count=400, batch_size=512):
    for i in xrange(sample_count):
        t = []
        iv = []
        jv = []
        for b in xrange(batch_size):
            u = random.randint(1, len(train_ratings))
            
            i = random.sample(train_ratings[u], 1)[0]
            while i == test_ratings[u] or i==val_ratings[u]: #make sure i is not in the test or val set
                i = random.sample(train_ratings[u], 1)[0]
            
            j = random.randint(0, item_count)
            while j in train_ratings[u]: #make sure j is not in users reviews (ie it is negative)
                j = random.randint(0, item_count)
            
            #sometimes there will not be an image for given item i or j
            try:
              image_features[i]
              image_features[j]
            except KeyError:
              continue  #if so, skip this item
            
            t.append([u, i, j])
            iv.append(image_features[i])
            jv.append(image_features[j])
        yield numpy.asarray(t), numpy.vstack(tuple(iv)), numpy.vstack(tuple(jv))

def test_batch_generator_by_user(train_ratings, test_ratings, item_count, image_features, sample_size=3000, neg_sample_size=1000, cold_start=False):
    # using leave one cv
    for u in random.sample(test_ratings.keys(), sample_size): #uniform random sampling w/o replacement
        t = []
        ilist = []
        jlist = []
        
        i = test_ratings[u]
        #check if we have an image for i, sometimes we dont...
        if i not in image_features:
          continue
        
        #filter for cold start
        if cold_start and item_dist[i] > 5:
          continue
        
        for _ in xrange(neg_sample_size):
            j = random.randint(1, item_count)
            if j != test_ratings[u] and not (j in train_ratings[u]):
                # find negative item not in train or test set
                
                #sometimes there will not be an image for given product
                try:
                  image_features[i]
                  image_features[j]
                except KeyError:
                  continue  #if image not found, skip item
                
                t.append([u, i, j])
                ilist.append(image_features[i])
                jlist.append(image_features[j])
        
        yield numpy.asarray(t), numpy.vstack(tuple(ilist)), numpy.vstack(tuple(jlist))

def vbpr(user_count, item_count, hidden_dim=20, hidden_img_dim=128,
         learning_rate=0.001,
          l2_regulization=0.001,
          bias_regulization=0.001,
          embed_regulization = 0.001,
          image_regulization = 0.0,
          visual_bias_regulization=0.0):
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
                                initializer=tf.random_normal_initializer(0, 0.1)) #theta_u
    item_emb_w = tf.get_variable("item_emb_w", [item_count+1, hidden_dim],
                                initializer=tf.random_normal_initializer(0, 0.1))
    item_b = tf.get_variable("item_b", [item_count+1, 1],
                                initializer=tf.constant_initializer(0.0))
    visual_bias = tf.get_variable("visual_bias", [1, 4096], initializer=tf.constant_initializer(0.0))
    
    img_emb_w = tf.get_variable("image_embedding_weights", [4096, hidden_img_dim],
                               initializer=tf.random_normal_initializer(0, 0.1))
                        
    
    
    #lookup the latent factors by user and id
    u_emb = tf.nn.embedding_lookup(user_emb_w, u)
    u_img = tf.nn.embedding_lookup(user_img_w, u)
    
    i_emb = tf.nn.embedding_lookup(item_emb_w, i)
    i_b = tf.nn.embedding_lookup(item_b, i)
    j_emb = tf.nn.embedding_lookup(item_emb_w, j)
    j_b = tf.nn.embedding_lookup(item_b, j)

                               

    
    # MF predict: u_i > u_j
    theta_i = tf.matmul(iv, img_emb_w) # (f_i * E), eq. 3 1xK2 x 4096xK2 => 1xK2 #plot these on 2d scatter
    theta_j = tf.matmul(jv, img_emb_w) # (f_j * E), eq. 3
    xui = i_b + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1, keep_dims=True) + tf.reduce_sum(tf.multiply(u_img, theta_i), 1, keep_dims=True) + tf.reduce_sum(tf.multiply(visual_bias, iv), 1, keep_dims=True)
    xuj = j_b + tf.reduce_sum(tf.multiply(u_emb, j_emb), 1, keep_dims=True) + tf.reduce_sum(tf.multiply(u_img, theta_j), 1, keep_dims=True) + tf.reduce_sum(tf.multiply(visual_bias, jv), 1, keep_dims=True)
    xuij = xui - xuj
    #
    
    # auc score is used in test/cv
    # reduce_mean is reasonable BECAUSE
    # all test (i, j) pairs of one user is in ONE batch
    auc = tf.reduce_mean(tf.to_float(xuij > 0))
    
    l2_norm = tf.add_n([
            l2_regulization * tf.reduce_sum(tf.multiply(u_emb, u_emb)),
            image_regulization * tf.reduce_sum(tf.multiply(u_img, u_img)),
            l2_regulization * tf.reduce_sum(tf.multiply(i_emb, i_emb)),
            l2_regulization * tf.reduce_sum(tf.multiply(j_emb, j_emb)),
            embed_regulization * tf.reduce_sum(tf.multiply(img_emb_w, img_emb_w)),
            bias_regulization * tf.reduce_sum(tf.multiply(i_b, i_b)),
            bias_regulization * tf.reduce_sum(tf.multiply(j_b, j_b)),
            visual_bias_regulization * tf.reduce_sum(tf.multiply(visual_bias,visual_bias))
        ])
    
    loss = l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(xuij)))
    #train_op =  tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
    #train_op =  tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    train_op =  tf.train.AdamOptimizer().minimize(loss)
    # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    print "Hyper-parameters: K=%d, K2=%d, lr=%f, l2r=%f"%(hidden_dim, hidden_img_dim, learning_rate, l2_regulization)
    return u, i, j, iv, jv, loss, auc, train_op



print "Loading dataset..."
data_dir = os.path.join("data", "amzn")
simple_path = os.path.join(data_dir, 'reviews_Women.txt')

users, items, reviews_all = load_simple(simple_path, user_min=5)
print "generating stats..."
user_dist, item_dist, train_ratings, item_users = stats(reviews_all)

user_count = len(train_ratings)
item_count = len(item_users)
reviews_count = len(reviews_all)
print user_count,item_count,reviews_count

  

images_path = "data/amzn/image_features_Women.b"
image_features = load_image_features(images_path, items)

print "extracted image feature count: ",len(image_features)
#4096 floats * 24bytes = 98,304 bytes/feature = 98KB
#4096 * 24 * 302,047 = 29 GB


val_ratings, test_ratings = generate_val_and_test(train_ratings)

sample_count = 400
batch_size = 612
epochs =255 # ideally we should not hard code this. GD should terminate when loss converges
K=10
K2=10
lam=10.0
bias_reg=.01

with tf.Graph().as_default(), tf.Session() as session:
    with tf.variable_scope('vbpr'):
        u, i, j, iv, jv, loss, auc, train_op = vbpr(user_count, item_count, hidden_dim=K, hidden_img_dim=K2, l2_regulization =lam, bias_regulization=bias_reg)
    
    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())
    
    epoch_durations = []
    best_auc=-1
    best_iter=-1
    for epoch in range(1, epochs):
        epoch_start = time.time()
        train_loss_vals=[]
        for d, _iv, _jv in uniform_sample_batch(train_ratings, val_ratings, test_ratings, item_count, image_features, sample_count=sample_count, batch_size=batch_size ):
            _loss, _ = session.run([loss, train_op], feed_dict={ u:d[:,0], i:d[:,1], j:d[:,2], iv:_iv, jv:_jv})
            train_loss_vals.append(_loss)
        epoch_durations.append(time.time() - epoch_start)
        
        if epoch % 5 != 0:
          print "epoch: %d (%.2fs)"%(epoch, np.mean(epoch_durations) )
          continue
        
        #train eval
        val_auc_vals=[]
        val_loss_vals=[]
        for d, fi, fj in test_batch_generator_by_user(train_ratings, val_ratings, item_count, image_features, sample_size=1000):
            _loss, _auc = session.run([loss, auc], feed_dict={u:d[:,0], i:d[:,1], j:d[:,2], iv:fi, jv:fj})
            val_loss_vals.append(_loss)
            val_auc_vals.append(_auc)
        val_auc = np.mean(val_auc_vals)
        print "epoch: %d (%.2fs), train loss: %.2f, val loss: %.2f, val auc: %.2f"%(epoch, np.mean(epoch_durations), np.mean(train_loss_vals) , np.mean(val_loss_vals), val_auc ),
        
        #early termination/checks for convergance
        if val_auc > best_auc:
          best_auc = val_auc
          best_iter = epoch
          print "*"
          saver.save(session, "logs/")
        elif val_auc < best_iter and epoch >= best_iter+21: #overfitting
          print "Overfitted. Exiting..."
          break
        else:
          print
    
    #restore best model from checkpoint
    saver.restore(session, "logs/")
    #test auc
    test_auc_vals=[]
    for d, fi, fj in test_batch_generator_by_user(train_ratings, test_ratings, item_count, image_features, sample_size=3000):
        _auc = session.run(auc, feed_dict={u:d[:,0], i:d[:,1], j:d[:,2], iv:fi, jv:fj})
        test_auc_vals.append(_auc)
    test_auc = np.mean(test_auc_vals)
    print "Best model iteration %d, test: %.3f, val: %.3f"%(best_iter,test_auc,best_auc)
    
    
    #cold auc
    cold_auc_vals=[]
    for d, fi, fj in test_batch_generator_by_user(train_ratings, test_ratings, item_count, image_features, sample_size=3000, cold_start=True):
        _auc = session.run(auc, feed_dict={u:d[:,0], i:d[:,1], j:d[:,2], iv:fi, jv:fj})
        cold_auc_vals.append(_auc)
    print "cold auc: %.2f"%(np.mean(cold_auc_vals) )

     
        


# nohup time python -u vbpr.py > vbpr3-test005.log 2>&1 &

# nohup time ./train ../tf-bpr/data/amzn/reviews_Women.txt ../tf-bpr/data/amzn/image_features_Women.b 10 10 na 0.01 9.5 0.01 na 20 "women" > logs/vbpr-test001.log 2>&1 &