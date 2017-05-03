import numpy
import tensorflow as tf
import os
import random
from utils import load_data_simple, load_image_features, load_data


def generate_test(user_ratings):
    '''
    for each user, random select one rating into test set
    '''
    user_test = dict()
    for u, i_list in user_ratings.items():
        user_test[u] = random.sample(user_ratings[u], 1)[0]
    return user_test



def generate_train_batch(user_ratings, user_ratings_test, item_count, batch_size=512):
    '''
    uniform sampling (user, item_rated, item_not_rated)
    '''
    t = []
    for b in xrange(batch_size):
        u = random.sample(user_ratings.keys(), 1)[0] #random user u
        
        i = random.sample(user_ratings[u], 1)[0] #random item i from u
        while i == user_ratings_test[u]: #make sure it's not in the test set
            i = random.sample(user_ratings[u], 1)[0]
            
        j = random.randint(1, item_count) #random item j
        while j in user_ratings[u]: #make sure it's not in u's items
            j = random.randint(1, item_count)
            
        t.append([u, i, j])
    return numpy.asarray(t)

#sample users but not items
def generate_test_batch(user_ratings, user_ratings_test, item_count, users_size=512):
    '''
    for an user u and an item i rated by u, 
    generate pairs (u,i,j) for all item j which u has't rated
    it's convenient for computing AUC score for u
    '''
  
    #time: O(UxI)
    #space: UxI = 39387*23033~=1B
    # print "generating test batch: ",len(user_ratings.keys())*item_count
    
    
    #emits a batch of rankings for each user
    for u in random.sample(user_ratings.keys(), users_size):
        t = []
        i = user_ratings_test[u]
        for j in range(1, item_count):
            if not (j in user_ratings[u]):
                t.append([u, i, j])
        yield numpy.asarray(t) #returns a batch per user
        
def bpr_mf(user_count, item_count, hidden_dim, starter_learning_rate=0.1, regulation_rate = 0.0001):
  
    #model input
    u = tf.placeholder(tf.int32, [None])
    i = tf.placeholder(tf.int32, [None])
    j = tf.placeholder(tf.int32, [None])

    #model paramenters
    #latent factors
    #hidden_dim is the k hyper parameter
    user_emb_w = tf.get_variable("user_emb_w", [user_count+1, hidden_dim], initializer=tf.random_normal_initializer(0, 0.1))
    item_emb_w = tf.get_variable("item_emb_w", [item_count+1, hidden_dim], initializer=tf.random_normal_initializer(0, 0.1))
    item_b = tf.get_variable("item_b", [item_count+1, 1], initializer=tf.constant_initializer(0.0))    #item bias
    user_b = tf.get_variable("user_b", [user_count+1, 1], initializer=tf.constant_initializer(0.0))    #user bias
    
    u_emb = tf.nn.embedding_lookup(user_emb_w, u) #lookup the latent factor for user u
    i_emb = tf.nn.embedding_lookup(item_emb_w, i) #lookup the latent factor fo item i
    i_b = tf.nn.embedding_lookup(item_b, i)       #lookup the bias vector for item i
    j_emb = tf.nn.embedding_lookup(item_emb_w, j) #lookup the latent factor for item j
    j_b = tf.nn.embedding_lookup(item_b, j)       #lookup the bias vector for item js
    u_b = tf.nn.embedding_lookup(user_b, u)       #lookup bias scalar for user u
    
    # MF predict: u_i > u_j
    # xuij = xui - xuj
    xui = u_b + i_b + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1, keep_dims=True)
    xuj = u_b + j_b + tf.reduce_sum(tf.multiply(u_emb, j_emb), 1, keep_dims=True)
    xuij = xui-xuj
    
    # AUC for one user:
    # reasonable iff all (u,i,j) pairs are from the same user
    # 
    # average AUC = mean( auc for each user in test set)
    mf_auc = tf.reduce_mean(tf.to_float(xuij > 0)) # xui - xui > 0 == xui > xuj
    
    l2_norm = tf.add_n([
            tf.reduce_sum(tf.multiply(u_emb, u_emb)), 
            tf.reduce_sum(tf.multiply(i_emb, i_emb)),
            tf.reduce_sum(tf.multiply(j_emb, j_emb)),
            #reg for biases
            tf.reduce_sum(tf.multiply(i_b, i_b)),
            tf.reduce_sum(tf.multiply(j_b, j_b)),
            tf.reduce_sum(tf.multiply(u_b, u_b))
        ])
    

    bprloss = regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(xuij))) #BPR loss
    
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 400, 0.8, staircase=True)
    #.1 ... .001
                                               
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(bprloss, global_step=global_step)
    return u, i, j, mf_auc, bprloss, train_op
    
#start
data_dir = os.path.join("data", "amzn")
simple_path = os.path.join(data_dir, 'reviews_Women_5.txt')
users_lut, items_lut, reviews_count, user_ratings = load_data_simple(simple_path, min_items=5)
user_count = len(users_lut)
item_count = len(items_lut)

user_ratings_test = generate_test(user_ratings)

# 512*x=239290; x=400
#UxIxJ = 30k x 8k x 8k = 
batch_size=512
batches=400
epochs=15
K=20
learning_rate=0.1
regulation_rate = 0.1


    
with tf.Graph().as_default(), tf.Session() as session:
    
    print "training..."
    u, i, j, mf_auc, bprloss, train_op = bpr_mf(user_count, item_count, K, starter_learning_rate=learning_rate, regulation_rate=regulation_rate)
    session.run(tf.global_variables_initializer())
    
    
    for epoch in range(1, epochs+1):
      
        #ideally, I want to run the whole dataset each epoch, so adjust batches and batch size accordingly. 
        #Typically batch size will be fixed and a product of teh hardware.
        _batch_bprloss = 0
        for k in range(1,batches): # uniform batch samples from training set
            if(k%(batches/10)==0):
              print "iteration,batch: ",epoch,k
            
            uij = generate_train_batch(user_ratings, user_ratings_test, item_count, batch_size=batch_size)

            _bprloss, _ = session.run([bprloss, train_op], feed_dict={u:uij[:,0], i:uij[:,1], j:uij[:,2]})
            _batch_bprloss += _bprloss
        
        print "epoch: ", epoch
        print "bpr_loss: ", _batch_bprloss / k

        user_count = 0
        auc_values=[]
        print "auc..."
        # each batch will return only one user's auc
    
        #these samples need to be generated in parallel
        test_samples = generate_test_batch(user_ratings, user_ratings_test, item_count, users_size=3000)
        for t_uij in test_samples:
        
            #process: t_uij
            user_count += 1
            if user_count % 1000 == 0:
              print "Current AUC mean (%s user samples): %0.5f" % (user_count, numpy.mean(auc_values))
            user_auc = session.run(mf_auc, feed_dict={u:t_uij[:,0], i:t_uij[:,1], j:t_uij[:,2]})
            auc_values.append(user_auc)

        print "test_auc: ", numpy.mean(auc_values)
        print ""
    
    summary_writer = tf.summary.FileWriter('log_simple_stats', session.graph)
    summary_writer.close()
        
#8:15 - 8:25: 1 epoch = 10  min * 10 = 100 min = 1 hour half
#typical sampling stragegies I see in BPR are for each epoch
#take n_votes random uij samples, where n_votes=278,677 is the number of obeservatiosn in teh dataset
#but this impl is doing 5000*512=2,560,000 samples per epoch which is way too much data I think
#so if I adjust keep batch size 512 constant, k would be 278677/512=544==600

#512/400 - 8:33-8:45 test_auc:  0.554972909522
#4096/50 - 8:49-9:04 test_auc:  0.513462866851

# run w/o GPU
# CUDA_VISIBLE_DEVICES="" time python bpr.py
#34 min 0.666018

#4:06 - 4:21