# this was an experiemet to see see if there was a bottlenech generating the training batches
# it starts up a process that queues up training batches which the training process consumes.
# there was no speedup, so therefore there was no bottleneck. 

import numpy
import tensorflow as tf
import os
import random
from collections import defaultdict
from multiprocessing import Process, Queue



def load_data(data_path):
    '''
    As for bpr experiment, all ratings are removed.
    '''
    user_ratings = defaultdict(set)
    max_u_id = -1
    max_i_id = -1
    user_min=5
    user_counts={}
    user_count=0
    item_count=0
    reviews=0
    users={} #aid to id LUT
    items={} #asid to id LUT
    with open(data_path, 'r') as f:
        for line in f.readlines():
            reviews+=1
            auid, asid, _= line.split(",")
            u, i = None, None
            
            if auid in users:
              u = users[auid]
            else:
              user_count+=1 #new user so increment
              users[auid]=user_count
              u = user_count
              user_counts[u]=1
            
            if asid in items:
              i = items[asid]
            else:
              item_count+=1 #new i so increment
              items[asid]=item_count
              i=item_count
            
            user_counts[u] += 1
            user_ratings[u].add(i)
            max_u_id = max(u, max_u_id)
            max_i_id = max(i, max_i_id)
            
    print "max_u_id: ", max_u_id
    print "max_i_id: ", max_i_id
    print "reviews : ", reviews 
    
    # #now filter out users w/ not enough purchase history
    # filtered_user_ratings=defaultdict(set)
    # for k,v in user_ratings.iteritems():
    #   if len(v) >= user_min:
    #     filtered_user_ratings[k]=v
    #
    # print "unfiltered len:",len(user_ratings)
    # print" filtered len:",len(filtered_user_ratings)
    
    #filter out users w/ less than X reviews
    user_ratings_filtered = defaultdict(set)
    for u,ids in user_ratings.items():
      if len(ids)>1:
        #keep
        user_ratings_filtered[u]=ids
    
    return max_u_id, max_i_id, user_ratings_filtered
    



def generate_test(user_ratings):
    '''
    for each user, random select one rating into test set
    '''
    user_test = dict()
    for u, i_list in user_ratings.items():
        user_test[u] = random.sample(user_ratings[u], 1)[0]
    return user_test



def generate_train_batch(user_ratings, user_ratings_test, item_count, batch_size=512, batch_count=400):
    '''
    uniform sampling (user, item_rated, item_not_rated)
    '''
    for _ in range(batch_count):
      t = []
      iv = [1,2]
      jv = [1,2]
      for b in xrange(batch_size):
          u = random.sample(user_ratings.keys(), 1)[0] #random user u
        
          i = random.sample(user_ratings[u], 1)[0] #random item i from u
          while i == user_ratings_test[u]: #make sure it's not in the test set
              i = random.sample(user_ratings[u], 1)[0]
            
          j = random.randint(1, item_count) #random item j
          while j in user_ratings[u]: #make sure it's not in u's items
              j = random.randint(1, item_count)
            
          t.append([u, i, j])

      # block if queue is full
      train_queue.put((numpy.asarray(t), numpy.vstack(tuple(iv)), numpy.vstack(tuple(jv))), True)
      # print train_queue.qsize()
    print "end batch"
    train_queue.put(None)
    print "Epoch Training Generation Complete..."
    
def train_data_process(sample_count=20000, batch_size=512):
    p = Process(target=generate_train_batch, args=(user_ratings, user_ratings_test, item_count))
    return p
    
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
    
    # xuij = xui - xuj
    xui = u_b + i_b + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1, keep_dims=True)
    xuj = u_b + j_b + tf.reduce_sum(tf.multiply(u_emb, j_emb), 1, keep_dims=True)
    xuij = xui-xuj
    
    # AUC for one user:
    # reasonable iff all (u,i,j) pairs are from the same user
    # 
    # average AUC = mean( auc for each user in test set)
    mf_auc = tf.reduce_mean(tf.to_float(xuij > 0))
    
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
                                               
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(bprloss, global_step=global_step)
    return u, i, j, mf_auc, bprloss, train_op
    
#start

data_path = os.path.join('data/amzn/', 'review_Women.csv')
# data_path = os.path.join('data/amzn/', 'reviews-women-full.csv')
user_count, item_count, user_ratings = load_data(data_path)

user_ratings_test = generate_test(user_ratings)

# 512*x=239290; x=400
#UxIxJ = 30k x 8k x 8k = 
batch_size=512
batches=400
epochs=15
K=20
learning_rate=0.1
regulation_rate = 0.1
train_queue = Queue(50)


    
with tf.Graph().as_default(), tf.Session() as session:
    
    print "training..."
    u, i, j, mf_auc, bprloss, train_op = bpr_mf(user_count, item_count, K, starter_learning_rate=learning_rate, regulation_rate=regulation_rate)
    session.run(tf.global_variables_initializer())
    
    
    for epoch in range(1, epochs+1):
        print "epoch: ", epoch
        #ideally, I want to run the whole dataset each epoch, so adjust batches and batch size accordingly. 
        #Typically batch size will be fixed and a product of teh hardware.
        _batch_bprloss = 0
        p = train_data_process(batches, batch_size)
        p.start()
        print "process started..."

        bi=1
        d= train_queue.get(True) #block if queue is empty
        while d:
            if(bi%(batches/10)==0):
              print "iteration,batch: ",epoch,bi
            uij = d[0]
            # uij = generate_train_batch(user_ratings, user_ratings_test, item_count, batch_size=batch_size)

            _bprloss, _ = session.run([bprloss, train_op], feed_dict={u:uij[:,0], i:uij[:,1], j:uij[:,2]})
            _batch_bprloss += _bprloss
            d = train_queue.get(True)
            bi+=1
        
        p.join() #wait till all processes are complete
        print "bpr_loss: ", _batch_bprloss / batches

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

#34 min 0.671134