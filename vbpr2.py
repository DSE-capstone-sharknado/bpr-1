from utils import load_image_features, load_simple, stats
import tensorflow as tf
import random
import time
import numpy as np

class Vbpr2(object):
  """docstring for Vbpr2"""
  def __init__(self, arg):
    super(Vbpr2, self).__init__()
    self.arg = arg

  
  def load_dataset(self, path):
    print "loading dataset..."
    users, items, reviews_all = load_simple(path, user_min=5)
    print "generating stats..."
    user_dist, item_dist, train_ratings, item_users = stats(reviews_all)
    
    user_count = len(train_ratings)
    item_count = len(item_users)
    reviews_count = len(reviews_all)
    print user_count,item_count,reviews_count
    
    self.user_count = user_count
    self.item_count = item_count
    self.reviews_count = reviews_count
    self.user_dist=user_dist
    self.item_dist=item_dist
    self.user_items=train_ratings
    self.item_users=item_users


  
    images_path = "data/amzn/image_features_Women.b"
    self.image_features = load_image_features(images_path, items)    
    
    print "extracted image feature count: ",len(self.image_features)
    
  def generate_val_test(self):
    '''
    for each user, random select one rating into test set
    '''
    user_val = dict()
    user_test = dict()
    
    for u, i_list in self.user_items.items():
        samples = random.sample(i_list, 2) #w/o replacement
        
        user_test[u] = samples[0]
        user_val[u] = samples[1]
        
    return user_val, user_test
    

    
  def uniform_sample_batch(self, sample_count=400, batch_size=512):
    for i in xrange(sample_count):
      t = []
      iv = []
      jv = []
      for b in xrange(batch_size):
        u = random.randint(1, len(self.user_items))

        i = random.sample(self.user_items[u], 1)[0]
        while i == self.test_ratings[u]: #make sure i is not in the test set
          i = random.sample(self.user_items[u], 1)[0]
    
        j = random.randint(0, self.item_count)
        while j in self.user_items[u]:
          j = random.randint(0, self.item_count)
    
        #sometimes there will not be an image for given item i or j
        try: 
          self.image_features[i]
          self.image_features[j]
        except KeyError:
          continue  #if so, skip this item
  
        t.append([u, i, j])
        iv.append(self.image_features[i])
        jv.append(self.image_features[j])
        yield np.asarray(t), np.vstack(tuple(iv)), np.vstack(tuple(jv))
        
  def batch_generator_by_user(self, dataset, sample_size=3000, neg_sample_size=1000):
    # using leave one cv
    for u in random.sample(self.user_items.keys(), sample_size): #uniform random sampling w/o replacement
      t = []
      ilist = []
      jlist = []

      i = dataset[u]
      #check if we have an image for i, sometimes we dont...
      if i not in self.image_features:
        continue

      for _ in xrange(neg_sample_size):
          j = random.randint(1, self.item_count)
          if j != dataset[u] and not (j in self.user_items[u]):
              # find negative item not in train or test set

              #sometimes there will not be an image for given product
              try: 
                self.image_features[i]
                self.image_features[j]
              except KeyError:
                continue  #if image not found, skip item
        
              t.append([u, i, j])
              ilist.append(self.image_features[i])
              jlist.append(self.image_features[j])
            
      yield np.asarray(t), np.vstack(tuple(ilist)), np.vstack(tuple(jlist))
          
  def vbpr(self, user_count, item_count, hidden_dim=20, hidden_img_dim=128, 
           learning_rate=0.005,
            l2_regulization=0.1,
            bias_regulization=0.1,
            embed_regulization = 0.007,
            image_regulization = 0.007,
            visual_bias_regulization=0.007):
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
    visual_bias = tf.get_variable("visual_bias", [1, 4096], initializer=tf.constant_initializer(0.0))
  
    img_emb_w = tf.get_variable("image_embedding_weights", [4096, hidden_img_dim], 
                               initializer=tf.random_normal_initializer(0, 0.1))

    #learn nusersx20 + nusers*128 + nitems*20 + 4096 + 4096*128 parameters
    # params = 83779*20 + 83779*128 + 302047*20 + 4096 + 524288 = 18,968,616
  
    #lookup the latent factors by user and id
    u_emb = tf.nn.embedding_lookup(user_emb_w, u)
    u_img = tf.nn.embedding_lookup(user_img_w, u)
  
    i_emb = tf.nn.embedding_lookup(item_emb_w, i)
    i_b = tf.nn.embedding_lookup(item_b, i)
    j_emb = tf.nn.embedding_lookup(item_emb_w, j)
    j_b = tf.nn.embedding_lookup(item_b, j)
  

                             

    # MF predict: u_i > u_j
    theta_i = tf.matmul(iv, img_emb_w) # (f_i * E), eq. 3 1x4096 x 4086x128 => 1x128 #plot these on 2d scatter
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
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    print "Hyper-parameters: K=%d, K2=%d, lr=%f, l2r=%f"%(hidden_dim, hidden_img_dim, learning_rate, l2_regulization)
    return u, i, j, iv, jv, loss, auc, train_op
    
    
  def train(self, lr, lam, lam2, lambias, k, k2, epochs, sample_count, batch_size):
    print "generating test set..."
    self.session = tf.Session()
    history = []
    self.val_ratings, self.test_ratings = self.generate_val_test()
    
    print self.session
    u, i, j, iv, jv, loss, auc, train_op = self.vbpr(self.user_count, self.item_count, hidden_dim=k, hidden_img_dim=k2, learning_rate =lr, l2_regulization =lam)
    self.session.run(tf.global_variables_initializer())
    
    for epoch in range(1, epochs):
        print "epoch ", epoch
        sample_train_loss=[]
        for d, _iv, _jv in self.uniform_sample_batch(sample_count=sample_count, batch_size=batch_size ):
            _loss, _ = self.session.run([loss, train_op], feed_dict={ u:d[:,0], i:d[:,1], j:d[:,2], iv:_iv, jv:_jv})
            sample_train_loss.append(_loss)
        train_loss = np.mean(sample_train_loss)
        
        #test
        sample_val_loss=[]
        sample_val_auc=[]
        for d, fi, fj in self.batch_generator_by_user(dataset=self.val_ratings, sample_size=10):
            _loss, _auc = self.session.run([loss, auc], feed_dict={u:d[:,0], i:d[:,1], j:d[:,2], iv:fi, jv:fj})
            sample_val_loss.append(_loss)
            sample_val_auc.append(_auc)
        val_loss = np.mean(sample_val_loss)
        val_auc  = np.mean(sample_val_auc)
        
        history.append([epoch, train_loss, val_loss, val_auc])
        
    self.session.close()
    return history

if __name__ == '__main__':
  import os
  vbpr = Vbpr2(1)
  
  data_dir = os.path.join("data", "amzn")
  simple_path = os.path.join(data_dir, 'reviews_Women_5.txt')
  vbpr.load_dataset(simple_path)
  
  #train
  lr=0.005
  lam = 9.5
  lam2 = 0.0
  lambias = .001
  k1=10
  k2=10
  epochs=10
  sample_count=20
  batch_size=20
  history = vbpr.train(lr, lam, lam2, lambias,k1,k2, epochs, sample_count, batch_size)
  print history
  