import random
import numpy as np


class Model(object):
  
  def __init__(self, corpus):
    self.corpus=corpus
    self.generate_val_and_test(corpus.user_items)

    
  def generate_val_and_test(self, user_ratings):
      '''
      for each user, random select one rating into test set
      '''
      user_test = dict()
      user_val = dict()
      for u, i_list in user_ratings.iteritems():
          samples = random.sample(i_list, 2)
          user_test[u] = samples[0]
          user_val[u] = samples[1]
      
      self.val_ratings = user_val
      self.test_ratings = user_test



  def generate_train_batch(self, batch_size=512, sample_count=400):
      '''
      uniform sampling (user, item_rated, item_not_rated)
      '''
      for _ in xrange(sample_count):
        t = []
        for b in xrange(batch_size):
            u = random.randint(1, len(self.corpus.user_items)) #random user u
            i = random.sample(self.corpus.user_items[u], 1)[0] #random item i from u
            while i == self.test_ratings[u] or i==self.val_ratings[u]: #make sure it's not in the test set
                i = random.sample(self.corpus.user_items[u], 1)[0]
            
            j = random.randint(1, len(self.corpus.items)) #random item j
            while j in self.corpus.user_items[u]: #make sure it's not in u's items
                j = random.randint(1, len(self.corpus.items))
            
            t.append([u, i, j])
        #return a batch
        yield np.asarray(t)

  #sample users but not items
  def generate_test_batch(self, user_item_dict, users_size=512, neg_sample_size=1000, cold_start=False):
      '''
      for an user u and an item i rated by u, 
      generate pairs (u,i,j) for all item j which u has't rated
      it's convenient for computing AUC score for u
      '''
    
      #emits a batch of rankings for each user
      #either iterate all users or sample
      for u in random.sample(user_item_dict, users_size): #uniform random sampling w/o replacement
          t = []
          i = user_item_dict[u]
        
          #filter for cold start
          if cold_start and self.corpus.item_dist[i] > 5:
            continue
        
          for _ in xrange(neg_sample_size):
            j = random.randint(1, len(self.corpus.items))
            if j != i and j not in self.corpus.user_items[u]: #not in training set
              t.append([u, i, j])

          yield np.asarray(t) #returns a batch per user
    
if __name__ == '__main__':
  import os
  import corpus
  print "Loading dataset..."
  data_dir = os.path.join("data", "amzn")
  simple_path = os.path.join(data_dir, 'reviews_Women_5.txt')

  corpus = corpus.Corpus()
  corpus.load_data(simple_path, user_min=5)
  model = Model(corpus)
  