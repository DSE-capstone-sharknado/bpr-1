from collections import defaultdict
import numpy as np

class Corpus(object):
  """docstring for Corpus"""
  def __init__(self):
    super(Corpus, self).__init__()
    
  def load_data(self, path, user_min=0):
    #load raw from disk
    reviews=[]
    with open((path), 'r') as f:
      for line in f.readlines():
        auid, asin, _ = line.split(" ", 2)
        reviews.append([auid,asin])
  
    #stats
    user_dist, item_dist, user_ratings, item_users = self.stats(reviews)
  
    #filter out based on distribution of users
    reviews_reduced=[]
    for auid, asin in reviews:
      if user_dist[auid] >=user_min:
        reviews_reduced.append([auid, asin])
  
    #map to sequential ids
    users = {}
    items = {}
    user_count=0
    item_count=0
    triples=[]    
    for auid, asin in reviews_reduced:
      if auid in users:
        u = users[auid]
      else:
        user_count+=1 #new user so increment
        users[auid]=user_count
        u = user_count
      
      if asin in items:
        i = items[asin]
      else:
        item_count+=1 #new user so increment
        items[asin]=item_count
        i = item_count
     
      triples.append([u, i])
    
    
    self.users=users
    self.items=items
    self.reviews = np.array(triples)
    
    
    print "generating stats..."
    user_dist, item_dist, user_items, item_users  = self.stats(self.reviews)
    
    self.user_dist = user_dist
    self.item_dist = item_dist
    self.user_items = user_items
    self.item_users = item_users
    
  def stats(self, reviews):
    #frequency distributions
    user_dist=defaultdict(int)
    item_dist=defaultdict(int)
  
    #cores
    user_items = defaultdict(list)
    item_users = defaultdict(list)
  
    for u,i in reviews:
       user_dist[u]+=1
       item_dist[i]+=1
       user_items[u].append(i)
       item_users[i].append(u)
     
    return user_dist, item_dist, user_items, item_users
    
  def generate_val_and_test(self, user_items):
      '''
      for each user, random select one rating into test set
      '''
      user_test = dict()
      user_val = dict()
      for u, i_list in user_items.iteritems():
          samples = random.sample(i_list, 2)
          user_test[u] = samples[0]
          user_val[u] = samples[1]
      return user_val, user_test
    
    
if __name__ == '__main__':
  import os
  data_dir = os.path.join("data", "amzn")
  simple_path = os.path.join(data_dir, 'reviews_Women_5.txt')

  corpus = Corpus()
  corpus.load_data(simple_path)