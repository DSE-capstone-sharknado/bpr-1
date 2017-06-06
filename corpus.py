from collections import defaultdict
import numpy as np
import random 

class Corpus(object):
  """docstring for Corpus"""
  def __init__(self):
    super(Corpus, self).__init__()
    
  # def load_data(self, path, user_min=0):
 #    #load raw from disk
 #    reviews=[]
 #    with open((path), 'r') as f:
 #      for line in f.readlines():
 #        auid, asin, _ = line.split(" ", 2)
 #        reviews.append([auid,asin])
 #
 #    #stats
 #    user_dist, item_dist, user_ratings, item_users = self.stats(reviews)
 #
 #    #filter out based on distribution of users
 #    reviews_reduced=[]
 #    for auid, asin in reviews:
 #      if user_dist[auid] >=user_min:
 #        reviews_reduced.append([auid, asin])
 #
 #    #map to sequential ids
 #    users = {}
 #    items = {}
 #    user_count=0
 #    item_count=0
 #    triples=[]
 #    for auid, asin in reviews_reduced:
 #      if auid in users:
 #        u = users[auid]
 #      else:
 #        user_count+=1 #new user so increment
 #        users[auid]=user_count
 #        u = user_count
 #
 #      if asin in items:
 #        i = items[asin]
 #      else:
 #        item_count+=1 #new user so increment
 #        items[asin]=item_count
 #        i = item_count
 #
 #      triples.append([u, i])
 #
 #
 #    self.users=users
 #    self.items=items
 #    self.reviews = np.array(triples)
 #
 #
 #    print "generating stats..."
 #    user_dist, item_dist, user_items, item_users  = self.stats(self.reviews)
 #
 #    self.user_dist = user_dist
 #    self.item_dist = item_dist
 #    self.user_items = user_items
 #    self.item_users = item_users
    
  @staticmethod
  def stats(reviews):
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
  
      
  @staticmethod
  def load_simple(path, user_min=5):
    #load raw from disk
    reviews=[]
    with open((path), 'r') as f:
      for line in f.readlines():
        auid, asin, _ = line.split(",", 2)
        reviews.append([auid,asin])
  
    #stats
    user_dist, item_dist, user_ratings, item_users = Corpus.stats(reviews)
  
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
  
    return users, items, np.array(triples)
    
  NORM_FACTOR = 58.388599

  #load image features for the given asin collection into dictionary
  @staticmethod
  def load_image_features(path, items):
    count=0
    image_features = {}
    f = open(path, 'rb')
    while True:
      asin = f.read(10)
      if asin == '': break
      features_bytes = f.read(16384) # 4 * 4096 = 16KB, fast read, don't unpack
  
      if asin in items: #only unpack 4096 bytes if w need it -- big speed up
        features = np.fromstring(features_bytes, dtype=np.float32)/Corpus.NORM_FACTOR
        iid=items[asin]
        image_features[iid] = features
  
      if count%50000==0:
        print count
      count+=1

    return image_features
    
  def load_reviews(self, path):
    print "Loading dataset from: ",path

    users, items, reviews_all = Corpus.load_simple(path, user_min=5)
    print "generating stats..."
    user_dist, item_dist, train_ratings, item_users = Corpus.stats(reviews_all)

    user_count = len(train_ratings)
    item_count = len(item_users)
    reviews_count = len(reviews_all)
    print user_count,item_count,reviews_count
  
    # return users, items, reviews_all,user_dist, item_dist, train_ratings, item_users
    
    self.users=users
    self.items=items
    self.reviews = reviews_all
    
    
    self.user_dist = user_dist
    self.item_dist = item_dist
    self.user_items = train_ratings
    self.item_users = item_users
    
    self.user_count = len(self.user_items)
    self.item_count = len(self.item_users)

  def load_images(self, path, items):
    print "Loading image features from: ",path
    self.image_features = Corpus.load_image_features(path, items)

    print "extracted image feature count: ",len(self.image_features)
  
  def load_data(self, reviews_path, images_path, user_min, item_min):
    #load reviews
    self.load_reviews(reviews_path)
    if images_path:
      self.load_images(images_path, self.items)
    
if __name__ == '__main__':
  import os
  data_dir = os.path.join("data", "amzn")
  simple_path = os.path.join(data_dir, 'reviews_Women_5.txt')

  corpus = Corpus()
  corpus.load_data(simple_path)