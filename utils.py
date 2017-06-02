#Amazon specific utils

from collections import defaultdict
import os
import struct
import numpy as np

def stats(triples):
  #frequency distributions
  user_dist=defaultdict(int)
  item_dist=defaultdict(int)
  
  #cores
  user_items = defaultdict(list)
  item_users = defaultdict(list)
  
  for u,i in triples:
     user_dist[u]+=1
     item_dist[i]+=1
     user_items[u].append(i)
     item_users[i].append(u)
  
  return user_dist, item_dist, user_items, item_users


def load_simple(path, user_min=5):
  #load raw from disk
  reviews=[]
  with open((path), 'r') as f:
    for line in f.readlines():
      auid, asin, _ = line.split(",", 2)
      reviews.append([auid,asin])
  
  #stats
  user_dist, item_dist, user_ratings, item_users = stats(reviews)
  
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
def load_image_features(path, items):
  count=0
  image_features = {}
  f = open(path, 'rb')
  while True:
    asin = f.read(10)
    if asin == '': break
    features_bytes = f.read(16384) # 4 * 4096 = 16KB, fast read, don't unpack
    
    if asin in items: #only unpack 4096 bytes if w need it -- big speed up
      features = np.fromstring(features_bytes, dtype=np.float32)/NORM_FACTOR
      iid=items[asin]
      image_features[iid] = features
    
    if count%50000==0:
      print count
    count+=1
  
  return image_features

import cPickle as pickle
def load_image_features_from_pickle(path):
  image_features = pickle.load( open( path, "rb" ) )
  return image_features

def load_and_save_image_features(path, items):
  image_features = load_image_features(path, items)
  pickle.dump(image_features, open( "women_5_image_features.pkl", "wb" ),  protocol=pickle.HIGHEST_PROTOCOL )
  return image_features
  

if __name__ == '__main__':
  
  print "loading raw..."
  simple_path = os.path.join('data', 'amzn', 'reviews_Women_ALL_scraped.csv')
  users_lut, items_lut, reviews = load_simple(simple_path, user_min=5)
  print "generating stats..."
  user_dist, item_dist, user_items, item_users = stats(reviews)
  print len(users_lut),len(items_lut),len(reviews)
  
  counts = []
  for user, items in user_items.iteritems():
    counts.append(len(items))
  
  print np.mean(counts)
  import collections
  counter=collections.Counter(counts)
  print counter
  
  print np.mean(counts)
  print len(counts)

  
  
  
  images_path = "data/amzn/image_features_Women.b"
  image_features = load_image_features(images_path, items_lut) #51 s
  # image_features = load_image_features_from_pickle("data/amzn/women_5_image_features.pkl") #48s
  print len(image_features)
  
  #now cross validate the image features w/ asin B000FIPV42 and the output I got from my C++ test
  iid = items_lut["B000FIPV42"]
  print image_features[iid]
  
  #old way takes 38Gb
    
    
