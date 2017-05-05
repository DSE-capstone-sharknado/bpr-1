#Amazon specific utils

from collections import defaultdict
import os
import struct
import numpy as np

#load data from amazon reviews dataset csv
def load_data(data_path):
    user_ratings = defaultdict(set)
    max_u_id = -1
    max_i_id = -1
    user_min=5
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

            if asid in items:
              i = items[asid]
            else:
              item_count+=1 #new i so increment
              items[asid]=item_count
              i=item_count

            user_ratings[u].add(i)
            max_u_id = max(u, max_u_id)
            max_i_id = max(i, max_i_id)

    print "max_u_id: ", max_u_id
    print "max_i_id: ", max_i_id
    print "reviews : ", reviews

    #filter out users w/ less than X reviews
    user_ratings_filtered = defaultdict(set)
    for u,ids in user_ratings.iteritems():
      if len(ids)>1:
        #keep
        user_ratings_filtered[u]=ids

    return max_u_id, max_i_id, users, items, user_ratings_filtered


# load data from amazon reviews dataset csv
def load_data_hybrid(data_path):
    data_path = os.path.join('', 'review_Women.csv')
    user_ratings = defaultdict(set)
    max_u_id = -1
    max_i_id = -1
    user_min = 5
    user_count = 0
    item_count = 0
    reviews = 0
    users = {}  # aid to id LUT
    items = {}  # asid to id LUT
    brands = {}
    prices = {}
    with open(data_path, 'r') as f:
        for line in f.readlines():
            reviews += 1
            try:
                auid, asid, _, brand, price = line.split(",")
                u, i = None, None
            except(ValueError):
                pass

            if auid in users:
                u = users[auid]
            else:
                user_count += 1  # new user so increment
                users[auid] = user_count
                u = user_count

            if asid in items:
                i = items[asid]
            else:
                item_count += 1  # new i so increment
                items[asid] = item_count
                i = item_count
                brands[i] = brand
                prices[i] = price.rstrip()

            user_ratings[u].add(i)
            max_u_id = max(u, max_u_id)
            max_i_id = max(i, max_i_id)

    print "max_u_id: ", max_u_id
    print "max_i_id: ", max_i_id
    print "reviews : ", reviews

    # filter out users w/ less than X reviews
    user_ratings_filtered = defaultdict(set)
    for u, ids in user_ratings.iteritems():
        if len(ids) > 1:
            # keep
            user_ratings_filtered[u] = ids

    return max_u_id, max_i_id, users, items, user_ratings_filtered, brands, prices

# TODO add function to seralize output of load_data to a picklefile


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
      features =[]
      for i in range(0, len(features_bytes), 4): #iterate 4 str bytes at aa time and unpack into 4 byte python float obj
        features.append(struct.unpack('f', features_bytes[i:i+4])[0])
      iid=items[asin]
      image_features[iid] = features
  
    if count%50000==0:
      print count
    count+=1
    
  return image_features
  
  
if __name__ == '__main__':
  data_path = os.path.join('data', 'amzn', 'review_Women.csv')
  user_count, item_count, users, items, user_ratings = load_data(data_path)

  #items: asin -> iid
  print "user count: ",len(users)
  print "item count:",len(items)
  
  images_path = "data/amzn/image_features_Women.b"
  image_features = load_image_features(images_path, items)    
    
  print "extracted image feature count: ",len(image_features)
  
  for asin, iid in items.iteritems():
    # print asin, iid
    image_features[iid]
    
