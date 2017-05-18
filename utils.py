#Amazon specific utils

from collections import defaultdict
import os
import struct
import numpy as np
import random

# returns: uid->auid dict, iid->asin dict, reviews count, items by user dict
def load_data_simple(path, min_items=5):
    user_reviews = defaultdict(list)

    with open((path), 'r') as f:
        for line in f.readlines():
            auid, asin, _ = line.split(" ", 2)
            user_reviews[auid].append(asin)

    # filter out by min_items
    reviews_count = 0
    user_reviews_filtered = defaultdict(list)
    for auid, asins in user_reviews.iteritems():
        if len(asins) >= min_items:  # keep
            user_reviews_filtered[auid] = asins
            reviews_count += len(asins)

    # now build auid,asin -> internal id LUT
    users_lut = {}
    items_lut = {}
    user_count = 0
    item_count = 0
    asins_filtered = set()
    for auid, asins in user_reviews_filtered.iteritems():

        if auid in users_lut:
            u = users_lut[auid]
        else:
            user_count += 1  # new user so increment
            users_lut[auid] = user_count
            u = user_count

        for asin in asins:
            if asin in items_lut:
                i = items_lut[asin]
            else:
                item_count += 1  # new i so increment
                items_lut[asin] = item_count
                i = item_count

    # now update all the keys to use internal id
    user_reviews_filtered_keyed = defaultdict(list)
    for auid, asins in user_reviews_filtered.iteritems():
        internal_asins = map(lambda asin: items_lut[asin], asins)
        internal_key = users_lut[auid]
        user_reviews_filtered_keyed[internal_key] = internal_asins

    return users_lut, items_lut, reviews_count, user_reviews_filtered_keyed



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
def load_data_hybrid(data_path, min_items=1, min_users=1, sampling= True, sample_size = 0.5):
    user_ratings = defaultdict(set)
    item_ratings = defaultdict(set)
    max_u_id = -1
    max_i_id = -1
    user_count = 0
    item_count = 0
    reviews = 0
    users = {}  # aid to id LUT
    items = {}  # asid to id LUT
    brands = {}
    prices = {}
    prod_desc = {}
    random.seed(0)
    with open(data_path, 'r') as f:
        for line in f.readlines()[1:]:
            if (sampling and random.random()>sample_size):
                continue
            reviews += 1
            if (len(line.split(","))==6):
                auid, asid, _, brand, price, product_desc = line.split(",")
            else:
                auid, asid, _, brand, price = line.split(",")

            u, i = None, None

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
                if (price=='' or price=='\r\n' or price=='\n'):
                    prices[i] = 0
                else:
                    prices[i] = float(price.rstrip())
                if (len(line.split(",")) == 6):
                    prod_desc[i] = [int(el) for el in list(product_desc)[:-2][1:]]
                    if (len(prod_desc[i])==0):
                        prod_desc[i] = list(np.zeros(4525))

            user_ratings[u].add(i)
            item_ratings[i].add(u)
            max_u_id = max(u, max_u_id)
            max_i_id = max(i, max_i_id)

    print "max_u_id: ", max_u_id
    print "max_i_id: ", max_i_id
    print "reviews : ", reviews

    # filter out users w/ less than X reviews
    num_u_id = 0
    num_i_id = 0
    num_reviews = 0
    user_ratings_filtered = defaultdict(set)
    for u, ids in user_ratings.iteritems():
        if len(ids) > min_items:
            # keep
            user_ratings_filtered[u] = ids
            num_u_id += 1
            num_reviews += len(ids)
    item_ratings_filtered = defaultdict(set)
    for ids, u in item_ratings.iteritems():
        if len(u) > min_users:
            # keep
            item_ratings_filtered[ids] = u
            num_i_id += 1


    print "u_id: ", num_u_id
    print "i_id: ", num_i_id
    print "reviews : ", num_reviews
    return max_u_id, max_i_id, users, items, user_ratings_filtered, item_ratings_filtered, brands, prices, prod_desc

#  TODO add function to seralize output of load_data to a picklefile


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
    
