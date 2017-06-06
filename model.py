import random
import numpy as np
import tensorflow as tf

class Model(object):
  
  def __init__(self, corpus, session):
    self.corpus = corpus
    self.session = session
    
    self.session.run(tf.global_variables_initializer())
    
    self.saver = tf.train.Saver()
    
    self.val_ratings, self.test_ratings = self.generate_val_and_test()
    
    
  def generate_val_and_test(self):
      '''
      for each user, random select one rating into test set
      '''
      user_test = dict()
      user_val = dict()
      for u, i_list in self.corpus.user_items.iteritems():
          samples = random.sample(i_list, 2)
          user_test[u] = samples[0]
          user_val[u] = samples[1]
      return user_val, user_test
      
  def save(self):
    self.saver.save(self.session, "logs/")
  
  def restore(self):
    self.saver.restore(self.session, "logs/")
      
  def train(self):
    raise Exception("Not implemented yet!")
    
  def evaluate(self):
    raise Exception("Not implemented yet!")
    
    
    
if __name__ == '__main__':
  import os
  import corpus
  print "Loading dataset..."
  data_dir = os.path.join("data", "amzn")
  simple_path = os.path.join(data_dir, 'reviews_Women_5.txt')


  