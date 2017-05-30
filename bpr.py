import numpy
import tensorflow as tf
import os
import random
import time
import numpy as np
from utils import load_image_features, load_simple, stats
import corpus
from datetime import datetime
import model
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

class Bpr(model.Model):
  def __init__(self, corpus, K, reg_rate, bias_reg, max_epochs, session):
    model.Model.__init__(self, corpus)
    print "starting BPR model..."
    
    self.K=K
    self.reg_rate = reg_rate
    self.bias_reg = bias_reg
    self.max_epochs = max_epochs
    self.session = session
  
  @classmethod
  def restore(cls, session):
    user_count=1
    item_count=1
    hidden_dim=1
    user_emb_w = tf.get_variable("user_emb_w", [user_count+1, hidden_dim], initializer=tf.random_normal_initializer(0, 0.1))
    item_emb_w = tf.get_variable("item_emb_w", [item_count+1, hidden_dim], initializer=tf.random_normal_initializer(0, 0.1))
    item_b = tf.get_variable("item_b", [item_count+1, 1], initializer=tf.random_normal_initializer(0, 0.1))    #item bias
    user_b = tf.get_variable("user_b", [user_count+1, 1], initializer=tf.random_normal_initializer(0, 0.1))    #user bias
    saver = tf.train.Saver()
    saver.restore(session, "logs/")
    return cls(None, -1, -1, -1, -1, session)
  
  def train(self, batch_size, sample_count, learning_rate=0.05):
    model_name="BPR-K=%d_l1=%.2f_l2=%.2f"%(self.K, self.reg_rate, self.bias_reg)
    print model_name
    
    user_count = len(self.corpus.user_items)
    item_count = len(self.corpus.item_users)
    
    
    print "training..."
    self.u, self.i, self.j, self.mf_auc, self.bprloss, train_op = self.bpr_mf(user_count, item_count, self.K, lr=learning_rate, regulation_rate=self.reg_rate, bias_reg=self.bias_reg)
    self.merged = tf.summary.merge_all()
    self.train_writer = tf.summary.FileWriter('logs/train/run-%s'%now, self.session.graph)
    self.test_writer  = tf.summary.FileWriter('logs/test/run-%s'%now, self.session.graph)
    saver = tf.train.Saver()
    self.session.run(tf.global_variables_initializer())
    
    epoch_durations = []
    best_auc=-1
    best_iter=-1
    for epoch in range(1, self.max_epochs+1):
        epoch_start_time = time.time()
        train_loss_vals=[]
        
        for batch in self.generate_train_batch(batch_size=batch_size, sample_count=sample_count):
          _bprloss, _ = self.session.run([self.bprloss, train_op], feed_dict={self.u:batch[:,0], self.i:batch[:,1], self.j:batch[:,2]})
          train_loss_vals.append(_bprloss)
        
        epoch_durations.append(time.time() - epoch_start_time)
          
        #val loss
        val_auc, val_loss, _ = self.eval(self.val_ratings)
        
        print "epoch: %d (%.2fs), \ttrain loss: %.2f, \tval loss: %.2f, \tval auc: %.2f"%(epoch, np.mean(epoch_durations), np.mean(train_loss_vals) , np.mean(val_loss), val_auc )
        
        #early termination/checks for convergance
        if val_auc > best_auc:
          best_auc = val_auc
          best_iter = epoch
          print "*"
          saver.save(self.session, "logs/")
        elif val_auc < best_iter and epoch >= best_iter+15: #overfitting
          print "Overfitted. Exiting..."
          break
    
    
    #restore best model from checkpoint
    saver.restore(self.session, "logs/")
    #test auc
    test_auc, test_loss, _  = self.eval(self.test_ratings, user_size=500)
    print "test auc: ",test_auc 
    
    
    #coldstart auc
    cold_auc, cold_loss, _ = self.eval(self.test_ratings, cold_start=True, user_size=500)
    print "cold auc: ", cold_auc
    
    self.session.close()
  
  
  def bpr_mf(self, user_count, item_count, hidden_dim, lr=0.1, regulation_rate = 0.0001, bias_reg=.01):
      
      #model input
      u = tf.placeholder(tf.int32, [None])
      i = tf.placeholder(tf.int32, [None])
      j = tf.placeholder(tf.int32, [None])
      
      #model paramenters
      #latent factors
      #hidden_dim is the k hyper parameter
      user_emb_w = tf.get_variable("user_emb_w", [user_count+1, hidden_dim], initializer=tf.random_normal_initializer(0, 0.1))
      item_emb_w = tf.get_variable("item_emb_w", [item_count+1, hidden_dim], initializer=tf.random_normal_initializer(0, 0.1))
      item_b = tf.get_variable("item_b", [item_count+1, 1], initializer=tf.random_normal_initializer(0, 0.1))    #item bias
      user_b = tf.get_variable("user_b", [user_count+1, 1], initializer=tf.random_normal_initializer(0, 0.1))    #user bias
      
      u_emb = tf.nn.embedding_lookup(user_emb_w, u) #lookup the latent factor for user u
      i_emb = tf.nn.embedding_lookup(item_emb_w, i) #lookup the latent factor fo item i
      j_emb = tf.nn.embedding_lookup(item_emb_w, j) #lookup the latent factor for item j
      i_b = tf.nn.embedding_lookup(item_b, i)       #lookup the bias vector for item i
      j_b = tf.nn.embedding_lookup(item_b, j)       #lookup the bias vector for item js
      
      
      # MF predict: u_i > u_j
      # xuij = xui - xuj
      xui = i_b + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1, keep_dims=True)
      xuj = j_b + tf.reduce_sum(tf.multiply(u_emb, j_emb), 1, keep_dims=True)
      xuij = xui-xuj
      
      # AUC for one user:
      # reasonable iff all (u,i,j) pairs are from the same user
      #
      # average AUC = mean( auc for each user in test set)
      mf_auc = tf.reduce_mean(tf.to_float(xuij > 0)) # xui - xui > 0 == xui > xuj
      tf.summary.scalar('user auc', mf_auc)
      
      l2_norm = tf.add_n([
              regulation_rate * tf.reduce_sum(tf.multiply(u_emb, u_emb)),
              regulation_rate * tf.reduce_sum(tf.multiply(i_emb, i_emb)),
              regulation_rate * tf.reduce_sum(tf.multiply(j_emb, j_emb)),
              #reg for biases
              bias_reg * tf.reduce_sum(tf.multiply(i_b, i_b)),
              bias_reg/10.0 * tf.reduce_sum(tf.multiply(j_b, j_b)),
          ])
      
      
      bprloss =  l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(xuij))) #BPR loss
      
      global_step = tf.Variable(0, trainable=False)
      learning_rate = tf.train.exponential_decay(lr, global_step, 400, 0.8, staircase=True)
      #.1 ... .001
      
      #optimizer updates 62 parameters
      # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(bprloss, global_step=global_step)
      train_op =  tf.train.AdamOptimizer().minimize(bprloss, global_step=global_step)
      return u, i, j, mf_auc, bprloss, train_op
  
  
  def eval(self, dataset, user_size=300, cold_start=False):
      '''
      dataset is a dictionary keyed by user where value is item id. This is either the val or test set to which we want to eval.
      '''
      user_count=0
      loss_vals=[]
      auc_vals=[]
      for t_uij in self.generate_test_batch(dataset, users_size=user_size, cold_start=cold_start):
          _loss, user_auc, summary = self.session.run([self.bprloss, self.mf_auc, self.merged], feed_dict={self.u:t_uij[:,0], self.i:t_uij[:,1], self.j:t_uij[:,2]})
          loss_vals.append(_loss)
          auc_vals.append(user_auc)
          user_count+=1
          self.train_writer.add_summary(summary, user_count)
      
      auc = np.mean(auc_vals)
      loss = np.mean(loss_vals)
      return auc, loss, user_count


if __name__ == '__main__':
  pass


