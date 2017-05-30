import os
import bpr
import corpus
import tensorflow as tf

print "Loading dataset..."
data_dir = os.path.join("data", "amzn")
simple_path = os.path.join(data_dir, 'reviews_Women.txt')

corpus = corpus.Corpus()
corpus.load_data(simple_path, user_min=3)

user_count = len(corpus.user_items)
item_count = len(corpus.item_users)
reviews_count = len(corpus.reviews)
print user_count,item_count,reviews_count
#83779 302047 711745

batch_size=128
sample_count = 400
max_epochs=100
K=20
learning_rate=0.0
reg_rate = 10.0
bias_reg = 0.01

session = tf.Session()

model = bpr.Bpr(corpus, K, reg_rate, bias_reg, max_epochs, session)
model.train(batch_size, sample_count)


# 229696 408160 1196785
# BPR-K=20_l1=10.00_l2=0.01
# test auc:  0.686584
# cold auc:  0.438954

# test auc:  0.71856
# cold auc:  0.431704