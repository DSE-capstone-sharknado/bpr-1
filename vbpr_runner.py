#multiprocess grid search

from time import gmtime, strftime
from multiprocessing import Process, Queue
from random import randint
from time import sleep
import os
import vbpr2

lam_list = [0, .001, .01, 1, 10] #log scale
lam2_list = [0, .001, .01, 1, 10]
lambias_list = [0, .001, .01, 1, 10]
#learning rate?
lr=.005




def train_eval_log(vbpr, lr, lam, lam2, lambias, combination):
  k1=10
  k2=10
  epochs=11
  sample_count=400
  batch_size=512
  
  print "Training combination: ",combination
  history = vbpr.train(lr, lam, lam2, lambias,k1,k2, epochs, sample_count, batch_size)
  ts=strftime("%Y-%m-%d %H:%M:%S", gmtime())

  hashstr = "lr=%.2f,lam=%.2f,lam2=%.2f,lambias=%.2f,k1=%d,k2=%d,epochs=%d,sample_count=%d,batch_size=%d"%(lr, lam, lam2, lambias,k1,k2, epochs, sample_count, batch_size)
  log_path = "logs/"+hashstr+"_"+ts+".log"
  log(history, log_path)
  print "finished combination: ",combination

def log(history, path):
  with open((path), 'a') as f:
    for epoch, train_loss, val_loss, val_auc in history:
      fstr="%d, %.2f, %.2f, %.2f"%(epoch, train_loss, val_loss, val_auc)
      f.write(fstr+"\n")
  
vbpr = vbpr2.Vbpr2(1)

data_dir = os.path.join("data", "amzn")
simple_path = os.path.join(data_dir, 'reviews_Women_5.txt')
vbpr.load_dataset(simple_path)

processes = []
combinations=0

for lam in lam_list:
  for lam2 in lam2_list:
    for lambias in lambias_list:
      #start new process
      p = Process(target=train_eval_log, args=(vbpr, lr, lam, lam2, lambias, combinations))
      processes.append(p)
      p.start()
      combinations+=1
      
[p.join() for p in processes]

print "DONE"