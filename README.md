# tf-bpr

BPR implemented in Tensorflow

Bayesian Personalized Ranking(BPR) is a learning algorithm for collaborative filtering first introduced in: BPR: Bayesian Personalized Ranking from Implicit Feedback. Steffen Rendle, Christoph Freudenthaler, Zeno Gantner and Lars Schmidt-Thieme, Proc. UAI 2009.   


gcloud beta ml init-project


## Amzn model

JOB_NAME=tfbpr_amzn_v9
TRAIN_BUCKET=gs://ml-engine-sharknado
TRAIN_PATH=${TRAIN_BUCKET}/${JOB_NAME}


gcloud beta ml jobs submit training amzn_gpu_east_1 \
--package-path=trainer \
--module-name=trainer.bpr \
--staging-bucket="${TRAIN_BUCKET}" \
--region=us-east1 \
--scale-tier=BASIC_GPU



## Movie lens model

JOB_NAME=tfbpr_movielens_v5gpu
TRAIN_BUCKET=gs://ml-engine-sharknado

personal
TRAIN_BUCKET=gs://tf-sharknado-ml
TRAIN_PATH=${TRAIN_BUCKET}/${JOB_NAME}
    
gcloud beta ml jobs submit training ml_gpu_east_6 \
--package-path=movielens \
--module-name=movielens.bpr \
--staging-bucket="${TRAIN_BUCKET}" \
--region=us-east1 \
--scale-tier=BASIC_GPU