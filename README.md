# tf-bpr

BPR implemented in Tensorflow

Bayesian Personalized Ranking(BPR) is a learning algorithm for collaborative filtering first introduced in: BPR: Bayesian Personalized Ranking from Implicit Feedback. Steffen Rendle, Christoph Freudenthaler, Zeno Gantner and Lars Schmidt-Thieme, Proc. UAI 2009.   





## Amzn model

Set `staging-bucket` to your GS bucket:

gcloud beta ml jobs submit training amzn_gpu_east_3 \
--package-path=amzn \
--module-name=amzn.bpr \
--staging-bucket="gs://tf-sharknado-ml" \
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