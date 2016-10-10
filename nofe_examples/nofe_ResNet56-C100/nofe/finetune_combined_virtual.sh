#!/usr/bin/env sh  
echo "..." 

## Important: Run on 4 gpus or change batch size 

## (PART -1 ) start from finetuning of genralist 
GLOG_logtostderr=1 /PATH_TO/caffe.bin train \
    --solver=solver_combinedVirtual.prototxt \
    --weights=../trained_models/caffe_generalist.caffemodel   2>&1 | tee combined_virtual.log


# (PART -2 )
GLOG_logtostderr=1 /PATH_TO/caffe.bin train \
    --solver=solver_combinedVirtual_lr1.prototxt \
    --snapshot=snapshots/caffe_combined_train_iter_15000.solverstate   2>&1 | tee combined_virtual_lr1.log



# (PART -3 )
GLOG_logtostderr=1 /PATH_TO/caffe.bin train \
    --solver=solver_combinedVirtual_lr2.prototxt \
    --snapshot=snapshots/caffe_combined_train_iter_225000.solverstate   2>&1 | tee combined_virtual_lr2.log

# (PART -4 )
#GLOG_logtostderr=1 /PATH_TO/caffe.bin train \
#    --solver=solver_combinedVirtual_lr3.prototxt \
#    --snapshot=snapshots/caffe_combined_train_iter_385000.solverstate   2>&1 | tee combined_virtual_lr3.log

