#!/usr/bin/env sh  
echo "..." 

## (PART -1 ) start from finetuning of genralist 
GLOG_logtostderr=1 /PATH_TO/caffe.bin train \
    --solver=solver_combinedVirtual.prototxt \
    --weights=../trained_models/caffe_generalist.caffemodel   2>&1 | tee combined_virtual.log


# (PART -2 )
GLOG_logtostderr=1 /PATH_TO/caffe.bin train \
    --solver=solver_combinedVirtual_lr1.prototxt \
    --snapshot=snapshots/caffe_combined_train_iter_120000.solverstate   2>&1 | tee combined_virtual_lr1.log



# (PART -3 )
GLOG_logtostderr=1 /PATH_TO/caffe.bin train \
    --solver=solver_combinedVirtual_lr2.prototxt \
    --snapshot=snapshots/caffe_combined_train_iter_130000.solverstate   2>&1 | tee combined_virtual_lr2.log

