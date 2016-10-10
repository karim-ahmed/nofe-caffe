
echo "..." 

## Important: run on 4 gpus, if not update the batch size
## (PART -1 ) start from finetuning of genralist 
GLOG_logtostderr=1 /PATH_TO_CAFFE/tools/caffe.bin train -gpu=0,1,2,3 \
   --solver=solver_combinedVirtual.prototxt \
    --weights=caffe_generalist_train_iter_92500.caffemodel   2>&1 | tee combined_virtual.log



GLOG_logtostderr=1 /PATH_TO_CAFFE/tools/caffe.bin train -gpu=0,1,2,3 \
    --solver=solver_combinedVirtual_lr1.prototxt \
    --snapshot=snapshots/caffe_combined_train_iter_49000.solverstate   2>&1 | tee combined_virtual_lr1.log




GLOG_logtostderr=1 /PATH_TO_CAFFE/tools/caffe.bin train -gpu=0,1,2,3 \
    --solver=solver_combinedVirtual_lr2.prototxt \
    --snapshot=snapshots_lr1/caffe_combined_train_iter_60000.solverstate   2>&1 | tee combined_virtual_lr2.log



GLOG_logtostderr=1 /PATH_TO_CAFFE/tools/caffe.bin train -gpu=0,1,2,3 \
    --solver=solver_combinedVirtual_lr3.prototxt \
    --snapshot=snapshots_lr2/caffe_combined_train_iter_65000.solverstate   2>&1 | tee combined_virtual_lr3.log


