

echo ".."

GLOG_logtostderr=1 PATH_TO_CAFFE/caffe.bin train \
    --solver=generalist_solver.prototxt   2>&1 | tee train.log 
