## train Generalist
# change to your build caffe path

GLOG_logtostderr=1 /path_to/caffe.bin train  \
    --solver=generalist_solver.prototxt   2>&1 | tee train.log 
