#!/usr/bin/env sh



GLOG_logtostderr=1 /PATH_TO_CAFFE/tools/generate_combinedVirtual_expert_datafile_dynamic.bin \
    --numRealClasses=100 \
    --numMetaClasses=10 \
    --mainRealTrain=../data/main_real_train.txt \
    --mainRealVal=../data/main_real_test.txt \
    --mapping=../trained_models/mapping_generalist.txt \
    --virtualRealmapping=virtualRealmapping.txt \
	--realVirtualmapping=realVirtualmapping.txt \
    --outTrain=train_virtual.txt \
    --outVal=val_virtual.txt \
    --numRealClassesInExpert_file=numRealClassesInExpert_file.txt \2>&1 | tee outlog.log



