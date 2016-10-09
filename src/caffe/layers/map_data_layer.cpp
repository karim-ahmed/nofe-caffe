/**
 * Map Data Layer.
 * @author: Karim Ahmed (karim@cs.dartmouth.edu)
 * Publication: "Network of Experts for large-scale Image Categorization", ECCV 2016
 */

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template<typename Dtype>
MapDataLayer<Dtype>::MapDataLayer(const LayerParameter& param) :
		BasePrefetchingDataLayer<Dtype>(param), reader_(param) {
}

template<typename Dtype>
MapDataLayer<Dtype>::~MapDataLayer() {
	this->StopInternalThread();
}

template<typename Dtype>
void MapDataLayer<Dtype>::setLabelsMap(vector<int> new_labels_map) {

	CHECK(labels_map_.size() == new_labels_map.size())
			<< "ReloadMapLabels: Error[Size of new labels_map not equal to old one.]";

	labels_map_ = new_labels_map;
}

template<typename Dtype>
vector<int> MapDataLayer<Dtype>::getLabelsMap() {
	return labels_map_;
}

// label_ : Real Label
// meta_label_ : Meta Label
// See Batch class in data_layers.hpp
template<typename Dtype>
void MapDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	labels_map_file_ = this->layer_param_.data_param().labels_map_file();

	std::ifstream mapping_file(labels_map_file_.c_str());

	string class_id_str;
	while (mapping_file >> class_id_str) {
		labels_map_.push_back(atoi(class_id_str.c_str()));
	}
	mapping_file.close();

	const int batch_size = this->layer_param_.data_param().batch_size();
	// Read a data point, and use it to initialize the top blob.
	Datum& datum = *(reader_.full().peek());

	// Use data_transformer to infer the expected blob shape from datum.
	vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
	this->transformed_data_.Reshape(top_shape);
	// Reshape top[0] and prefetch_data according to the batch_size.
	top_shape[0] = batch_size;
	top[0]->Reshape(top_shape);
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		this->prefetch_[i].data_.Reshape(top_shape);
	}
	LOG(INFO) << "output data size: " << top[0]->num() << ","
			<< top[0]->channels() << "," << top[0]->height() << ","
			<< top[0]->width();
	// label
	if (this->output_labels_) {
		vector<int> label_shape(1, batch_size);
		vector<int> meta_label_shape(1, batch_size);
		top[1]->Reshape(label_shape); // for label
		top[2]->Reshape(meta_label_shape); // for meta_label
		for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			this->prefetch_[i].label_.Reshape(label_shape);
			this->prefetch_[i].meta_label_.Reshape(meta_label_shape);
		}
	}
}

// This function is called on prefetch thread
template<typename Dtype>
void MapDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {

	CPUTimer batch_timer;
	batch_timer.Start();
	double read_time = 0;
	double trans_time = 0;
	CPUTimer timer;
	CHECK(batch->data_.count());
	CHECK(this->transformed_data_.count());

	// Reshape according to the first datum of each batch
	// on single input batches allows for inputs of varying dimension.
	const int batch_size = this->layer_param_.data_param().batch_size();
	Datum& datum = *(reader_.full().peek());
	// Use data_transformer to infer the expected blob shape from datum.
	vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
	this->transformed_data_.Reshape(top_shape);
	// Reshape batch according to the batch_size.
	top_shape[0] = batch_size;
	batch->data_.Reshape(top_shape);

	Dtype* top_data = batch->data_.mutable_cpu_data();
	Dtype* top_label = NULL;
	Dtype* top_meta_label = NULL;

	if (this->output_labels_) {
		top_label = batch->label_.mutable_cpu_data();
		top_meta_label = batch->meta_label_.mutable_cpu_data();
	}
	for (int item_id = 0; item_id < batch_size; ++item_id) {
		timer.Start();
		// get a datum
		Datum& datum = *(reader_.full().pop("Waiting for data"));
		read_time += timer.MicroSeconds();
		timer.Start();
		// Apply data transformations (mirror, scale, crop...)
		int offset = batch->data_.offset(item_id);
		this->transformed_data_.set_cpu_data(top_data + offset);
		this->data_transformer_->Transform(datum, &(this->transformed_data_));
		// Copy label.
		if (this->output_labels_) {
			int real_label = datum.label();
			int meta_label = labels_map_[real_label];

			top_label[item_id] = real_label;
			top_meta_label[item_id] = meta_label;
		}

		trans_time += timer.MicroSeconds();

		reader_.free().push(const_cast<Datum*>(&datum));
	}
	timer.Stop();
	batch_timer.Stop();
	DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
	DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
	DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(MapDataLayer);
REGISTER_LAYER_CLASS(MapData);

}  // namespace caffe
