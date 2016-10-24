#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MapPredictionLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.prediction_param().top_k();

  has_ignore_label_ =
    this->layer_param_.prediction_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.prediction_param().ignore_label();
  }

  this->predicted_labels_.clear();
  this->meta_labels_.clear();
  this->labels_.clear();
  this->predicted_probs_.clear();
  this->predicted_losses_.clear();


}


template <typename Dtype>
const vector<int> MapPredictionLayer<Dtype>::get_labels(){ // real labels
	return this->labels_;
}

template <typename Dtype>
const vector<int> MapPredictionLayer<Dtype>::get_meta_labels(){
	return this->meta_labels_;
}

template <typename Dtype>
const vector<int> MapPredictionLayer<Dtype>::get_predicted_labels(){
	return this->predicted_labels_;
}

template <typename Dtype>
const vector< vector<Dtype> > MapPredictionLayer<Dtype>::get_predicted_probs(){
	return this->predicted_probs_;
}
template <typename Dtype>
const vector< vector<Dtype> > MapPredictionLayer<Dtype>::get_predicted_losses(){
	return this->predicted_losses_;
}

template <typename Dtype>
void MapPredictionLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
  //    << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.prediction_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);

  // Must be: outer_num_ * inner_num_ = bottom[1]->count() = bottom[2]->count()
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";

  CHECK_EQ(outer_num_ * inner_num_, bottom[2]->count())
       << "Number of labels must match number of predictions; "
       << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
       << "label count (number of labels) must be N*H*W, "
       << "with integer values in {0, 1, ..., C-1}.";


  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    vector<int> top_shape_per_class(1);
    top_shape_per_class[0] = bottom[0]->shape(label_axis_);
    top[1]->Reshape(top_shape_per_class);
    nums_buffer_.Reshape(top_shape_per_class);
  }
}

template <typename Dtype>
void MapPredictionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  this->meta_labels_.clear();
  this->labels_.clear();
  this->predicted_labels_.clear();
  this->predicted_probs_.clear();
  this->predicted_losses_.clear();

  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data(); // real labels
  const Dtype* bottom_meta_label = bottom[2]->cpu_data(); // meta labels

  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);

  //LOG(INFO) << "MapPredictionLayer:FORWARD:  num_labels: " << num_labels;

  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  if (top.size() > 1) {
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }
  int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {

      const int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);

      const int meta_label_value =
          static_cast<int>(bottom_meta_label[i * inner_num_ + j]);


      // Put groundTruth label
      this->labels_.push_back(label_value); // real
      this->meta_labels_.push_back(meta_label_value);



      if (has_ignore_label_ && meta_label_value == ignore_label_) {
        continue;
      }
      if (top.size() > 1) ++nums_buffer_.mutable_cpu_data()[meta_label_value];
      DCHECK_GE(meta_label_value, 0);
      DCHECK_LT(meta_label_value, num_labels);
      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k));
      }

      vector<Dtype> current_probs;
      current_probs.clear();

      vector<Dtype> current_losses;
      current_losses.clear();

      for (int ind = 0 ; ind <  bottom_data_vector.size(); ind++){
    	  Dtype prob_val = bottom_data_vector[ind].first;
		  current_probs.push_back(prob_val);
		  Dtype th_prob_val = std::max(prob_val, Dtype(kLOG_THRESHOLD));
		  Dtype loss_val = - log(th_prob_val);
		  current_losses.push_back(loss_val);
      }
      this->predicted_probs_.push_back(current_probs);
      this->predicted_losses_.push_back(current_losses);

      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());

      // put top-1 prediction in vector prediction labels
      this->predicted_labels_.push_back(bottom_data_vector[0].second);

      // check if true label is in top k predictions
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == meta_label_value) {
          ++accuracy;
          if (top.size() > 1) ++top[1]->mutable_cpu_data()[meta_label_value];
          break;
        }
      }
      ++count;
    }
  }

  if (count == 0 ){
	   top[0]->mutable_cpu_data()[0] = 0;
   }else {
	   top[0]->mutable_cpu_data()[0] = accuracy / count;
   }


  if (top.size() > 1) {
    for (int i = 0; i < top[1]->count(); ++i) {
      top[1]->mutable_cpu_data()[i] =
          nums_buffer_.cpu_data()[i] == 0 ? 0
          : top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i];
    }
  }
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(MapPredictionLayer);
REGISTER_LAYER_CLASS(MapPrediction);

}  // namespace caffe
