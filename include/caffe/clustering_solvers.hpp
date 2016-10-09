#ifndef CAFFE_CLUSTERING_SOLVERS_HPP_
#define CAFFE_CLUSTERING_SOLVERS_HPP_

#include <string>
#include <vector>

#include "solver.hpp"
#include "sgd_solvers.hpp"


namespace caffe {




/**
 * @brief An interface for Class/Category Clustering Solver.
 *
 */

template <typename Dtype>
class ClassClustSolver: public SGDSolver<Dtype> {
 public:

 explicit ClassClustSolver(const SolverParameter& param)
     : SGDSolver<Dtype>(param) {}
 explicit ClassClustSolver(const string& param_file)
     : SGDSolver<Dtype>(param_file) {}
  virtual inline const char* type() const { return "ClassClust"; }
  virtual inline const bool blockFirstTestIter() const { return true; }


  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }

  void InitClusteringMode(bool append);
  void FinalizeClusteringMode();



 protected:

  void Test(const int test_net_id = 0); //override ;

  int num_real_labels_;
  int num_meta_labels_;
  int threshold_real_meta_;
  vector<int> current_labels_map_;
  shared_ptr<MapDataLayer<Dtype> >  train_data_layer_;

  shared_ptr<MapDataLayer<Dtype> >  test_data_layer_;

  shared_ptr<MapPredictionLayer<Dtype> > prediction_layer_;
  std::ofstream train_class_switch_file_, confusion_file_, accuracy_file_, distribution_file_;


  string online_optimization_type_;
  float lambda_;
  bool permute_;
  int epsilon_;

  /* vars for selecting subset dataset during Test() */
  int selected_test_dataset_size_; //input in solver

  vector<int> count_samples_per_real_class_;  // For internal usage
  int threshold_samples_per_real_class_; // For internal usage

  //
  DISABLE_COPY_AND_ASSIGN(ClassClustSolver);
};



}  // namespace caffe


#endif  // CAFFE_SGD_SOLVERS_HPP_
