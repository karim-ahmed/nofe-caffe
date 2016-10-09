/**
 * Generalist SOLVER.
 * @author: Karim Ahmed (karim@cs.dartmouth.edu)
 * Publication: "Network of Experts for large-scale Image Categorization", ECCV 2016
 */

#include <cstdio>

#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/bind.hpp>
#include "caffe/util/rng.hpp"
#include "caffe/util/common_utils.hpp"
#include "caffe/clustering_solvers.hpp"

namespace caffe {

template<typename Dtype>
void ClassClustSolver<Dtype>::InitClusteringMode(bool append) {

	this->train_data_layer_ = boost::dynamic_pointer_cast<MapDataLayer<Dtype> >(
			this->net_->layer_by_name("data"));	//(layers[0]);  // or net_->layers()[0]
	this->num_real_labels_ = this->param_.num_real_labels();
	this->num_meta_labels_ = this->param_.num_meta_labels();
	if (this->param_.has_threshold_real_meta()) {
		this->threshold_real_meta_ = this->param_.threshold_real_meta();
	}
	this->current_labels_map_ = this->train_data_layer_->getLabelsMap();

	//LOG(FATAL) << "TYPE>>>>>" << this->test_nets_[0]->layer_by_name("data")->type() ;

	//this->test_data_layer_ =  boost::dynamic_pointer_cast<MapDataLayer<Dtype> > (this->test_nets_[0]->layer_by_name("data"));

	//this->test_data_layer_ =  boost::dynamic_pointer_cast<SelectMapImageDataLayer<Dtype> > (this->test_nets_[0]->layer_by_name("data")); // TODO make generic or check data layer type
	this->test_data_layer_ = boost::dynamic_pointer_cast<MapDataLayer<Dtype> >(
			this->test_nets_[0]->layer_by_name("data")); // TODO make generic or check data layer type

	this->prediction_layer_ = boost::dynamic_pointer_cast<
			MapPredictionLayer<Dtype> >(
			this->test_nets_[0]->layer_by_name("prediction"));

	LOG(INFO) << "InitOnlineMode: append = " << append;
	if (append) {
		this->train_class_switch_file_.open(
				this->param_.train_class_switch_file().c_str(),
				ios::out | ios::app);
		this->confusion_file_.open(this->param_.confusion_file().c_str(),
				ios::out | ios::app);
		this->accuracy_file_.open(this->param_.accuracy_file().c_str(),
				ios::out | ios::app);

		this->distribution_file_.open(this->param_.distribution_file().c_str(),
				ios::out | ios::app);
	} else {
		this->train_class_switch_file_.open(
				this->param_.train_class_switch_file().c_str(),
				ios::out | ios::trunc);
		this->confusion_file_.open(this->param_.confusion_file().c_str(),
				ios::out | ios::trunc);
		this->accuracy_file_.open(this->param_.accuracy_file().c_str(),
				ios::out | ios::trunc);

		this->distribution_file_.open(this->param_.distribution_file().c_str(),
				ios::out | ios::trunc);
	}

	// Read from solver
	this->online_optimization_type_ = this->param_.online_optimization_type();
	this->lambda_ = this->param_.lambda();
	this->permute_ = this->param_.permute();
	this->epsilon_ = this->param_.epsilon();
	this->selected_test_dataset_size_ =
			this->param_.selected_test_dataset_size();

	LOG(INFO) << "### [Init ClassClust Solver: RESET class counters] ###";
	for (int k = 0; k < this->num_real_labels_; k++) {
		this->count_samples_per_real_class_.push_back(0);
	}

	this->threshold_samples_per_real_class_ = this->selected_test_dataset_size_
			/ this->num_real_labels_;

	LOG(INFO) << "### selected_test_dataset_size_:= "
			<< this->selected_test_dataset_size_;
	LOG(INFO) << "### threshold_samples_per_real_class_:= "
			<< this->threshold_samples_per_real_class_;

}

template<typename Dtype>
void ClassClustSolver<Dtype>::FinalizeClusteringMode() {

	this->train_class_switch_file_.close();
	this->confusion_file_.close();
	this->accuracy_file_.close();
	this->distribution_file_.close();

}

template<typename Dtype>
void ClassClustSolver<Dtype>::Solve(const char* resume_file) {

	LOG(INFO) << "## ClassClustSolver ##";

	CHECK(Caffe::root_solver());
	LOG(INFO) << "Solving " << this->net_->name();
	LOG(INFO) << "Learning Rate Policy: " << this->param_.lr_policy();

	LOG(INFO) << "ClassClustSolver:resume_file:= " << resume_file;

	if (resume_file)
		InitClusteringMode(true);
	else
		InitClusteringMode(false);

	// Initialize to false every time we start solving.
	this->requested_early_exit_ = false;

	// For a network that is trained by the solver, no bottom or top vecs
	// should be given, and we will just provide dummy vecs.
	this->Step(this->param_.max_iter() - this->iter_);
	// If we haven't already, save a snapshot after optimization, unless
	// overridden by setting snapshot_after_train := false
	if (this->param_.snapshot_after_train()
			&& (!this->param_.snapshot()
					|| this->iter_ % this->param_.snapshot() != 0)) {
		this->Snapshot();
	}
	if (this->requested_early_exit_) {
		LOG(INFO) << "Optimization stopped early.";
		return;
	}
	// After the optimization is done, run an additional train and test pass to
	// display the train and test loss/outputs if appropriate (based on the
	// display and test_interval settings, respectively).  Unlike in the rest of
	// training, for the train net we only run a forward pass as we've already
	// updated the parameters "max_iter" times -- this final pass is only done to
	// display the loss, which is computed in the forward pass.
	if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
		Dtype loss;
		this->net_->ForwardPrefilled(&loss);
		LOG(INFO) << "Iteration " << this->iter_ << ", loss = " << loss;
	}
	if (this->param_.test_interval()
			&& this->iter_ % this->param_.test_interval() == 0) {
		this->TestAll();
	}

	FinalizeClusteringMode();
	LOG(INFO) << "Optimization Done.";
}

template<typename Dtype>
void ClassClustSolver<Dtype>::Test(const int test_net_id) {

	LOG(INFO) << "-------------------------------------------";

	int conf_table[this->num_meta_labels_][this->num_meta_labels_];
	for (int row = 0; row < this->num_meta_labels_; row++) {
		for (int col = 0; col < this->num_meta_labels_; col++) {
			conf_table[row][col] = 0;
		}
	}

	// Used for optimization "MIN_LOSS"
	Dtype loss_conf[this->num_real_labels_][this->num_meta_labels_];

	int real_class_count[this->num_real_labels_][this->num_meta_labels_];
	float optimized_real_class_count[this->num_real_labels_][this->num_meta_labels_];

	vector<int> mapping(this->num_real_labels_, -1);

	for (int i = 0; i < this->num_real_labels_; i++) {
		for (int j = 0; j < this->num_meta_labels_; j++) {
			real_class_count[i][j] = 0;
			optimized_real_class_count[i][j] = 0.0;
			loss_conf[i][j] = 0.0;

		}
	}


	int count_all_samples = 0;

	// *********** Start Generalist Testing *********************************** //
	LOG(INFO) << "....... [Start Generalist Testing] .......";

	CHECK(Caffe::root_solver());
	LOG(INFO) << "Iteration " << this->iter_ << ", Testing net (#"
			<< test_net_id << ")";
	CHECK_NOTNULL(this->test_nets_[test_net_id].get())->ShareTrainedLayersWith(
			this->net_.get());
	vector<Dtype> test_score;
	vector<int> test_score_output_id;
	vector<Blob<Dtype>*> bottom_vec;
	const shared_ptr<Net<Dtype> >& test_net = this->test_nets_[test_net_id];
	Dtype loss = 0;

	count_all_samples = 0;
	for (int k = 0; k < this->num_real_labels_; k++) {
		this->count_samples_per_real_class_[k] = 0;
	}

	int count_iter = 0;

	/* i: current test_iter */
	for (int i = 0; i < this->param_.test_iter(test_net_id); ++i) {

		if (i == 0) { // first test_iter
			count_all_samples = 0;
			for (int k = 0; k < this->num_real_labels_; k++) {
				this->count_samples_per_real_class_[k] = 0;
			}
		}
		SolverAction::Enum request = this->GetRequestedAction();

		// Check to see if stoppage of testing/training has been requested.
		while (request != SolverAction::NONE) {
			if (SolverAction::SNAPSHOT == request) {
				this->Snapshot();
			} else if (SolverAction::STOP == request) {
				this->requested_early_exit_ = true;
			}
			request = this->GetRequestedAction();
		}

		if (this->requested_early_exit_) {
			// break out of test loop.
			break;
		}

		Dtype iter_loss;
		const vector<Blob<Dtype>*>& result = test_net->Forward(bottom_vec,
				&iter_loss);

		count_iter++;

		if (this->param_.test_compute_loss()) {
			loss += iter_loss;
		}
		if (i == 0) {
			for (int j = 0; j < result.size(); ++j) {
				const Dtype* result_vec = result[j]->cpu_data();

				for (int k = 0; k < result[j]->count(); ++k) {
					test_score.push_back(result_vec[k]);
					test_score_output_id.push_back(j);
				}
			}
		} else {
			int idx = 0;
			for (int j = 0; j < result.size(); ++j) {
				const Dtype* result_vec = result[j]->cpu_data();
				for (int k = 0; k < result[j]->count(); ++k) {
					test_score[idx++] += result_vec[k];
				}
			}
		}

		/***************  Per each Iteration count ****************************/
		// Fill real_class_count //
		vector<int> real_labels = this->prediction_layer_->get_labels();
		vector<int> meta_labels = this->prediction_layer_->get_meta_labels();
		vector<int> predicted_labels =
				this->prediction_layer_->get_predicted_labels();
		vector<vector<Dtype> > predicted_losses =
				this->prediction_layer_->get_predicted_losses();

		CHECK_EQ(real_labels.size(), meta_labels.size()) << "real_labels.size() not equal to meta_labels.size()";
		CHECK_EQ(real_labels.size(), predicted_labels.size())
				<< "real_labels.size() not equal to predicted_labels.size()";

		int num_samples = real_labels.size();

		for (int i = 0; i < num_samples; i++) {

			int real_class_label = real_labels[i];

			int old_meta_label = this->current_labels_map_[real_class_label]; // previous groundTruth
			int new_meta_label = predicted_labels[i]; // predicted

			/* Normal case: selected_test_dataset_size_ not provided */

			/*if (this->count_samples_per_real_class_[real_class_label] < this->threshold_samples_per_real_class_) {
			 this->count_samples_per_real_class_[real_class_label] ++;

			 }*/
			count_samples_per_real_class_[real_class_label]++;

			real_class_count[real_class_label][new_meta_label]++;
			conf_table[old_meta_label][new_meta_label]++;
			count_all_samples++;

			vector<Dtype> current_losses = predicted_losses[i];

			for (int j = 0; j < current_losses.size(); j++) {
				loss_conf[real_class_label][j] += current_losses[j];
			}

		}

	}

	if (this->requested_early_exit_) {
		LOG(INFO) << "Test interrupted.";
		return;
	}

	if (this->param_.test_compute_loss()) {
		loss /= this->param_.test_iter(test_net_id);
		LOG(INFO) << " Test loss: " << loss;
	}

	for (int i = 0; i < test_score.size(); ++i) {
		const int output_blob_index =
				test_net->output_blob_indices()[test_score_output_id[i]];
		const string& output_name = test_net->blob_names()[output_blob_index];
		const Dtype loss_weight =
				test_net->blob_loss_weights()[output_blob_index];
		ostringstream loss_msg_stream;
		const Dtype mean_score = test_score[i]
				/ this->param_.test_iter(test_net_id);
		if (loss_weight) {
			loss_msg_stream << " (* " << loss_weight << " = "
					<< loss_weight * mean_score << " loss)";
		}
		LOG(INFO) << "    Online ClassCLust Test net output #" << i << ": "
				<< output_name << " = " << mean_score << loss_msg_stream.str();
	}

	// After finishing Testing iterations:
	LOG(INFO) << "Writing confusion...";
	confusion_file_ << "===================== Iter: " << this->iter_
			<< " ====================\n";
	for (int row = 0; row < num_meta_labels_; row++) {
		int sum_current_row = 0;
		for (int col = 0; col < num_meta_labels_; col++) {
			if (col != (num_meta_labels_ - 1)) {
				sum_current_row += conf_table[row][col];
				confusion_file_ << convertIntToString(conf_table[row][col])
						<< ",";
			} else {
				sum_current_row += conf_table[row][col];
				confusion_file_ << convertIntToString(conf_table[row][col])
						<< ": " << sum_current_row << "\n";
			}
		}
	}
	confusion_file_.flush();

	LOG(INFO) << "     ..[Start online Refining].. ";

	LOG(INFO) << "count_all_samples: " << count_all_samples;

	LOG(INFO) << "Get New Mapping....";
	vector<int> rand_seq;	 // on real classes
	for (int i = 0; i < num_real_labels_; i++) {
		rand_seq.push_back(i);
	}

	if (permute_)
		shuffle(rand_seq.begin(), rand_seq.end());

	/*************************  FULLY_BALANCED **********************************/
	/* Also known as "MAX_ROW" */
	if ((string("FULLY_BALANCED")).compare(online_optimization_type_) == 0) {

		LOG(INFO) << "Online Optimization Type: FULLY_BALANCED";
		for (int i = 0; i < num_real_labels_; i++) { // row
			for (int k = 0; k < num_meta_labels_; k++) { // column
				optimized_real_class_count[i][k] = real_class_count[i][k];
			}
		}
		// For each real class_id, assign the the meta class_id
		// with maximum number of predicted samples.
		// optimized_real_class_count: Matrix |R<real_classes,meta_classes>
		for (int i = 0; i < num_real_labels_; i++) {
			int cur_real_index = rand_seq[i]; // row
			std::vector<std::pair<int, float> > sorted_cur_row; // <index, count>
			// ind = K (meta_class) (columns)
			// Sort on K
			for (int ind = 0; ind < num_meta_labels_; ind++) {
				sorted_cur_row.push_back(
						std::make_pair(ind,
								optimized_real_class_count[cur_real_index][ind]));
			}
			std::sort(sorted_cur_row.begin(), sorted_cur_row.end(),
					boost::bind(&std::pair<int, float>::second, _1)
							< boost::bind(&std::pair<int, float>::second, _2));

			// loop on sorted
			int max_meta_index = -1;			// 0-9
			for (int ss = (num_meta_labels_ - 1); ss >= 0; ss--) {
				// find index of current sorted meta
				max_meta_index = sorted_cur_row[ss].first;
				// set mapping
				// count real classes with same meta_index
				int max_meta_count = 0;
				for (int j = 0; j < num_real_labels_; j++) {
					if (mapping[j] == max_meta_index)
						max_meta_count++;
				}
				if (this->param_.has_threshold_real_meta()) { // Constraint
					if (max_meta_count < threshold_real_meta_) { // Threshold
						// this is it
						mapping[cur_real_index] = max_meta_index;
						break;
					} else {
						continue;
					}
				} else { // No Constraint/ Relaxed
					// Always take top-1 of sorted (first)
					mapping[cur_real_index] = max_meta_index;
					break;
				}
			}

		}


		/*************************  ELASSO **********************************/
		// Also known as "ELASSO_ACCUM"
	} else if ((string("ELASSO")).compare(online_optimization_type_) == 0) {

		LOG(INFO) << "Online Optimization Type: ELASSO";

		int FC_indicator[this->num_real_labels_][this->num_meta_labels_];

		// init FC with current labels map
		for (int i = 0; i < num_real_labels_; i++) { // row
			int cur_index = this->current_labels_map_[i];

			for (int k = 0; k < num_meta_labels_; k++) { // column
				FC_indicator[i][k] = 0;
				if (cur_index == k) {
					FC_indicator[i][k] = 1;
				}
			}
		}

		// get cluster_indicator matrix .
		for (int i = 0; i < num_real_labels_; i++) {

			int cur_real_index = rand_seq[i];

			float min_cost = INFINITY;
			float min_elasso = INFINITY;

			int min_cost_K_index = -1;
			float max_cost = 0.0;
			float max_elasso = 0.0;

			// Calculate cost function.
			for (int cur_k = 0; cur_k < num_meta_labels_; cur_k++) {

				FC_indicator[cur_real_index][cur_k] = 1;
				// set rest of this row to zero
				for (int sind = 0; sind < num_meta_labels_; sind++) {
					if (sind != cur_k) {
						FC_indicator[cur_real_index][sind] = 0;
					}
				}

				// calc elasso
				float elasso = 0.0;
				for (int col = 0; col < num_meta_labels_; col++) {
					float sum_col = 0.0;
					for (int row = 0; row < num_real_labels_; row++) {
						sum_col += FC_indicator[row][col];
					}
					sum_col = pow(sum_col, 2.0);
					elasso += sum_col;

				}

				elasso = sqrt(elasso);

				// Calculate first part ||C.*F||
				int CF_part = 0;
				int CFMat[this->num_real_labels_][this->num_meta_labels_];
				for (int col = 0; col < num_meta_labels_; col++) {
					float sum_col = 0.0;
					for (int row = 0; row < num_real_labels_; row++) {
						if (FC_indicator[row][col] == 1) {
							CFMat[row][col] = real_class_count[row][col];
						} else {
							CFMat[row][col] = 0;
						}
						sum_col += CFMat[row][col];
					}
					CF_part += sum_col;

				}

				// Cost Function
				float cost_value = (CF_part * -1) + (this->lambda_ * elasso);

				if (cost_value < min_cost) {
					min_cost = cost_value;
					min_cost_K_index = cur_k;
				}
				if (cost_value > max_cost) {
					max_cost = cost_value;
				}
				if (elasso < min_elasso) {
					min_elasso = elasso;
				}
				if (elasso > max_elasso) {
					max_elasso = elasso;
				}
			}



			// after end of this current row, reset the FC_indicator row with
			// the correct index (min_elasso_K_index)
			for (int sind = 0; sind < num_meta_labels_; sind++) {
				FC_indicator[cur_real_index][sind] = 0;

			}
			FC_indicator[cur_real_index][min_cost_K_index] = 1;

		}

		for (int i = 0; i < num_real_labels_; i++) {
			for (int k = 0; k < num_meta_labels_; k++) {
				if (FC_indicator[i][k] == 1) {
					mapping[i] = k;
				}
			}
		}


	}

	/* Check mapping and write */
	LOG(INFO) << "Write New Mapping file iter:= " << this->iter_;
	string map_fileName = this->param_.mapping_file_prefix() + "_iter"
			+ convertIntToString(this->iter_) + ".txt";
	std::ofstream mapping_file(map_fileName.c_str());
	int check_sum = 0;
	int check_sum_grounttruth = 0;
	int num_switch_labels = 0;

	for (int i = 0; i < num_real_labels_; i++) {
		mapping_file << convertIntToString(mapping[i]) + "\n";
		check_sum += mapping[i];
		if (this->current_labels_map_[i] != mapping[i])
			num_switch_labels++;
	}
	for (int i = 0; i < num_meta_labels_; i++)
		check_sum_grounttruth += i * threshold_real_meta_;

	LOG(INFO) << "CHECK_SUM:" + convertIntToString(check_sum);
	LOG(INFO)
			<< "CHECK_SUM_GroundTruth:"
					+ convertIntToString(check_sum_grounttruth);
	mapping_file.close();

	LOG(INFO) << "num_switch_labels:= " << num_switch_labels;
	train_class_switch_file_ << "==Iter: " << this->iter_ << ": "
			<< num_switch_labels << "\n";
	train_class_switch_file_.flush();

	LOG(INFO) << "Reload Mapping Files into Data Layers...";
	this->current_labels_map_ = mapping;
	this->train_data_layer_->setLabelsMap(mapping);
	this->test_data_layer_->setLabelsMap(mapping);

	/* Calculate and write distribution after Refining */
	LOG(INFO) << "Calculate and Write distribution... ";

	int real_per_meta[num_meta_labels_][this->num_real_labels_];
	for (int i = 0; i < num_meta_labels_; i++) {
		for (int j = 0; j < num_real_labels_; j++) {
			real_per_meta[i][j] = 0;
		}
	}

	for (int i = 0; i < num_real_labels_; i++) {
		int meta_label = this->current_labels_map_[i];
		real_per_meta[meta_label][i] = 1;
	}

	for (int i = 0; i < num_meta_labels_; i++) {
		std::stringstream temp_str;
		temp_str.clear();
		int count_real = 0;
		for (int j = 0; j < num_real_labels_; j++) {
			if (real_per_meta[i][j] == 1) {
				temp_str << convertIntToString(j) << ",";
				count_real++;
			}
		}
		distribution_file_ << "[" << convertIntToString(count_real) << "], ";
		distribution_file_ << temp_str;

		distribution_file_ << "\n";

	}
	distribution_file_.flush();

	LOG(INFO) << "....... [ClassClust Testing Phase Done] .......";
	LOG(INFO) << "-------------------------------------------";

}

template<typename Dtype>
void Solver<Dtype>::ReTest(const int test_net_id) {
	LOG(INFO) << "....... [Start - classClust- ReTesting] .......";

	CHECK(Caffe::root_solver());
	LOG(INFO) << "Iteration " << iter_ << ", ReTesting net - classClust- (#"
			<< test_net_id << ")";
	CHECK_NOTNULL(test_nets_[test_net_id].get())->ShareTrainedLayersWith(
			net_.get());
	vector<Dtype> test_score;
	vector<int> test_score_output_id;
	vector<Blob<Dtype>*> bottom_vec;
	const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
	Dtype loss = 0;
	for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
		SolverAction::Enum request = GetRequestedAction();
		// Check to see if stoppage of testing/training has been requested.
		while (request != SolverAction::NONE) {
			if (SolverAction::SNAPSHOT == request) {
				Snapshot();
			} else if (SolverAction::STOP == request) {
				requested_early_exit_ = true;
			}
			request = GetRequestedAction();
		}
		if (requested_early_exit_) {
			// break out of test loop.
			break;
		}

		Dtype iter_loss;
		const vector<Blob<Dtype>*>& result = test_net->Forward(bottom_vec,
				&iter_loss);
		if (param_.test_compute_loss()) {
			loss += iter_loss;
		}
		if (i == 0) {
			for (int j = 0; j < result.size(); ++j) {
				const Dtype* result_vec = result[j]->cpu_data();
				for (int k = 0; k < result[j]->count(); ++k) {
					test_score.push_back(result_vec[k]);
					test_score_output_id.push_back(j);
				}
			}
		} else {
			int idx = 0;
			for (int j = 0; j < result.size(); ++j) {
				const Dtype* result_vec = result[j]->cpu_data();
				for (int k = 0; k < result[j]->count(); ++k) {
					test_score[idx++] += result_vec[k];
				}
			}
		}
	}
	if (requested_early_exit_) {
		LOG(INFO) << "ReTest interrupted.";
		return;
	}
	if (param_.test_compute_loss()) {
		loss /= param_.test_iter(test_net_id);
		LOG(INFO) << "ReTest loss: " << loss;
	}
	for (int i = 0; i < test_score.size(); ++i) {
		const int output_blob_index =
				test_net->output_blob_indices()[test_score_output_id[i]];
		const string& output_name = test_net->blob_names()[output_blob_index];
		const Dtype loss_weight =
				test_net->blob_loss_weights()[output_blob_index];
		ostringstream loss_msg_stream;
		const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
		if (loss_weight) {
			loss_msg_stream << " (* " << loss_weight << " = "
					<< loss_weight * mean_score << " loss)";
		}
		LOG(INFO) << "    ReTest net output #" << i << ": " << output_name
				<< " = " << mean_score << loss_msg_stream.str();
	}
	LOG(INFO) << "-------------------------------------------";

}

INSTANTIATE_CLASS(ClassClustSolver);
REGISTER_SOLVER_CLASS(ClassClust);

}  // namespace caffe

