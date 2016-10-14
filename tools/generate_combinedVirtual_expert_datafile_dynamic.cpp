/**
 * @ author Karim Ahmed
 * Generate combined Data files for all Experts.
 * Used for 1000 shared softmax for ouptput of fc8 of all Experts
 * 1-D label (Virtual). Total map 1000 virtual labels to 1000 real labels.
 * Should generate output mapping file.
 * Could generate the real class labels along in same file (2 labels per image)
 * To check accuracy automatically while training.
 */

#include <glog/logging.h>
#include <leveldb/db.h>
#include <lmdb.h>
#include <stdint.h>
#include <iostream>
#include <typeinfo>
#include <fstream>

using namespace std;

#include <cstring>
#include <map>

#include <boost/algorithm/string.hpp>
#include <boost/bind.hpp>

#include <algorithm>

#include <string>

//#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <utility>
#include <vector>
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using std::string;
using std::max;
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Timer;
using caffe::vector;

using namespace caffe;

// Input:

DEFINE_string(mainRealTrain, "main_real_train.txt",
		"File contains training set with real classes labels ");

DEFINE_string(mainRealVal, "main_real_val.txt",
		"File contains validation set with real classes labels ");

DEFINE_string(mapping, "mapping.txt",
		"Input: Mapping file from real class to Meta class for generalist. ");

/* instead read from mapping file in all cases..
DEFINE_int32(numRealClassesInExpert, 100,
		"Number of Real Classes assigned to the expert. 1000/10(Clusters) = 100"); */

/*DEFINE_string(numRealClassesInExpert_file, "",
		"Same as numRealClassesInExpert but read from file. Used in case of different number of classes per each meta."
		"In this case, must put ['numRealClassesInExpert' = -1], to be ignored. "
		"Number of lines in file = numMetaClasses."
		"Each line is number of real classes for this meta class.");*/

DEFINE_int32(numRealClasses, 1000, "Number of Real Classes");

DEFINE_int32(numMetaClasses, 10, "Number of Meta Classes/Experts");


// Output:

DEFINE_string(virtualRealmapping, "virtualRealmapping.txt",
		"Output: (0-999) maps Virtual Class IDs  to the Real Class id (0-999) ");


DEFINE_string(realVirtualmapping, "realVirtualmapping.txt",
		"Output: (0-999) maps Real Class IDs  to the Virtual Class id (0-999) ");

DEFINE_string(outTrain, "out_train.txt",
		"Output: File contains training set db, text or lmdb ");

DEFINE_string(outVal, "out_val.txt",
		"Output: File contains val set db, text or lmdb ");

DEFINE_string(numRealClassesInExpert_file, "numRealClassesInExpert_file.txt",
              "Output: File contains numRealClassesInExpert list");


std::vector<std::pair<int, std::pair<std::string, int> > > MAIN_REAL_VAL_lines;
std::vector<std::pair<int, std::pair<std::string, int> > > MAIN_REAL_TRAIN_lines;

string convertIntToString(int number) {
	stringstream ss;
	ss << number;
	return ss.str();
}

void Init_lines() {

	LOG(INFO)<< "*****Init_lines:******";
	LOG(INFO) << FLAGS_mainRealTrain;
	LOG(INFO) << FLAGS_mainRealVal;

	// Reset:
	MAIN_REAL_TRAIN_lines.clear();
	MAIN_REAL_VAL_lines.clear();

	std::ifstream main_real_train_file(FLAGS_mainRealTrain.c_str());
	std::ifstream main_real_val_file(FLAGS_mainRealVal.c_str());

	string filename = "";
	int label;
	int line_id = 0;

	LOG(INFO) << "Reading MAIN_REAL_TRAIN_lines...";
	line_id = 0;
	while (main_real_train_file >> filename >> label) {
		MAIN_REAL_TRAIN_lines.push_back(std::make_pair(line_id, std::make_pair(filename, label)));
		line_id++;
	}

	LOG(INFO) << "Reading MAIN_REAL_VAL_lines...";
	line_id = 0;
	while (main_real_val_file >> filename >> label) {
		MAIN_REAL_VAL_lines.push_back(std::make_pair(line_id, std::make_pair(filename, label)));
		line_id++;
	}

	main_real_val_file.close();
	main_real_train_file.close();

}
int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Generate Expert Data Files:\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	FLAGS_alsologtostderr = 1;

	// Load train and val main real files
	Init_lines();
    CHECK(MAIN_REAL_TRAIN_lines.size() != 0 ) << "Train File Empty !!";
    CHECK(MAIN_REAL_VAL_lines.size() != 0 ) << "Val File Empty !!";


	// No shuffle db, will be shuffled in proto, since label is 1-D only.
	//shuffle(MAIN_REAL_TRAIN_lines.begin(), MAIN_REAL_TRAIN_lines.end());
	//shuffle(MAIN_REAL_VAL_lines.begin(), MAIN_REAL_VAL_lines.end());

	// Load mapping file (mapping_164 from real to meta)
	std::ifstream mapping_file(FLAGS_mapping.c_str());
	vector<int> mapping;

	int count_real_class = 0;
	string meta_class_id_str;
	while (mapping_file >> meta_class_id_str) {
		mapping.push_back(atoi(meta_class_id_str.c_str()));
		LOG(INFO)<< "MAP:" << meta_class_id_str;
		count_real_class++;
	}
    CHECK(count_real_class ==  FLAGS_numRealClasses)  << "numRealClasses param not equal to mapping file lines";

    
    /* -- Get the new numMetaClasses and shift mapping -- */
    int NUM_META_CLASSES = 0 ; // new num will stored here...
    vector<int> meta_indicator (FLAGS_numMetaClasses, 0);
    for (int i = 0 ; i < FLAGS_numRealClasses; i++){
        int meta_class_id = mapping[i];
        meta_indicator[meta_class_id] = 1;
    }
    int count_meta = 0 ;
    for (int k = 0 ; k < FLAGS_numMetaClasses; k++){
        count_meta += meta_indicator[k];
    }
    NUM_META_CLASSES =count_meta;
    LOG(INFO)<< "NUM_META_CLASSES: " << NUM_META_CLASSES;
    
    // Get meta-to-meta mapping
    vector<int> metaToMetaMapping (FLAGS_numMetaClasses, -1);
    for (int k = 0 ; k < FLAGS_numMetaClasses; k++){
        if (meta_indicator[k] == 0) // loop on non-zeros only
            continue;
        int count_zeros = 0;
        for (int j = 0 ; j < FLAGS_numMetaClasses; j++){
            if (k == j)
                break;
            
            if (meta_indicator[j] == 0){
                count_zeros ++;
            }
            
        }
        
        metaToMetaMapping[k] = k - count_zeros;
        LOG(INFO)<< "metaToMetaMapping:" << metaToMetaMapping[k] ;

    }
    
    // shift mapping
    for (int i = 0 ; i < FLAGS_numRealClasses; i++){
        int meta_class_id = mapping[i];
        int new_meta_class_id = metaToMetaMapping[meta_class_id];
        mapping[i] = new_meta_class_id;
        LOG(INFO)<< "NEW_mapping:" << mapping[i] ;
    }
    
    /** Start process **/
    
	std::ofstream out_train_file(FLAGS_outTrain.c_str());
	std::ofstream out_val_file(FLAGS_outVal.c_str());

	vector<vector<int> > real_to_expert_mapping;
	for (int exp_id = 0; exp_id < NUM_META_CLASSES; exp_id++) { // init with -1
		vector<int> real_mapping(FLAGS_numRealClasses, -1); // init with -1
		real_to_expert_mapping.push_back(real_mapping);
	}

	  
    // variable to count numRealClasses per each Meta from mapping file
    vector<int> numRealClassesInExpertList (NUM_META_CLASSES, 0);
    
    for (int i = 0 ; i < FLAGS_numRealClasses; i++){
        int meta_class_id = mapping[i];
        numRealClassesInExpertList[meta_class_id]++;
    }
   
    vector<int> virtualExpertStartIndex (NUM_META_CLASSES, 0);
    
    for (int k = 0 ; k < NUM_META_CLASSES; k++){
        int start_index = 0;
        for (int expertId=0; expertId<NUM_META_CLASSES; expertId++ ){
            if (expertId == k)
                break;
            start_index += numRealClassesInExpertList[expertId];
            LOG(INFO)<< "numRealClassesInExpertList:[" << expertId << "]: " << numRealClassesInExpertList[expertId];
            
        }
        virtualExpertStartIndex[k] = start_index;
        LOG(INFO)<< "startIndex:[" << k << "]: " << start_index;
    }
    
    // write file numRealClassesInExpert_file
    std::ofstream num_rc_file(FLAGS_numRealClassesInExpert_file.c_str());
    num_rc_file << "NUM_META_CLASSES:= " << NUM_META_CLASSES << "\n";
    num_rc_file << "----------------------\n";
    for (int k = 0 ; k < NUM_META_CLASSES; k++){
        num_rc_file << "Expert[" << k << "]: " << numRealClassesInExpertList[k] << "\n";

    }
    num_rc_file.close();


	vector<int> current_expert_class_id(NUM_META_CLASSES, 0); //counter for each expert, values: 0-99
	vector<int> virtual_to_real_mapping(FLAGS_numRealClasses, -1); // Virtual <line_index> to Real mapping<value>.
	vector<int> real_to_virtual_mapping(FLAGS_numRealClasses, -1); // Real <line_index> to Virtual mapping<value>.

	/**
	 * Training Data.
	 */
	for (int i = 0; i < MAIN_REAL_TRAIN_lines.size(); i++) {
		string imgName = MAIN_REAL_TRAIN_lines[i].second.first;
		int real_class_id = MAIN_REAL_TRAIN_lines[i].second.second;
		// get meta class
		int expert_id = mapping[real_class_id]; // expert_id = meta_class_id

		// append image_name
		out_train_file << imgName;

		if (real_to_expert_mapping[expert_id][real_class_id] != -1) {
			//int virtual_class_id = (FLAGS_numRealClassesInExpert * expert_id) + real_to_expert_mapping[expert_id][real_class_id];
            int virtual_class_id = (virtualExpertStartIndex[expert_id]) + real_to_expert_mapping[expert_id][real_class_id];
            
			out_train_file << " " << virtual_class_id;

		} else {
			int cur_eid = current_expert_class_id[expert_id];
			real_to_expert_mapping[expert_id][real_class_id] = cur_eid;
			int virtual_class_id = (virtualExpertStartIndex[expert_id]) + cur_eid;
			out_train_file << " " << virtual_class_id;

			virtual_to_real_mapping[virtual_class_id] = real_class_id;
			real_to_virtual_mapping[real_class_id] = virtual_class_id;

			LOG(INFO)<< "**********************real_class_id:  " <<real_class_id;
            LOG(INFO)<< "**********************virtual_class_id:  " <<virtual_class_id;

			LOG(INFO)<< "**********************expertClassIdtoRealmapping:  " << virtual_to_real_mapping[virtual_class_id];

			current_expert_class_id[expert_id] = current_expert_class_id[expert_id] + 1;
		}

		out_train_file << "\n";
	}

	for (int expert_id = 0; expert_id < NUM_META_CLASSES; expert_id++) {
		LOG(INFO)<< "current_expert_class_id: expert" << expert_id << ":: " <<current_expert_class_id[expert_id];
	}

	for (int i = 0; i < FLAGS_numRealClasses; i++) {
		LOG(INFO)<< "Virtual_to_real: " << i << " => " << virtual_to_real_mapping[i];
	}

	// Write Virtual to Real
	std::ofstream virtualRealmappingFile(FLAGS_virtualRealmapping.c_str());
	for (int j = 0; j < FLAGS_numRealClasses; j++) {
		virtualRealmappingFile << virtual_to_real_mapping[j] << "\n";
	}
	virtualRealmappingFile.close();

	// Write Real to Virtual
	std::ofstream realVirtualmappingFile(FLAGS_realVirtualmapping.c_str());
	for (int j = 0; j < FLAGS_numRealClasses; j++) {
		realVirtualmappingFile << real_to_virtual_mapping[j] << "\n";
	}
	realVirtualmappingFile.close();




	/**
	 * Create val file.
	 */
	for (int i = 0; i < MAIN_REAL_VAL_lines.size(); i++) {
		int line_id = MAIN_REAL_VAL_lines[i].first;
		string imgName = MAIN_REAL_VAL_lines[i].second.first;
		int real_class_id = MAIN_REAL_VAL_lines[i].second.second;

		// get meta class
		int expert_id = mapping[real_class_id];

		out_val_file << imgName;

		int virtual_class_id = (virtualExpertStartIndex[expert_id])
				+ real_to_expert_mapping[expert_id][real_class_id];

		out_val_file << " " << virtual_class_id;

		out_val_file << "\n";

	}

	out_val_file.close();
	out_train_file.close();

	LOG(INFO)<< "<><><><><><>DONE: All Data files generated!<><><><><><>";
	return 0;

}

