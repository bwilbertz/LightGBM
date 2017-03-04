#include <omp.h>

#include <LightGBM/utils/common.h>
#include <LightGBM/utils/random.h>
#include <LightGBM/api.hpp>
#include <LightGBM/dataset_loader.h>


#include <cstdio>
#include <string>
#include <cstring>


namespace LightGBM {

  Booster::Booster(const Dataset* train_data,
    std::vector<const Dataset*> valid_data,
    std::vector<std::string> valid_names,
    const char* parameters)
    :train_data_(train_data), valid_datas_(valid_data) {
    config_.LoadFromString(parameters);
    Init(train_data, valid_data, valid_names);
  }
  
  Booster::Booster(const Dataset* train_data,
    std::vector<const Dataset*> valid_data,
    std::vector<std::string> valid_names,
    std::unordered_map<std::string, std::string>& params)
    :train_data_(train_data), valid_datas_(valid_data) {
    ParameterAlias::KeyAliasTransform(&params);
    config_.Set(params);
    Init(train_data, valid_data, valid_names);
  }
   
  void Booster::Init(const Dataset* train_data,
      std::vector<const Dataset*> valid_data,
      std::vector<std::string> valid_names) {
     if (config_.num_threads > 0) {
       omp_set_num_threads(config_.num_threads);
     }
     // create boosting
     if (config_.io_config.input_model.size() > 0) {
       Log::Warning("continued train from model is not support for c_api, \
         please use continued train with input score");
     }
     boosting_.reset(Boosting::CreateBoosting(config_.boosting_type, ""));
     // create objective function
     objective_fun_.reset(ObjectiveFunction::CreateObjectiveFunction(config_.objective_type,
       config_.objective_config));
     // create training metric
     for (auto metric_type : config_.metric_types) {
       auto metric = std::unique_ptr<Metric>(
         Metric::CreateMetric(metric_type, config_.metric_config));
       if (metric == nullptr) { continue; }
       metric->Init("training", train_data_->metadata(),
         train_data_->num_data());
       train_metric_.push_back(std::move(metric));
     }

     // add metric for validation data
     for (size_t i = 0; i < valid_datas_.size(); ++i) {
       valid_metrics_.emplace_back();
       for (auto metric_type : config_.metric_types) {
         auto metric = std::unique_ptr<Metric>(Metric::CreateMetric(metric_type, config_.metric_config));
         if (metric == nullptr) { continue; }
         metric->Init(valid_names[i].c_str(),
           valid_datas_[i]->metadata(),
           valid_datas_[i]->num_data());
         valid_metrics_.back().push_back(std::move(metric));
       }
     }
     // initialize the objective function
     objective_fun_->Init(train_data_->metadata(), train_data_->num_data());
     // initialize the boosting
     boosting_->Init(&config_.boosting_config, train_data_, objective_fun_.get(),
       Common::ConstPtrInVectorWrapper<Metric>(train_metric_));
     // add validation data into boosting
     for (size_t i = 0; i < valid_datas_.size(); ++i) {
       boosting_->AddDataset(valid_datas_[i],
         Common::ConstPtrInVectorWrapper<Metric>(valid_metrics_[i]));
     }
   }

  Booster::~Booster() {
  }

  bool Booster::TrainOneIter(bool is_eval) {
    return boosting_->TrainOneIter(nullptr, nullptr, is_eval);
  }

  bool Booster::TrainOneIter(const float* gradients, const float* hessians, bool is_eval) {
    return boosting_->TrainOneIter(gradients, hessians, false);
  }

  void Booster::PrepareForPrediction(int num_used_model, int predict_type) {
    boosting_->SetNumUsedModel(num_used_model);
    bool is_predict_leaf = false;
    bool is_raw_score = false;
    if (predict_type == C_API_PREDICT_LEAF_INDEX) {
      is_predict_leaf = true;
    } else if (predict_type == C_API_PREDICT_RAW_SCORE) {
      is_raw_score = true;
    } else {
      is_raw_score = false;
    }
    predictor_.reset(new Predictor(boosting_.get(), is_raw_score, is_predict_leaf));
  }

  std::vector<double> Booster::Predict(const std::vector<std::pair<int, double>>& features) {
    return predictor_->GetPredictFunction()(features);
  }

  void Booster::PredictForFile(const char* data_filename, const char* result_filename, bool data_has_header) {
    predictor_->Predict(data_filename, result_filename, data_has_header);
  }

  void Booster::SaveModelToFile(int num_used_model, const char* filename) {
    boosting_->SaveModelToFile(num_used_model, true, filename);
  }
  
}

using namespace LightGBM;


DllExport Dataset* CreateDatasetFromMat(const void* data,
  int data_type,
  int32_t nrow,
  int32_t ncol,
  int is_row_major,
  const char* parameters,
  const DatesetHandle* reference) {

  OverallConfig config;
  config.LoadFromString(parameters);
  DatasetLoader loader(config.io_config, nullptr);
  std::unique_ptr<Dataset> ret;
  auto get_row_fun = RowFunctionFromDenseMatric(data, nrow, ncol, data_type, is_row_major);
  if (reference == nullptr) {
    // sample data first
    Random rand(config.io_config.data_random_seed);
    const int sample_cnt = static_cast<int>(nrow < config.io_config.bin_construct_sample_cnt ? nrow : config.io_config.bin_construct_sample_cnt);
    auto sample_indices = rand.Sample(nrow, sample_cnt);
    std::vector<std::vector<double>> sample_values(ncol);
    for (size_t i = 0; i < sample_indices.size(); ++i) {
      auto idx = sample_indices[i];
      auto row = get_row_fun(static_cast<int>(idx));
      for (size_t j = 0; j < row.size(); ++j) {
        if (std::fabs(row[j]) > 1e-15) {
          sample_values[j].push_back(row[j]);
        }
      }
    }
    ret.reset(loader.CostructFromSampleData(sample_values, sample_cnt, nrow));
  } else {
    ret.reset(new Dataset(nrow, config.io_config.num_class));
    ret->CopyFeatureMapperFrom(reinterpret_cast<const Dataset*>(*reference), config.io_config.is_enable_sparse);
  }

#pragma omp parallel for schedule(guided)
  for (int i = 0; i < nrow; ++i) {
    const int tid = omp_get_thread_num();
    auto one_row = get_row_fun(i);
    ret->PushOneRow(tid, i, one_row);
  }
  ret->FinishLoad();
  return ret.release();
}

DllExport Dataset* CreateDatasetFromCSR(const void* indptr,
  int indptr_type,
  const int32_t* indices,
  const void* data,
  int data_type,
  int64_t nindptr,
  int64_t nelem,
  int64_t num_col,
  const char* parameters,
  const DatesetHandle* reference) {

  OverallConfig config;
  config.LoadFromString(parameters);
  DatasetLoader loader(config.io_config, nullptr);
  std::unique_ptr<Dataset> ret;
  auto get_row_fun = RowFunctionFromCSR(indptr, indptr_type, indices, data, data_type, nindptr, nelem);
  int32_t nrow = static_cast<int32_t>(nindptr - 1);
  if (reference == nullptr) {
    // sample data first
    Random rand(config.io_config.data_random_seed);
    const int sample_cnt = static_cast<int>(nrow < config.io_config.bin_construct_sample_cnt ? nrow : config.io_config.bin_construct_sample_cnt);
    auto sample_indices = rand.Sample(nrow, sample_cnt);
    std::vector<std::vector<double>> sample_values;
    for (size_t i = 0; i < sample_indices.size(); ++i) {
      auto idx = sample_indices[i];
      auto row = get_row_fun(static_cast<int>(idx));
      for (std::pair<int, double>& inner_data : row) {
        if (std::fabs(inner_data.second) > 1e-15) {
          if (static_cast<size_t>(inner_data.first) >= sample_values.size()) {
            // if need expand feature set
            size_t need_size = inner_data.first - sample_values.size() + 1;
            for (size_t j = 0; j < need_size; ++j) {
              sample_values.emplace_back();
            }
          }
          // edit the feature value
          sample_values[inner_data.first].push_back(inner_data.second);
        }
      }
    }
    CHECK(num_col >= static_cast<int>(sample_values.size()));
    ret.reset(loader.CostructFromSampleData(sample_values, sample_cnt, nrow));
  } else {
    ret.reset(new Dataset(nrow, config.io_config.num_class));
    ret->CopyFeatureMapperFrom(reinterpret_cast<const Dataset*>(*reference), config.io_config.is_enable_sparse);
  }

#pragma omp parallel for schedule(guided)
  for (int i = 0; i < nindptr - 1; ++i) {
    const int tid = omp_get_thread_num();
    auto one_row = get_row_fun(i);
    ret->PushOneRow(tid, i, one_row);
  }
  ret->FinishLoad();
  return ret.release();
}


DllExport Dataset* CreateDatasetFromCSC(const void* col_ptr,
  int col_ptr_type,
  const int32_t* indices,
  const void* data,
  int data_type,
  int64_t ncol_ptr,
  int64_t nelem,
  int64_t num_row,
  const char* parameters,
  const DatesetHandle* reference) {
  OverallConfig config;
  config.LoadFromString(parameters);
  DatasetLoader loader(config.io_config, nullptr);
  std::unique_ptr<Dataset> ret;
  auto get_col_fun = ColumnFunctionFromCSC(col_ptr, col_ptr_type, indices, data, data_type, ncol_ptr, nelem);
  int32_t nrow = static_cast<int32_t>(num_row);
  if (reference == nullptr) {
    Log::Info("Construct from CSC format is not efficient");
    // sample data first
    Random rand(config.io_config.data_random_seed);
    const int sample_cnt = static_cast<int>(nrow < config.io_config.bin_construct_sample_cnt ? nrow : config.io_config.bin_construct_sample_cnt);
    auto sample_indices = rand.Sample(nrow, sample_cnt);
    std::vector<std::vector<double>> sample_values(ncol_ptr - 1);
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < static_cast<int>(sample_values.size()); ++i) {
      auto cur_col = get_col_fun(i);
      sample_values[i] = SampleFromOneColumn(cur_col, sample_indices);
    }
    ret.reset(loader.CostructFromSampleData(sample_values, sample_cnt, nrow));
  } else {
    ret.reset(new Dataset(nrow, config.io_config.num_class));
    ret->CopyFeatureMapperFrom(reinterpret_cast<const Dataset*>(*reference), config.io_config.is_enable_sparse);
  }

#pragma omp parallel for schedule(guided)
  for (int i = 0; i < ncol_ptr - 1; ++i) {
    const int tid = omp_get_thread_num();
    auto one_col = get_col_fun(i);
    ret->PushOneColumn(tid, i, one_col);
  }
  ret->FinishLoad();
  return ret.release();
}

// ---- start of some help functions

std::function<std::vector<double>(int row_idx)>
RowFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major) {
  if (data_type == C_API_DTYPE_FLOAT32) {
    const float* data_ptr = reinterpret_cast<const float*>(data);
    if (is_row_major) {
      return [data_ptr, num_col, num_row](int row_idx) {
        CHECK(row_idx < num_row);
        std::vector<double> ret;
        auto tmp_ptr = data_ptr + num_col * row_idx;
        for (int i = 0; i < num_col; ++i) {
          ret.push_back(static_cast<double>(*(tmp_ptr + i)));
        }
        return ret;
      };
    } else {
      return [data_ptr, num_col, num_row](int row_idx) {
        CHECK(row_idx < num_row);
        std::vector<double> ret;
        for (int i = 0; i < num_col; ++i) {
          ret.push_back(static_cast<double>(*(data_ptr + num_row * i + row_idx)));
        }
        return ret;
      };
    }
  } else if (data_type == C_API_DTYPE_FLOAT64) {
    const double* data_ptr = reinterpret_cast<const double*>(data);
    if (is_row_major) {
      return [data_ptr, num_col, num_row](int row_idx) {
        CHECK(row_idx < num_row);
        std::vector<double> ret;
        auto tmp_ptr = data_ptr + num_col * row_idx;
        for (int i = 0; i < num_col; ++i) {
          ret.push_back(static_cast<double>(*(tmp_ptr + i)));
        }
        return ret;
      };
    } else {
      return [data_ptr, num_col, num_row](int row_idx) {
        CHECK(row_idx < num_row);
        std::vector<double> ret;
        for (int i = 0; i < num_col; ++i) {
          ret.push_back(static_cast<double>(*(data_ptr + num_row * i + row_idx)));
        }
        return ret;
      };
    }
  } else {
    Log::Fatal("unknown data type in RowFunctionFromDenseMatric");
  }
  return nullptr;
}

std::function<std::vector<std::pair<int, double>>(int row_idx)>
RowPairFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major) {
  auto inner_function = RowFunctionFromDenseMatric(data, num_row, num_col, data_type, is_row_major);
  if (inner_function != nullptr) {
    return [inner_function](int row_idx) {
      auto raw_values = inner_function(row_idx);
      std::vector<std::pair<int, double>> ret;
      for (int i = 0; i < static_cast<int>(raw_values.size()); ++i) {
        if (std::fabs(raw_values[i]) > 1e-15) {
          ret.emplace_back(i, raw_values[i]);
        }
      }
      return ret;
    };
  }
  return nullptr;
}

std::function<std::vector<std::pair<int, double>>(int idx)>
RowFunctionFromCSR(const void* indptr, int indptr_type, const int32_t* indices, const void* data, int data_type, int64_t nindptr, int64_t nelem) {
  if (data_type == C_API_DTYPE_FLOAT32) {
    const float* data_ptr = reinterpret_cast<const float*>(data);
    if (indptr_type == C_API_DTYPE_INT32) {
      const int32_t* ptr_indptr = reinterpret_cast<const int32_t*>(indptr);
      return [ptr_indptr, indices, data_ptr, nindptr, nelem](int idx) {
        CHECK(idx + 1 < nindptr);
        std::vector<std::pair<int, double>> ret;
        int64_t start = ptr_indptr[idx];
        int64_t end = ptr_indptr[idx + 1];
        CHECK(start >= 0 && end <= nelem);
        for (int64_t i = start; i < end; ++i) {
          ret.emplace_back(indices[i], data_ptr[i]);
        }
        return ret;
      };
    } else if (indptr_type == C_API_DTYPE_INT64) {
      const int64_t* ptr_indptr = reinterpret_cast<const int64_t*>(indptr);
      return [ptr_indptr, indices, data_ptr, nindptr, nelem](int idx) {
        CHECK(idx + 1 < nindptr);
        std::vector<std::pair<int, double>> ret;
        int64_t start = ptr_indptr[idx];
        int64_t end = ptr_indptr[idx + 1];
        CHECK(start >= 0 && end <= nelem);
        for (int64_t i = start; i < end; ++i) {
          ret.emplace_back(indices[i], data_ptr[i]);
        }
        return ret;
      };
    } else {
      Log::Fatal("unknown data type in RowFunctionFromCSR");
    }
  } else if (data_type == C_API_DTYPE_FLOAT64) {
    const double* data_ptr = reinterpret_cast<const double*>(data);
    if (indptr_type == C_API_DTYPE_INT32) {
      const int32_t* ptr_indptr = reinterpret_cast<const int32_t*>(indptr);
      return [ptr_indptr, indices, data_ptr, nindptr, nelem](int idx) {
        CHECK(idx + 1 < nindptr);
        std::vector<std::pair<int, double>> ret;
        int64_t start = ptr_indptr[idx];
        int64_t end = ptr_indptr[idx + 1];
        CHECK(start >= 0 && end <= nelem);
        for (int64_t i = start; i < end; ++i) {
          ret.emplace_back(indices[i], data_ptr[i]);
        }
        return ret;
      };
    } else if (indptr_type == C_API_DTYPE_INT64) {
      const int64_t* ptr_indptr = reinterpret_cast<const int64_t*>(indptr);
      return [ptr_indptr, indices, data_ptr, nindptr, nelem](int idx) {
        CHECK(idx + 1 < nindptr);
        std::vector<std::pair<int, double>> ret;
        int64_t start = ptr_indptr[idx];
        int64_t end = ptr_indptr[idx + 1];
        CHECK(start >= 0 && end <= nelem);
        for (int64_t i = start; i < end; ++i) {
          ret.emplace_back(indices[i], data_ptr[i]);
        }
        return ret;
      };
    } else {
      Log::Fatal("unknown data type in RowFunctionFromCSR");
    }
  } else {
    Log::Fatal("unknown data type in RowFunctionFromCSR");
  }
  return nullptr;
}

std::function<std::vector<std::pair<int, double>>(int idx)>
ColumnFunctionFromCSC(const void* col_ptr, int col_ptr_type, const int32_t* indices, const void* data, int data_type, int64_t ncol_ptr, int64_t nelem) {
  if (data_type == C_API_DTYPE_FLOAT32) {
    const float* data_ptr = reinterpret_cast<const float*>(data);
    if (col_ptr_type == C_API_DTYPE_INT32) {
      const int32_t* ptr_col_ptr = reinterpret_cast<const int32_t*>(col_ptr);
      return [ptr_col_ptr, indices, data_ptr, ncol_ptr, nelem](int idx) {
        CHECK(idx + 1 < ncol_ptr);
        std::vector<std::pair<int, double>> ret;
        int64_t start = ptr_col_ptr[idx];
        int64_t end = ptr_col_ptr[idx + 1];
        CHECK(start >= 0 && end <= nelem);
        for (int64_t i = start; i < end; ++i) {
          ret.emplace_back(indices[i], data_ptr[i]);
        }
        return ret;
      };
    } else if (col_ptr_type == C_API_DTYPE_INT64) {
      const int64_t* ptr_col_ptr = reinterpret_cast<const int64_t*>(col_ptr);
      return [ptr_col_ptr, indices, data_ptr, ncol_ptr, nelem](int idx) {
        CHECK(idx + 1 < ncol_ptr);
        std::vector<std::pair<int, double>> ret;
        int64_t start = ptr_col_ptr[idx];
        int64_t end = ptr_col_ptr[idx + 1];
        CHECK(start >= 0 && end <= nelem);
        for (int64_t i = start; i < end; ++i) {
          ret.emplace_back(indices[i], data_ptr[i]);
        }
        return ret;
      };
    } else {
      Log::Fatal("unknown data type in ColumnFunctionFromCSC");
    }
  } else if (data_type == C_API_DTYPE_FLOAT64) {
    const double* data_ptr = reinterpret_cast<const double*>(data);
    if (col_ptr_type == C_API_DTYPE_INT32) {
      const int32_t* ptr_col_ptr = reinterpret_cast<const int32_t*>(col_ptr);
      return [ptr_col_ptr, indices, data_ptr, ncol_ptr, nelem](int idx) {
        CHECK(idx + 1 < ncol_ptr);
        std::vector<std::pair<int, double>> ret;
        int64_t start = ptr_col_ptr[idx];
        int64_t end = ptr_col_ptr[idx + 1];
        CHECK(start >= 0 && end <= nelem);
        for (int64_t i = start; i < end; ++i) {
          ret.emplace_back(indices[i], data_ptr[i]);
        }
        return ret;
      };
    } else if (col_ptr_type == C_API_DTYPE_INT64) {
      const int64_t* ptr_col_ptr = reinterpret_cast<const int64_t*>(col_ptr);
      return [ptr_col_ptr, indices, data_ptr, ncol_ptr, nelem](int idx) {
        CHECK(idx + 1 < ncol_ptr);
        std::vector<std::pair<int, double>> ret;
        int64_t start = ptr_col_ptr[idx];
        int64_t end = ptr_col_ptr[idx + 1];
        CHECK(start >= 0 && end <= nelem);
        for (int64_t i = start; i < end; ++i) {
          ret.emplace_back(indices[i], data_ptr[i]);
        }
        return ret;
      };
    } else {
      Log::Fatal("unknown data type in ColumnFunctionFromCSC");
    }
  } else {
    Log::Fatal("unknown data type in ColumnFunctionFromCSC");
  }
  return nullptr;
}

std::vector<double> SampleFromOneColumn(const std::vector<std::pair<int, double>>& data, const std::vector<int>& indices) {
  size_t j = 0;
  std::vector<double> ret;
  for (auto row_idx : indices) {
    while (j < data.size() && data[j].first < static_cast<int>(row_idx)) {
      ++j;
    }
    if (j < data.size() && data[j].first == static_cast<int>(row_idx)) {
      ret.push_back(data[j].second);
    } else {
      ret.push_back(0);
    }
  }
  return ret;
}
