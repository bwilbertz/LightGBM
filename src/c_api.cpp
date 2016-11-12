#include <omp.h>

#include <LightGBM/utils/common.h>
#include <LightGBM/utils/random.h>
#include <LightGBM/c_api.h>
#include <LightGBM/api.hpp>
#include <LightGBM/dataset_loader.h>
#include <LightGBM/dataset.h>
#include <LightGBM/boosting.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/metric.h>
#include <LightGBM/config.h>
#include <LightGBM/predictor.hpp>

#include <cstdio>
#include <string>
#include <cstring>


using namespace LightGBM;

DllExport const char* LGBM_GetLastError() {
  return LastErrorMsg().c_str();
}

DllExport int LGBM_CreateDatasetFromFile(const char* filename,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out) {
  API_BEGIN();
  OverallConfig config;
  config.LoadFromString(parameters);
  DatasetLoader loader(config.io_config, nullptr);
  loader.SetHeader(filename);
  if (reference == nullptr) {
    *out = loader.LoadFromFile(filename);
  } else {
    *out = loader.LoadFromFileAlignWithOtherDataset(filename,
      reinterpret_cast<const Dataset*>(*reference));
  }
  API_END();
}

DllExport int LGBM_CreateDatasetFromBinaryFile(const char* filename,
  DatesetHandle* out) {
  API_BEGIN();
  OverallConfig config;
  DatasetLoader loader(config.io_config, nullptr);
  *out = loader.LoadFromBinFile(filename, 0, 1);
  API_END();
}

DllExport int LGBM_CreateDatasetFromMat(const void* data,
  int data_type,
  int32_t nrow,
  int32_t ncol,
  int is_row_major,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out) {
  API_BEGIN();
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
    ret->CopyFeatureMapperFrom(
      reinterpret_cast<const Dataset*>(*reference),
      config.io_config.is_enable_sparse);
  }

#pragma omp parallel for schedule(guided)
  for (int i = 0; i < nrow; ++i) {
    const int tid = omp_get_thread_num();
    auto one_row = get_row_fun(i);
    ret->PushOneRow(tid, i, one_row);
  }
  ret->FinishLoad();
  *out = ret.release();
  API_END();
}

DllExport int LGBM_CreateDatasetFromCSR(const void* indptr,
  int indptr_type,
  const int32_t* indices,
  const void* data,
  int data_type,
  int64_t nindptr,
  int64_t nelem,
  int64_t num_col,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out) {
  API_BEGIN();
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
    ret->CopyFeatureMapperFrom(
      reinterpret_cast<const Dataset*>(*reference),
      config.io_config.is_enable_sparse);
  }

#pragma omp parallel for schedule(guided)
  for (int i = 0; i < nindptr - 1; ++i) {
    const int tid = omp_get_thread_num();
    auto one_row = get_row_fun(i);
    ret->PushOneRow(tid, i, one_row);
  }
  ret->FinishLoad();
  *out = ret.release();
  API_END();
}

DllExport int LGBM_CreateDatasetFromCSC(const void* col_ptr,
  int col_ptr_type,
  const int32_t* indices,
  const void* data,
  int data_type,
  int64_t ncol_ptr,
  int64_t nelem,
  int64_t num_row,
  const char* parameters,
  const DatesetHandle* reference,
  DatesetHandle* out) {
  API_BEGIN();
  OverallConfig config;
  config.LoadFromString(parameters);
  DatasetLoader loader(config.io_config, nullptr);
  std::unique_ptr<Dataset> ret;
  auto get_col_fun = ColumnFunctionFromCSC(col_ptr, col_ptr_type, indices, data, data_type, ncol_ptr, nelem);
  int32_t nrow = static_cast<int32_t>(num_row);
  if (reference == nullptr) {
    Log::Warning("Construct from CSC format is not efficient");
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
    ret->CopyFeatureMapperFrom(
      reinterpret_cast<const Dataset*>(*reference),
      config.io_config.is_enable_sparse);
  }

#pragma omp parallel for schedule(guided)
  for (int i = 0; i < ncol_ptr - 1; ++i) {
    const int tid = omp_get_thread_num();
    auto one_col = get_col_fun(i);
    ret->PushOneColumn(tid, i, one_col);
  }
  ret->FinishLoad();
  *out = ret.release();
  API_END();
}

DllExport int LGBM_DatasetFree(DatesetHandle handle) {
  API_BEGIN();
  delete reinterpret_cast<Dataset*>(handle);
  API_END();
}

DllExport int LGBM_DatasetSaveBinary(DatesetHandle handle,
  const char* filename) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  dataset->SaveBinaryFile(filename);
  API_END();
}

DllExport int LGBM_DatasetSetField(DatesetHandle handle,
  const char* field_name,
  const void* field_data,
  int64_t num_element,
  int type) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  bool is_success = false;
  if (type == C_API_DTYPE_FLOAT32) {
    is_success = dataset->SetFloatField(field_name, reinterpret_cast<const float*>(field_data), static_cast<int32_t>(num_element));
  } else if (type == C_API_DTYPE_INT32) {
    is_success = dataset->SetIntField(field_name, reinterpret_cast<const int*>(field_data), static_cast<int32_t>(num_element));
  }
  if (!is_success) { throw std::runtime_error("Input data type erorr or field not found"); }
  API_END();
}

DllExport int LGBM_DatasetGetField(DatesetHandle handle,
  const char* field_name,
  int64_t* out_len,
  const void** out_ptr,
  int* out_type) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  bool is_success = false;
  if (dataset->GetFloatField(field_name, out_len, reinterpret_cast<const float**>(out_ptr))) {
    *out_type = C_API_DTYPE_FLOAT32;
    is_success = true;
  } else if (dataset->GetIntField(field_name, out_len, reinterpret_cast<const int**>(out_ptr))) {
    *out_type = C_API_DTYPE_INT32;
    is_success = true;
  }
  if (!is_success) { throw std::runtime_error("Field not found"); }
  API_END();
}

DllExport int LGBM_DatasetGetNumData(DatesetHandle handle,
  int64_t* out) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  *out = dataset->num_data();
  API_END();
}

DllExport int LGBM_DatasetGetNumFeature(DatesetHandle handle,
  int64_t* out) {
  API_BEGIN();
  auto dataset = reinterpret_cast<Dataset*>(handle);
  *out = dataset->num_total_features();
  API_END();
}


// ---- start of booster

DllExport int LGBM_BoosterCreate(const DatesetHandle train_data,
  const DatesetHandle valid_datas[],
  const char* valid_names[],
  int n_valid_datas,
  const char* parameters,
  BoosterHandle* out) {
  API_BEGIN();
  const Dataset* p_train_data = reinterpret_cast<const Dataset*>(train_data);
  std::vector<const Dataset*> p_valid_datas;
  std::vector<std::string> p_valid_names;
  for (int i = 0; i < n_valid_datas; ++i) {
    p_valid_datas.emplace_back(reinterpret_cast<const Dataset*>(valid_datas[i]));
    p_valid_names.emplace_back(valid_names[i]);
  }
  *out = new Booster(p_train_data, p_valid_datas, p_valid_names, parameters);
  API_END();
}

DllExport int LGBM_BoosterLoadFromModelfile(
  const char* filename,
  BoosterHandle* out) {
  API_BEGIN();
  *out = new Booster(filename);
  API_END();
}

DllExport int LGBM_BoosterFree(BoosterHandle handle) {
  API_BEGIN();
  delete reinterpret_cast<Booster*>(handle);
  API_END();
}

DllExport int LGBM_BoosterUpdateOneIter(BoosterHandle handle, int* is_finished) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  if (ref_booster->TrainOneIter()) {
    *is_finished = 1;
  } else {
    *is_finished = 0;
  }
  API_END();
}

DllExport int LGBM_BoosterUpdateOneIterCustom(BoosterHandle handle,
  const float* grad,
  const float* hess,
  int* is_finished) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  if (ref_booster->TrainOneIter(grad, hess)) {
    *is_finished = 1;
  } else {
    *is_finished = 0;
  }
  API_END();
}

DllExport int LGBM_BoosterEval(BoosterHandle handle,
  int data,
  int64_t* out_len,
  float* out_results) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  auto boosting = ref_booster->GetBoosting();
  auto result_buf = boosting->GetEvalAt(data);
  *out_len = static_cast<int64_t>(result_buf.size());
  for (size_t i = 0; i < result_buf.size(); ++i) {
    (out_results)[i] = static_cast<float>(result_buf[i]);
  }
  API_END();
}

DllExport int LGBM_BoosterGetScore(BoosterHandle handle,
  int64_t* out_len,
  const float** out_result) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  int len = 0;
  *out_result = ref_booster->GetTrainingScore(&len);
  *out_len = static_cast<int64_t>(len);
  API_END();
}

DllExport int LGBM_BoosterGetPredict(BoosterHandle handle,
  int data,
  int64_t* out_len,
  float* out_result) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  auto boosting = ref_booster->GetBoosting();
  int len = 0;
  boosting->GetPredictAt(data, out_result, &len);
  *out_len = static_cast<int64_t>(len);
  API_END();
}

DllExport int LGBM_BoosterPredictForFile(BoosterHandle handle,
  int predict_type,
  int64_t n_used_trees,
  int data_has_header,
  const char* data_filename,
  const char* result_filename) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->PrepareForPrediction(static_cast<int>(n_used_trees), predict_type);
  bool bool_data_has_header = data_has_header > 0 ? true : false;
  ref_booster->PredictForFile(data_filename, result_filename, bool_data_has_header);
  API_END();
}

DllExport int LGBM_BoosterPredictForCSR(BoosterHandle handle,
  const void* indptr,
  int indptr_type,
  const int32_t* indices,
  const void* data,
  int data_type,
  int64_t nindptr,
  int64_t nelem,
  int64_t,
  int predict_type,
  int64_t n_used_trees,
  double* out_result) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->PrepareForPrediction(static_cast<int>(n_used_trees), predict_type);

  auto get_row_fun = RowFunctionFromCSR(indptr, indptr_type, indices, data, data_type, nindptr, nelem);
  int num_class = ref_booster->NumberOfClasses();
  int nrow = static_cast<int>(nindptr - 1);
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < nrow; ++i) {
    auto one_row = get_row_fun(i);
    auto predicton_result = ref_booster->Predict(one_row);
    for (int j = 0; j < num_class; ++j) {
      out_result[i * num_class + j] = predicton_result[j];
    }
  }
  API_END();
}

DllExport int LGBM_BoosterPredictForMat(BoosterHandle handle,
  const void* data,
  int data_type,
  int32_t nrow,
  int32_t ncol,
  int is_row_major,
  int predict_type,
  int64_t n_used_trees,
  double* out_result) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->PrepareForPrediction(static_cast<int>(n_used_trees), predict_type);

  auto get_row_fun = RowPairFunctionFromDenseMatric(data, nrow, ncol, data_type, is_row_major);
  int num_class = ref_booster->NumberOfClasses();
#pragma omp parallel for schedule(guided)
  for (int i = 0; i < nrow; ++i) {
    auto one_row = get_row_fun(i);
    auto predicton_result = ref_booster->Predict(one_row);
    for (int j = 0; j < num_class; ++j) {
      out_result[i * num_class + j] = predicton_result[j];
    }
  }
  API_END();
}

DllExport int LGBM_BoosterSaveModel(BoosterHandle handle,
  int num_used_model,
  const char* filename) {
  API_BEGIN();
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->SaveModelToFile(num_used_model, filename);
  API_END();
}

