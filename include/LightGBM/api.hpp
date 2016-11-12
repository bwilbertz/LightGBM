#ifndef LIGHTGBM_API_H_
#define LIGHTGBM_API_H_
#include<cstdint>

#include <vector>
#include <functional>
#include <memory>

#include <LightGBM/dataset.h>
#include <LightGBM/boosting.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/metric.h>
#include <LightGBM/config.h>


/*!
* To avoid type conversion on large data, most of our expose interface support both for float_32 and float_64.
* Except following:
* 1. gradients and hessians.
* 2. Get current score for training data and validation
* The reason is because they are called frequently, the type-conversion on them maybe time cost.
*/

#include <LightGBM/export.h>

typedef void* DatasetHandle;
typedef void* BoosterHandle;

#define C_API_DTYPE_FLOAT32 (0)
#define C_API_DTYPE_FLOAT64 (1)
#define C_API_DTYPE_INT32   (2)
#define C_API_DTYPE_INT64   (3)

#define C_API_PREDICT_NORMAL     (0)
#define C_API_PREDICT_RAW_SCORE  (1)
#define C_API_PREDICT_LEAF_INDEX (2)
#define C_API_PREDICT_CONTRIB    (3)

namespace LightGBM {

class Booster {
public:
  explicit Booster(const char* filename)
      : train_data_(nullptr),
        boosting_(Boosting::CreateBoosting(filename)),
        objective_fun_(nullptr) {
  }

  explicit Booster()
      : train_data_(nullptr),
        boosting_(Boosting::CreateBoosting("gbdt", nullptr)),
        objective_fun_(nullptr) {
  }

  explicit Booster(const Dataset* train_data,
          const char* parameters);

  explicit Booster(const Dataset* train_data,
          std::unordered_map<std::string, std::string>& params);

  virtual ~Booster() {
  }

  void Init(const Dataset* train_data);

  void MergeFrom(const Booster* other);

  void CreateObjectiveAndMetrics();

  void ResetTrainingData(const Dataset* train_data);

  void ResetConfig(const char* parameters);

  void AddValidData(const Dataset* valid_data);

  bool TrainOneIter();

  bool TrainOneIter(const float* gradients, const float* hessians);

  void RollbackOneIter();

  void Predict(int num_iteration, int predict_type, int nrow,
               std::function<std::vector<std::pair<int, double>>(int row_idx)> get_row_fun,
               double* out_result, int64_t* out_len);

  void Predict(int num_iteration, int predict_type, int nrow,
               std::function<std::vector<std::pair<int, double>>(int row_idx)> get_row_fun,
               const IOConfig& config,
               double* out_result, int64_t* out_len);

  void Predict(int num_iteration, int predict_type, const char* data_filename,
               int data_has_header, const IOConfig& config,
               const char* result_filename);

  inline void GetPredictAt(int data_idx, double* out_result, int64_t* out_len) {
    boosting_->GetPredictAt(data_idx, out_result, out_len);
  }

  inline void SaveModelToFile(int num_iteration, const char* filename) {
    boosting_->SaveModelToFile(num_iteration, filename);
  }

  inline void LoadModelFromString(const char* model_str) {
    boosting_->LoadModelFromString(model_str);
  }

  inline std::string SaveModelToString(int num_iteration) {
    return boosting_->SaveModelToString(num_iteration);
  }

  inline std::string DumpModel(int num_iteration) {
    return boosting_->DumpModel(num_iteration);
  }

  inline std::vector<double> FeatureImportance(int num_iteration, int importance_type) {
    return boosting_->FeatureImportance(num_iteration, importance_type);
  }

  inline double GetLeafValue(int tree_idx, int leaf_idx) const {
    return dynamic_cast<GBDTBase*>(boosting_.get())->GetLeafValue(tree_idx, leaf_idx);
  }

  void SetLeafValue(int tree_idx, int leaf_idx, double val);

  int GetEvalCounts() const;

  int GetEvalNames(char** out_strs) const;

  int GetFeatureNames(char** out_strs) const;

  const inline Boosting* GetBoosting() const { return boosting_.get(); }

  inline int NumberOfClasses() const { return boosting_->NumberOfClasses(); }

  inline int NumberOfSubModels() const { return boosting_->NumberOfTotalModel(); }

private:

  const Dataset* train_data_;
  std::unique_ptr<Boosting> boosting_;
  /*! \brief All configs */
  OverallConfig config_;
  /*! \brief Metric for training data */
  std::vector<std::unique_ptr<Metric>> train_metric_;
  /*! \brief Metrics for validation data */
  std::vector<std::vector<std::unique_ptr<Metric>>> valid_metrics_;
  /*! \brief Training objective function */
  std::unique_ptr<ObjectiveFunction> objective_fun_;
  /*! \brief mutex for threading safe call */
  std::mutex mutex_;
};

}

LIGHTGBM_C_EXPORT LightGBM::Dataset* CreateDatasetFromMat(const void* data,
  int data_type,
  int32_t nrow,
  int32_t ncol,
  int is_row_major,
  const char* parameters,
  const DatasetHandle reference);

LIGHTGBM_C_EXPORT LightGBM::Dataset* CreateDatasetFromCSR(const void* indptr,
  int indptr_type,
  const int32_t* indices,
  const void* data,
  int data_type,
  int64_t nindptr,
  int64_t nelem,
  int64_t num_col,
  const char* parameters,
  const DatasetHandle reference);

LIGHTGBM_C_EXPORT LightGBM::Dataset* CreateDatasetFromCSC(const void* col_ptr,
  int col_ptr_type,
  const int32_t* indices,
  const void* data,
  int data_type,
  int64_t ncol_ptr,
  int64_t nelem,
  int64_t num_row,
  const char* parameters,
  const DatasetHandle reference);


// some help functions used to convert data
std::function<std::vector<double>(int row_idx)>
RowFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major);

std::function<std::vector<std::pair<int, double>>(int row_idx)>
RowPairFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major);

std::function<std::vector<std::pair<int, double>>(int idx)>
RowFunctionFromCSR(const void* indptr, int indptr_type, const int32_t* indices, 
  const void* data, int data_type, int64_t nindptr, int64_t nelem);

// Row iterator of on column for CSC matrix
class CSC_RowIterator {
public:
  CSC_RowIterator(const void* col_ptr, int col_ptr_type, const int32_t* indices,
                  const void* data, int data_type, int64_t ncol_ptr, int64_t nelem, int col_idx);
  ~CSC_RowIterator() {}
  // return value at idx, only can access by ascent order
  double Get(int idx);
  // return next non-zero pair, if index < 0, means no more data
  std::pair<int, double> NextNonZero();
private:
  int nonzero_idx_ = 0;
  int cur_idx_ = -1;
  double cur_val_ = 0.0f;
  bool is_end_ = false;
  std::function<std::pair<int, double>(int idx)> iter_fun_;
};

#endif // LIGHTGBM_API_H_
