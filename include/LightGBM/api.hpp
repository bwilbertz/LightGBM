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
#include <LightGBM/predictor.hpp>



/*!
* To avoid type conversion on large data, most of our expose interface support both for float_32 and float_64.
* Except following:
* 1. gradients and hessians. 
* 2. Get current score for training data and validation
* The reason is because they are called frequently, the type-conversion on them maybe time cost. 
*/

#ifdef __cplusplus
#define DLL_EXTERN_C extern "C"
#else
#define DLL_EXTERN_C
#endif

#ifdef _MSC_VER
#define DllExport DLL_EXTERN_C __declspec(dllexport)
#else
#define DllExport DLL_EXTERN_C
#endif

typedef void* DatesetHandle;
typedef void* BoosterHandle;

#define C_API_DTYPE_FLOAT32 (0)
#define C_API_DTYPE_FLOAT64 (1)
#define C_API_DTYPE_INT32   (2)
#define C_API_DTYPE_INT64   (3)

#define C_API_PREDICT_NORMAL     (0)
#define C_API_PREDICT_RAW_SCORE  (1)
#define C_API_PREDICT_LEAF_INDEX (2)

namespace LightGBM {

class Booster {
public:
  explicit Booster(const char* filename):
    boosting_(Boosting::CreateBoosting(filename)),
    train_data_(nullptr),
    objective_fun_(nullptr),
    predictor_(nullptr) {}

  explicit Booster(const Dataset* train_data,
    std::vector<const Dataset*> valid_data,
    std::vector<std::string> valid_names,
    const char* parameters);

  explicit Booster(const Dataset* train_data,
    std::vector<const Dataset*> valid_data,
    std::vector<std::string> valid_names,
    std::unordered_map<std::string, std::string>& params);


  virtual ~Booster();

  void Init(const Dataset* train_data, std::vector<const Dataset*> valid_data,
        std::vector<std::string> valid_names);

  bool TrainOneIter(bool is_eval = false);

  bool TrainOneIter(const float* gradients, const float* hessians, bool is_eval = false);

  void PrepareForPrediction(int num_used_model, int predict_type);

  std::vector<double> Predict(const std::vector<std::pair<int, double>>& features);

  void PredictForFile(const char* data_filename, const char* result_filename, bool data_has_header);

  void SaveModelToFile(int num_used_model, const char* filename);

  const inline Boosting* GetBoosting() const { return boosting_.get(); }

  const inline float* GetTrainingScore(int* out_len) const { return boosting_->GetTrainingScore(out_len); }

  const inline int NumberOfClasses() const { return boosting_->NumberOfClasses(); }

private:

  std::unique_ptr<Boosting> boosting_;
  /*! \brief All configs */
  OverallConfig config_;
  /*! \brief Training data */
  const Dataset* train_data_;
  /*! \brief Validation data */
  std::vector<const Dataset*> valid_datas_;
  /*! \brief Metric for training data */
  std::vector<std::unique_ptr<Metric>> train_metric_;
  /*! \brief Metrics for validation data */
  std::vector<std::vector<std::unique_ptr<Metric>>>valid_metrics_;
  /*! \brief Training objective function */
  std::unique_ptr<ObjectiveFunction> objective_fun_;
  /*! \brief Using predictor for prediction task */
  std::unique_ptr<Predictor> predictor_;

};

}

DllExport LightGBM::Dataset* CreateDatasetFromMat(const void* data,
  int data_type,
  int32_t nrow,
  int32_t ncol,
  int is_row_major,
  const char* parameters,
  const DatesetHandle* reference);

DllExport LightGBM::Dataset* CreateDatasetFromCSR(const void* indptr,
  int indptr_type,
  const int32_t* indices,
  const void* data,
  int data_type,
  int64_t nindptr,
  int64_t nelem,
  int64_t num_col,
  const char* parameters,
  const DatesetHandle* reference);

DllExport LightGBM::Dataset* CreateDatasetFromCSC(const void* col_ptr,
  int col_ptr_type,
  const int32_t* indices,
  const void* data,
  int data_type,
  int64_t ncol_ptr,
  int64_t nelem,
  int64_t num_row,
  const char* parameters,
  const DatesetHandle* reference);


// some help functions used to convert data

std::function<std::vector<double>(int row_idx)>
RowFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major);

std::function<std::vector<std::pair<int, double>>(int row_idx)>
RowPairFunctionFromDenseMatric(const void* data, int num_row, int num_col, int data_type, int is_row_major);

std::function<std::vector<std::pair<int, double>>(int idx)>
RowFunctionFromCSR(const void* indptr, int indptr_type, const int32_t* indices, 
  const void* data, int data_type, int64_t nindptr, int64_t nelem);

std::function<std::vector<std::pair<int, double>>(int idx)>
ColumnFunctionFromCSC(const void* col_ptr, int col_ptr_type, const int32_t* indices, 
  const void* data, int data_type, int64_t ncol_ptr, int64_t nelem);

std::vector<double> 
SampleFromOneColumn(const std::vector<std::pair<int, double>>& data, const std::vector<int>& indices);

#endif // LIGHTGBM_API_H_
