#include <LightGBM/tree_learner.h>

#include "serial_tree_learner.h"
#ifndef NO_NETWORK
#include "parallel_tree_learner.h"
#endif

namespace LightGBM {

TreeLearner* TreeLearner::CreateTreeLearner(TreeLearnerType type, const TreeConfig& tree_config) {
#ifdef NO_NETWORK
  if (type == TreeLearnerType::kSerialTreeLearner) {
      return new SerialTreeLearner(tree_config);
  }
  Log::Fatal("LightGBM was not compiled for tree learner type %s", type);
#else
  if (type == TreeLearnerType::kSerialTreeLearner) {
    return new SerialTreeLearner(tree_config);
  } else if (type == TreeLearnerType::kFeatureParallelTreelearner) {
    return new FeatureParallelTreeLearner(tree_config);
  } else if (type == TreeLearnerType::kDataParallelTreeLearner) {
    return new DataParallelTreeLearner(tree_config);
  }
#endif
  return nullptr;
}

}  // namespace LightGBM
