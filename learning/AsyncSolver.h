#pragma once

#include "NNSolver.h"

// Async optimizer execution is intentionally managed outside of the model class.
// Keep this alias header so existing includes continue to build while migrating
// actor/learner orchestration to dedicated worker threads or loops.
using cAsyncOptimizerExecutor = cOptimizerExecutor;
