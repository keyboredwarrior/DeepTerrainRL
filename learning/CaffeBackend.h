#pragma once

#include <memory>
#include <string>

#include "INeuralModel.h"
#include "IOptimizer.h"

std::shared_ptr<INeuralModel> BuildCaffeNeuralModel();
std::shared_ptr<IOptimizer> BuildCaffeOptimizer(const std::string& solver_file, bool async);
