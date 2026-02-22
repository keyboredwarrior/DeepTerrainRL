#pragma once

#include <memory>
#include <string>

#include "IOptimizer.h"
#include "ITrainerBackend.h"

class cNNSolver
{
public:
	static void BuildSolver(const std::string& solver_file, std::shared_ptr<cNNSolver>& out_solver);
	static void BuildSolverAsync(const std::string& solver_file, std::shared_ptr<cNNSolver>& out_solver);

	cNNSolver();
	~cNNSolver();

	void SetOptimizer(const tOptimizerPtr& optimizer);
	void SetTrainerBackend(const tTrainerBackendPtr& backend);

	void ApplySteps(int steps);
	double ForwardBackward();
	void Reset();
	void ZeroGrad();
	void Update();

	tOptimizerPtr GetOptimizer() const;
	tTrainerBackendPtr GetTrainerBackend() const;

private:
	tOptimizerPtr mOptimizer;
	tTrainerBackendPtr mBackend;
};
