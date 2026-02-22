#include "NNSolver.h"

#include <cassert>

#include "CaffeBackend.h"

void cNNSolver::BuildSolver(const std::string& solver_file, std::shared_ptr<cNNSolver>& out_solver)
{
	out_solver = std::shared_ptr<cNNSolver>(new cNNSolver());
	auto optimizer = BuildCaffeOptimizer(solver_file, false);
	out_solver->SetOptimizer(optimizer);
}

void cNNSolver::BuildSolverAsync(const std::string& solver_file, std::shared_ptr<cNNSolver>& out_solver)
{
	out_solver = std::shared_ptr<cNNSolver>(new cNNSolver());
	auto optimizer = BuildCaffeOptimizer(solver_file, true);
	out_solver->SetOptimizer(optimizer);
}

cNNSolver::cNNSolver()
{
}

cNNSolver::~cNNSolver()
{
}

void cNNSolver::SetOptimizer(const tOptimizerPtr& optimizer)
{
	mOptimizer = optimizer;
}

void cNNSolver::SetTrainerBackend(const tTrainerBackendPtr& backend)
{
	mBackend = backend;
}

void cNNSolver::ApplySteps(int steps)
{
	assert(mOptimizer != nullptr);
	mOptimizer->Step(steps);
}

double cNNSolver::ForwardBackward()
{
	if (mBackend != nullptr)
	{
		return mBackend->ForwardBackward();
	}
	assert(mOptimizer != nullptr);
	return mOptimizer->ForwardBackward();
}

void cNNSolver::Reset()
{
	if (mOptimizer != nullptr)
	{
		mOptimizer->Reset();
	}
}

void cNNSolver::ZeroGrad()
{
	if (mOptimizer != nullptr)
	{
		mOptimizer->ZeroGrad();
	}
}

void cNNSolver::Update()
{
	if (mOptimizer != nullptr)
	{
		mOptimizer->Update();
	}
}

tOptimizerPtr cNNSolver::GetOptimizer() const
{
	return mOptimizer;
}

tTrainerBackendPtr cNNSolver::GetTrainerBackend() const
{
	return mBackend;
}
