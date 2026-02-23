#pragma once
#include "NeuralNet.h"
#include <pytorch/net.hpp>
#include <memory>
#include <string>

class cOptimizerExecutor
{
public:
	struct tConfig
	{
		tConfig();
		std::string mBackend;
		std::string mOptimizer;
		std::string mPolicyCheckpoint;

		bool IsValid() const;
	};

	static void BuildExecutor(const std::string& config_file, std::shared_ptr<cOptimizerExecutor>& out_executor);
	virtual ~cOptimizerExecutor();

	virtual boost::shared_ptr<pytorch::Net<cNeuralNet::tNNData>> GetNet() = 0;
	virtual void ApplySteps(int steps) = 0;
	virtual cNeuralNet::tNNData ForwardBackward() = 0;

protected:
	cOptimizerExecutor();
};

class cNNSolver : public cOptimizerExecutor
{
public:
	using cOptimizerExecutor::BuildExecutor;
};

