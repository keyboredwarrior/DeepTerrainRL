#pragma once

#include <memory>
#include <Eigen/Dense>

class ITrainerBackend
{
public:
	virtual ~ITrainerBackend() = default;

	virtual double TrainStep(int iters) = 0;
	virtual double ForwardBackward() = 0;
	virtual void IngestData(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) = 0;
};

using tTrainerBackendPtr = std::shared_ptr<ITrainerBackend>;
