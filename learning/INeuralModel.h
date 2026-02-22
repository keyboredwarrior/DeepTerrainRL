#pragma once

#include <memory>
#include <string>
#include <vector>
#include <Eigen/Dense>

class INeuralModel
{
public:
	using tNNData = double;
	virtual ~INeuralModel() = default;

	virtual void LoadNet(const std::string& net_file) = 0;
	virtual void LoadModel(const std::string& model_file) = 0;
	virtual void SaveModel(const std::string& out_file) const = 0;

	virtual void Eval(const Eigen::VectorXd& x, Eigen::VectorXd& out_y) const = 0;
	virtual void EvalBatch(const Eigen::MatrixXd& X, Eigen::MatrixXd& out_Y) const = 0;
	virtual void Backward(const Eigen::VectorXd& y_diff, Eigen::VectorXd& out_x_diff) const = 0;

	virtual int GetInputSize() const = 0;
	virtual int GetOutputSize() const = 0;
	virtual int GetBatchSize() const = 0;
	virtual int CalcNumParams() const = 0;

	virtual void GetParams(std::vector<tNNData>& out_params) const = 0;
	virtual void SetParams(const std::vector<tNNData>& params) = 0;
	virtual void BlendParams(const std::vector<tNNData>& params, double this_weight, double other_weight) = 0;
	virtual bool CompareParams(const std::vector<tNNData>& params) const = 0;

	virtual bool HasLayer(const std::string& layer_name) const = 0;
	virtual void ForwardInjectNoisePrefilled(double mean, double stdev, const std::string& layer_name, Eigen::VectorXd& out_y) const = 0;
	virtual void GetLayerState(const std::string& layer_name, Eigen::VectorXd& out_state) const = 0;
	virtual void SetLayerState(const Eigen::VectorXd& state, const std::string& layer_name) const = 0;
};

using tNeuralModelPtr = std::shared_ptr<INeuralModel>;
