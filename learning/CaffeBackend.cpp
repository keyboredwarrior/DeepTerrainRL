#include "CaffeBackend.h"

#include <cassert>
#include <cmath>
#include <cstring>

#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <caffe/net.hpp>
#include <caffe/sgd_solvers.hpp>

#include "util/MathUtil.h"

class cCaffeNeuralModel : public INeuralModel
{
public:
	void LoadNet(const std::string& net_file) override
	{
		mNet = std::unique_ptr<caffe::Net<tNNData>>(new caffe::Net<tNNData>(net_file, caffe::TEST));
	}

	void LoadModel(const std::string& model_file) override
	{
		assert(mNet != nullptr);
		mNet->CopyTrainedLayersFromHDF5(model_file);
	}

	void SaveModel(const std::string& out_file) const override
	{
		assert(mNet != nullptr);
		mNet->ToHDF5(out_file);
	}

	void Eval(const Eigen::VectorXd& x, Eigen::VectorXd& out_y) const override
	{
		assert(mNet != nullptr);
		caffe::Blob<tNNData> blob(1, 1, 1, static_cast<int>(x.size()));
		auto* data = blob.mutable_cpu_data();
		for (int i = 0; i < blob.count(); ++i) data[i] = x[i];
		mNet->input_blobs()[0]->CopyFrom(blob);
		const auto& out = mNet->Forward();
		auto* out_blob = out[0];
		out_y.resize(out_blob->count());
		for (int i = 0; i < out_blob->count(); ++i) out_y[i] = out_blob->cpu_data()[i];
	}

	void EvalBatch(const Eigen::MatrixXd& X, Eigen::MatrixXd& out_Y) const override
	{
		out_Y.resize(X.rows(), GetOutputSize());
		Eigen::VectorXd y;
		for (int i = 0; i < X.rows(); ++i)
		{
			Eval(X.row(i), y);
			out_Y.row(i) = y;
		}
	}

	void Backward(const Eigen::VectorXd& y_diff, Eigen::VectorXd& out_x_diff) const override
	{
		assert(mNet != nullptr);
		const auto& top = mNet->top_vecs().back()[0];
		auto* top_diff = top->mutable_cpu_diff();
		for (int i = 0; i < y_diff.size(); ++i) top_diff[i] = y_diff[i];
		mNet->ClearParamDiffs();
		mNet->Backward();
		const auto& bottom = mNet->bottom_vecs()[0][0];
		out_x_diff.resize(bottom->count());
		for (int i = 0; i < bottom->count(); ++i) out_x_diff[i] = bottom->cpu_diff()[i];
	}

	int GetInputSize() const override { return (mNet == nullptr) ? 0 : mNet->input_blobs()[0]->count(); }
	int GetOutputSize() const override { return (mNet == nullptr) ? 0 : mNet->output_blobs()[0]->count(); }
	int GetBatchSize() const override { return 1; }
	int CalcNumParams() const override
	{
		if (mNet == nullptr) return 0;
		int total = 0;
		for (auto* b : mNet->learnable_params()) total += b->count();
		return total;
	}

	void GetParams(std::vector<tNNData>& out_params) const override
	{
		out_params.clear();
		if (mNet == nullptr) return;
		for (auto* b : mNet->learnable_params())
		{
			auto* d = b->cpu_data();
			out_params.insert(out_params.end(), d, d + b->count());
		}
	}

	void SetParams(const std::vector<tNNData>& params) override
	{
		if (mNet == nullptr) return;
		size_t offset = 0;
		for (auto* b : mNet->learnable_params())
		{
			auto* d = b->mutable_cpu_data();
			for (int i = 0; i < b->count(); ++i) d[i] = params[offset + i];
			offset += b->count();
		}
	}

	void BlendParams(const std::vector<tNNData>& params, double this_weight, double other_weight) override
	{
		if (mNet == nullptr) return;
		size_t offset = 0;
		for (auto* b : mNet->learnable_params())
		{
			auto* d = b->mutable_cpu_data();
			for (int i = 0; i < b->count(); ++i) d[i] = this_weight * d[i] + other_weight * params[offset + i];
			offset += b->count();
		}
	}

	bool CompareParams(const std::vector<tNNData>& params) const override
	{
		if (mNet == nullptr) return params.empty();
		size_t offset = 0;
		for (auto* b : mNet->learnable_params())
		{
			auto* d = b->cpu_data();
			for (int i = 0; i < b->count(); ++i)
			{
				if (d[i] != params[offset + i]) return false;
			}
			offset += b->count();
		}
		return true;
	}

	bool HasLayer(const std::string& layer_name) const override
	{
		return mNet != nullptr && mNet->has_blob(layer_name) && mNet->has_layer(layer_name);
	}

	void ForwardInjectNoisePrefilled(double mean, double stdev, const std::string& layer_name, Eigen::VectorXd& out_y) const override
	{
		auto blob = mNet->blob_by_name(layer_name);
		auto* data = blob->mutable_cpu_data();
		for (int i = 0; i < blob->count(); ++i) data[i] += cMathUtil::RandDoubleNorm(mean, stdev);
		mNet->ForwardFrom(1);
		auto out_blob = mNet->output_blobs()[0];
		out_y.resize(out_blob->count());
		for (int i = 0; i < out_blob->count(); ++i) out_y[i] = out_blob->cpu_data()[i];
	}

	void GetLayerState(const std::string& layer_name, Eigen::VectorXd& out_state) const override
	{
		auto blob = mNet->blob_by_name(layer_name);
		out_state.resize(blob->count());
		for (int i = 0; i < blob->count(); ++i) out_state[i] = blob->cpu_data()[i];
	}

	void SetLayerState(const Eigen::VectorXd& state, const std::string& layer_name) const override
	{
		auto blob = mNet->blob_by_name(layer_name);
		auto* data = blob->mutable_cpu_data();
		for (int i = 0; i < state.size(); ++i) data[i] = state[i];
	}

private:
	std::unique_ptr<caffe::Net<tNNData>> mNet;
};

class cCaffeOptimizer : public IOptimizer, public ITrainerBackend
{
public:
	explicit cCaffeOptimizer(const std::string& solver_file, bool async)
		: mAsync(async)
	{
		caffe::SolverParameter param;
		caffe::ReadProtoFromTextFileOrDie(solver_file, &param);
		caffe::Caffe::set_mode(caffe::Caffe::CPU);
		switch (param.solver_type())
		{
		case caffe::SolverParameter_SolverType_SGD: mSolver.reset(new caffe::SGDSolver<double>(param)); break;
		case caffe::SolverParameter_SolverType_NESTEROV: mSolver.reset(new caffe::NesterovSolver<double>(param)); break;
		case caffe::SolverParameter_SolverType_ADAGRAD: mSolver.reset(new caffe::AdaGradSolver<double>(param)); break;
		case caffe::SolverParameter_SolverType_RMSPROP: mSolver.reset(new caffe::RMSPropSolver<double>(param)); break;
		case caffe::SolverParameter_SolverType_ADADELTA: mSolver.reset(new caffe::AdaDeltaSolver<double>(param)); break;
		case caffe::SolverParameter_SolverType_ADAM: mSolver.reset(new caffe::AdamSolver<double>(param)); break;
		default: LOG(FATAL) << "Unknown SolverType";
		}
	}

	void Step(int steps) override
	{
		if (!mAsync)
		{
			mSolver->Step(steps);
			return;
		}
		const int stop_iter = mSolver->iter() + steps;
		while (mSolver->iter() < stop_iter)
		{
			mSolver->ApplyUpdate();
			mSolver->set_iter(mSolver->iter() + 1);
		}
	}

	void Reset() override {}
	void ZeroGrad() override { mSolver->net()->ClearParamDiffs(); }
	void Update() override { mSolver->ApplyUpdate(); }
	double ForwardBackward() override { mSolver->net()->ClearParamDiffs(); return mSolver->net()->ForwardBackward(); }

	double TrainStep(int iters) override { Step(iters); return 0.0; }
	void IngestData(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) override
	{
		auto data_layer = boost::static_pointer_cast<caffe::MemoryDataLayer<double>>(mSolver->net()->layer_by_name("data"));
		int batch_size = data_layer->batch_size();
		int data_dim = static_cast<int>(X.cols());
		int label_dim = static_cast<int>(Y.cols());
		std::vector<double> data(batch_size * data_dim, 0.0);
		std::vector<double> labels(batch_size * label_dim, 0.0);
		for (int i = 0; i < batch_size && i < X.rows(); ++i)
		{
			for (int j = 0; j < data_dim; ++j) data[i * data_dim + j] = X(i, j);
			for (int j = 0; j < label_dim; ++j) labels[i * label_dim + j] = Y(i, j);
		}
		data_layer->AddData(data, labels);
	}

private:
	bool mAsync;
	std::unique_ptr<caffe::Solver<double>> mSolver;
};

std::shared_ptr<INeuralModel> BuildCaffeNeuralModel()
{
	return std::shared_ptr<INeuralModel>(new cCaffeNeuralModel());
}

std::shared_ptr<IOptimizer> BuildCaffeOptimizer(const std::string& solver_file, bool async)
{
	return std::shared_ptr<IOptimizer>(new cCaffeOptimizer(solver_file, async));
}
