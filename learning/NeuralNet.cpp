#include "NeuralNet.h"

#include <json/json.h>

#include "CaffeBackend.h"
#include "NNSolver.h"
#include "ITrainerBackend.h"
#include "util/FileUtil.h"
#include "util/JsonUtil.h"

const std::string gInputOffsetKey = "InputOffset";
const std::string gInputScaleKey = "InputScale";
const std::string gOutputOffsetKey = "OutputOffset";
const std::string gOutputScaleKey = "OutputScale";

std::mutex cNeuralNet::gOutputLock;

cNeuralNet::tProblem::tProblem()
{
	mX.resize(0, 0);
	mY.resize(0, 0);
	mPassesPerStep = 100;
}

bool cNeuralNet::tProblem::HasData() const
{
	return mX.size() > 0;
}

cNeuralNet::cNeuralNet()
{
	Clear();
	mAsync = false;
}

cNeuralNet::~cNeuralNet()
{
}

void cNeuralNet::LoadNet(const std::string& net_file)
{
	if (net_file != "")
	{
		if (!HasNet())
		{
			mModel = BuildCaffeNeuralModel();
		}
		mModel->LoadNet(net_file);

		if (!ValidOffsetScale())
		{
			InitOffsetScale();
		}
	}
}

void cNeuralNet::LoadModel(const std::string& model_file)
{
	if (model_file != "")
	{
		if (!HasNet())
		{
			mModel = BuildCaffeNeuralModel();
		}
		mModel->LoadModel(model_file);
		LoadScale(GetOffsetScaleFile(model_file));
		SyncSolverParams();
		mValidModel = true;
	}
}

void cNeuralNet::LoadSolver(const std::string& solver_file, bool async)
{
	if (solver_file != "")
	{
		mSolverFile = solver_file;
		mAsync = async;

		if (mAsync)
		{
			cNNSolver::BuildSolverAsync(solver_file, mSolver);
		}
		else
		{
			cNNSolver::BuildSolver(solver_file, mSolver);
		}

		if (!HasNet())
		{
			mModel = BuildCaffeNeuralModel();
		}
		auto backend = std::dynamic_pointer_cast<ITrainerBackend>(mSolver->GetOptimizer());
		mSolver->SetTrainerBackend(backend);
		if (!ValidOffsetScale())
		{
			InitOffsetScale();
		}
		SyncSolverParams();
	}
}

void cNeuralNet::LoadScale(const std::string& scale_file)
{
	std::ifstream f_stream(scale_file);
	Json::Reader reader;
	Json::Value root;
	bool succ = reader.parse(f_stream, root);
	f_stream.close();

	int input_size = GetInputSize();
	if (succ && !root[gInputOffsetKey].isNull())
	{
		Eigen::VectorXd offset;
		succ &= cJsonUtil::ReadVectorJson(root[gInputOffsetKey], offset);
		if (offset.size() == input_size) mInputOffset = offset;
	}
	if (succ && !root[gInputScaleKey].isNull())
	{
		Eigen::VectorXd scale;
		succ &= cJsonUtil::ReadVectorJson(root[gInputScaleKey], scale);
		if (scale.size() == input_size) mInputScale = scale;
	}

	int output_size = GetOutputSize();
	if (succ && !root[gOutputOffsetKey].isNull())
	{
		Eigen::VectorXd offset;
		succ &= cJsonUtil::ReadVectorJson(root[gOutputOffsetKey], offset);
		if (offset.size() == output_size) mOutputOffset = offset;
	}
	if (succ && !root[gOutputScaleKey].isNull())
	{
		Eigen::VectorXd scale;
		succ &= cJsonUtil::ReadVectorJson(root[gOutputScaleKey], scale);
		if (scale.size() == output_size) mOutputScale = scale;
	}
}

void cNeuralNet::Clear()
{
	mModel.reset();
	mSolver.reset();
	mValidModel = false;
	mGradBuffer.clear();
	mInputOffset.resize(0);
	mInputScale.resize(0);
	mOutputOffset.resize(0);
	mOutputScale.resize(0);
}

void cNeuralNet::Train(const tProblem& prob)
{
	if (!HasSolver()) return;
	LoadTrainData(prob.mX, prob.mY);
	int batch_size = GetBatchSize();
	int num_batches = static_cast<int>(prob.mX.rows()) / std::max(1, batch_size);
	StepSolver(prob.mPassesPerStep * num_batches);
}

double cNeuralNet::ForwardBackward(const tProblem& prob)
{
	if (!HasSolver()) return 0;
	LoadTrainData(prob.mX, prob.mY);
	return mSolver->ForwardBackward();
}

void cNeuralNet::StepSolver(int iters)
{
	mSolver->ApplySteps(iters);
	SyncNetParams();
	mValidModel = true;
}

void cNeuralNet::ResetSolver()
{
	mSolver.reset();
	LoadSolver(mSolverFile, mAsync);
}

void cNeuralNet::CalcOffsetScale(const Eigen::MatrixXd& X, Eigen::VectorXd& out_offset, Eigen::VectorXd& out_scale) const
{
	int num_pts = static_cast<int>(X.rows());
	double norm = 1.0 / num_pts;
	const int input_size = GetInputSize();
	out_offset = Eigen::VectorXd::Zero(input_size);
	out_scale = Eigen::VectorXd::Zero(input_size);
	for (int i = 0; i < num_pts; ++i) out_offset += norm * X.row(i);
	for (int i = 0; i < num_pts; ++i)
	{
		Eigen::VectorXd curr_x = X.row(i);
		curr_x -= out_offset;
		out_scale += norm * curr_x.cwiseProduct(curr_x);
	}
	out_offset = -out_offset;
	out_scale = out_scale.cwiseSqrt();
	for (int i = 0; i < out_scale.size(); ++i) out_scale[i] = (out_scale[i] == 0) ? 0 : (1 / out_scale[i]);
}

void cNeuralNet::SetInputOffsetScale(const Eigen::VectorXd& offset, const Eigen::VectorXd& scale){mInputOffset = offset; mInputScale = scale;}
void cNeuralNet::SetOutputOffsetScale(const Eigen::VectorXd& offset, const Eigen::VectorXd& scale){mOutputOffset = offset; mOutputScale = scale;}
const Eigen::VectorXd& cNeuralNet::GetInputOffset() const{return mInputOffset;}
const Eigen::VectorXd& cNeuralNet::GetInputScale() const{return mInputScale;}
const Eigen::VectorXd& cNeuralNet::GetOutputOffset() const{return mOutputOffset;}
const Eigen::VectorXd& cNeuralNet::GetOutputScale() const{return mOutputScale;}

void cNeuralNet::Eval(const Eigen::VectorXd& x, Eigen::VectorXd& out_y) const
{
	Eigen::VectorXd norm_x = x;
	NormalizeInput(norm_x);
	mModel->Eval(norm_x, out_y);
	UnnormalizeOutput(out_y);
}

void cNeuralNet::EvalBatch(const Eigen::MatrixXd& X, Eigen::MatrixXd& out_Y) const
{
	Eigen::MatrixXd norm_X = X;
	NormalizeInput(norm_X);
	mModel->EvalBatch(norm_X, out_Y);
	for (int i = 0; i < out_Y.rows(); ++i)
	{
		Eigen::VectorXd row = out_Y.row(i);
		UnnormalizeOutput(row);
		out_Y.row(i) = row;
	}
}

void cNeuralNet::Backward(const Eigen::VectorXd& y_diff, Eigen::VectorXd& out_x_diff) const
{
	Eigen::VectorXd norm_y_diff = y_diff;
	UnnormalizeOutputDiff(norm_y_diff);
	mModel->Backward(norm_y_diff, out_x_diff);
	NormalizeInputDiff(out_x_diff);
}

int cNeuralNet::GetInputSize() const { return HasNet() ? mModel->GetInputSize() : 0; }
int cNeuralNet::GetOutputSize() const { return HasNet() ? mModel->GetOutputSize() : 0; }
int cNeuralNet::GetBatchSize() const { return HasNet() ? mModel->GetBatchSize() : 0; }
int cNeuralNet::CalcNumParams() const { return HasNet() ? mModel->CalcNumParams() : 0; }

void cNeuralNet::OutputModel(const std::string& out_file) const
{
	if (!HasNet()) return;
	std::lock_guard<std::mutex> output_lock(gOutputLock);
	mModel->SaveModel(out_file);
	WriteOffsetScale(GetOffsetScaleFile(out_file));
}

void cNeuralNet::PrintParams() const {}

bool cNeuralNet::HasNet() const { return mModel != nullptr; }
bool cNeuralNet::HasSolver() const { return mSolver != nullptr; }
bool cNeuralNet::HasLayer(const std::string layer_name) const { return HasNet() && mModel->HasLayer(layer_name); }
bool cNeuralNet::HasValidModel() const { return mValidModel; }

void cNeuralNet::CopyModel(const cNeuralNet& other)
{
	std::vector<tNNData> params;
	other.mModel->GetParams(params);
	mModel->SetParams(params);
	mInputOffset = other.GetInputOffset();
	mInputScale = other.GetInputScale();
	mOutputOffset = other.GetOutputOffset();
	mOutputScale = other.GetOutputScale();
	SyncSolverParams();
	mValidModel = true;
}

void cNeuralNet::LerpModel(const cNeuralNet& other, double lerp){BlendModel(other, 1 - lerp, lerp);} 
void cNeuralNet::BlendModel(const cNeuralNet& other, double this_weight, double other_weight)
{
	std::vector<tNNData> other_params;
	other.mModel->GetParams(other_params);
	mModel->BlendParams(other_params, this_weight, other_weight);
	SyncSolverParams();
	mValidModel = true;
}

bool cNeuralNet::CompareModel(const cNeuralNet& other) const
{
	std::vector<tNNData> params;
	other.mModel->GetParams(params);
	bool same = mModel->CompareParams(params);
	same &= mInputOffset.isApprox(other.GetInputOffset(), 0);
	same &= mInputScale.isApprox(other.GetInputScale(), 0);
	same &= mOutputOffset.isApprox(other.GetOutputOffset(), 0);
	same &= mOutputScale.isApprox(other.GetOutputScale(), 0);
	return same;
}

void cNeuralNet::ForwardInjectNoisePrefilled(double mean, double stdev, const std::string& layer_name, Eigen::VectorXd& out_y) const { mModel->ForwardInjectNoisePrefilled(mean, stdev, layer_name, out_y); UnnormalizeOutput(out_y); }
void cNeuralNet::GetLayerState(const std::string& layer_name, Eigen::VectorXd& out_state) const { mModel->GetLayerState(layer_name, out_state); }
void cNeuralNet::SetLayerState(const Eigen::VectorXd& state, const std::string& layer_name) const { mModel->SetLayerState(state, layer_name); }

void cNeuralNet::SyncSolverParams() {}
void cNeuralNet::SyncNetParams() {}

void cNeuralNet::CopyGrad(const cNeuralNet& other)
{
	other.mModel->GetParams(mGradBuffer);
	mModel->SetParams(mGradBuffer);
}

bool cNeuralNet::ValidOffsetScale() const { return mInputOffset.size() > 0 && mInputScale.size() > 0 && mOutputOffset.size() > 0 && mOutputScale.size() > 0; }
void cNeuralNet::InitOffsetScale(){ mInputOffset = Eigen::VectorXd::Zero(GetInputSize()); mInputScale = Eigen::VectorXd::Ones(GetInputSize()); mOutputOffset = Eigen::VectorXd::Zero(GetOutputSize()); mOutputScale = Eigen::VectorXd::Ones(GetOutputSize()); }

void cNeuralNet::NormalizeInput(Eigen::MatrixXd& X) const{ if (ValidOffsetScale()) { for (int i = 0; i < X.rows(); ++i){ auto r = X.row(i); r += mInputOffset; r = r.cwiseProduct(mInputScale);} } }
void cNeuralNet::NormalizeInput(Eigen::VectorXd& x) const{ if (ValidOffsetScale()) { x += mInputOffset; x = x.cwiseProduct(mInputScale);} }
void cNeuralNet::NormalizeInputDiff(Eigen::VectorXd& x_diff) const{ if (ValidOffsetScale()) x_diff = x_diff.cwiseProduct(mInputScale); }
void cNeuralNet::UnnormalizeInput(Eigen::VectorXd& x) const{ if (ValidOffsetScale()) { x = x.cwiseQuotient(mInputScale); x -= mInputOffset; } }
void cNeuralNet::UnnormalizeInputDiff(Eigen::VectorXd& x_diff) const{ if (ValidOffsetScale()) x_diff = x_diff.cwiseQuotient(mInputScale); }
void cNeuralNet::NormalizeOutput(Eigen::VectorXd& y) const{ if (ValidOffsetScale()) { y += mOutputOffset; y = y.cwiseProduct(mOutputScale);} }
void cNeuralNet::UnnormalizeOutput(Eigen::VectorXd& y) const{ if (ValidOffsetScale()) { y = y.cwiseQuotient(mOutputScale); y -= mOutputOffset;} }
void cNeuralNet::NormalizeOutputDiff(Eigen::VectorXd& y_diff) const{ if (ValidOffsetScale()) y_diff = y_diff.cwiseProduct(mOutputScale); }
void cNeuralNet::UnnormalizeOutputDiff(Eigen::VectorXd& y_diff) const{ if (ValidOffsetScale()) y_diff = y_diff.cwiseQuotient(mOutputScale); }

void cNeuralNet::LoadTrainData(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
{
	if (!HasSolver()) return;
	auto backend = mSolver->GetTrainerBackend();
	if (backend == nullptr) return;
	Eigen::MatrixXd nX = X;
	Eigen::MatrixXd nY = Y;
	if (ValidOffsetScale())
	{
		NormalizeInput(nX);
		for (int i = 0; i < nY.rows(); ++i)
		{
			Eigen::VectorXd row = nY.row(i);
			NormalizeOutput(row);
			nY.row(i) = row;
		}
	}
	backend->IngestData(nX, nY);
}

std::string cNeuralNet::GetOffsetScaleFile(const std::string& model_file) const
{
	std::string scale_file = cFileUtil::RemoveExtension(model_file);
	scale_file += "_scale.txt";
	return scale_file;
}

void cNeuralNet::WriteOffsetScale(const std::string& norm_file) const
{
	FILE* f = cFileUtil::OpenFile(norm_file, "w");
	if (f == nullptr) return;
	std::string input_offset_json = cJsonUtil::BuildVectorJson(mInputOffset);
	std::string input_scale_json = cJsonUtil::BuildVectorJson(mInputScale);
	std::string output_offset_json = cJsonUtil::BuildVectorJson(mOutputOffset);
	std::string output_scale_json = cJsonUtil::BuildVectorJson(mOutputScale);
	fprintf(f, "{\n\"%s\": %s,\n\"%s\": %s,\n\"%s\": %s,\n\"%s\": %s\n}", gInputOffsetKey.c_str(), input_offset_json.c_str(), gInputScaleKey.c_str(), input_scale_json.c_str(), gOutputOffsetKey.c_str(), output_offset_json.c_str(), gOutputScaleKey.c_str(), output_scale_json.c_str());
	cFileUtil::CloseFile(f);
}
