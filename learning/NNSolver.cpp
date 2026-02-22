#include "NNSolver.h"

#include <algorithm>
#include <cctype>
#include <caffe/caffe.hpp>
#include <caffe/sgd_solvers.hpp>
#include <fstream>
#include <json/json.h>
#include <sstream>

namespace
{
	template <typename tSolverType>
	class cCaffeOptimizerExecutor : public cOptimizerExecutor, protected tSolverType
	{
	public:
		explicit cCaffeOptimizerExecutor(const caffe::SolverParameter& param)
			: cOptimizerExecutor(), tSolverType(param)
		{
		}

		virtual boost::shared_ptr<caffe::Net<cNeuralNet::tNNData>> GetNet() override
		{
			return tSolverType::net();
		}

		virtual void ApplySteps(int steps) override
		{
			tSolverType::Step(steps);
		}

		virtual cNeuralNet::tNNData ForwardBackward() override
		{
			this->GetNet()->ClearParamDiffs();
			return this->GetNet()->ForwardBackward();
		}
	};

	std::string Trim(const std::string& str)
	{
		const auto begin = str.find_first_not_of(" \t\r\n");
		if (begin == std::string::npos)
		{
			return "";
		}
		const auto end = str.find_last_not_of(" \t\r\n");
		return str.substr(begin, end - begin + 1);
	}

	std::string ToLower(std::string str)
	{
		std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		return str;
	}

	bool LoadJsonConfig(const std::string& config_file, cOptimizerExecutor::tConfig& out_config)
	{
		std::ifstream f_stream(config_file);
		if (!f_stream.is_open())
		{
			return false;
		}
		Json::Reader reader;
		Json::Value root;
		if (!reader.parse(f_stream, root))
		{
			return false;
		}

		if (!root["backend"].isNull())
		{
			out_config.mBackend = root["backend"].asString();
		}
		if (!root["optimizer"].isNull())
		{
			out_config.mOptimizer = root["optimizer"].asString();
		}
		if (!root["solver_file"].isNull())
		{
			out_config.mPolicyCheckpoint = root["solver_file"].asString();
		}
		return true;
	}

	bool LoadYamlConfig(const std::string& config_file, cOptimizerExecutor::tConfig& out_config)
	{
		std::ifstream f_stream(config_file);
		if (!f_stream.is_open())
		{
			return false;
		}

		std::string line;
		while (std::getline(f_stream, line))
		{
			line = Trim(line);
			if (line.empty() || line[0] == '#')
			{
				continue;
			}

			auto delim = line.find(':');
			if (delim == std::string::npos)
			{
				continue;
			}
			std::string key = ToLower(Trim(line.substr(0, delim)));
			std::string val = Trim(line.substr(delim + 1));
			if (!val.empty() && (val.front() == '"' || val.front() == '\''))
			{
				val = val.substr(1, val.size() - 2);
			}

			if (key == "backend")
			{
				out_config.mBackend = val;
			}
			else if (key == "optimizer")
			{
				out_config.mOptimizer = val;
			}
			else if (key == "solver_file")
			{
				out_config.mPolicyCheckpoint = val;
			}
		}
		return true;
	}

	cOptimizerExecutor::tConfig ParseConfig(const std::string& config_file)
	{
		cOptimizerExecutor::tConfig config;
		if (!LoadJsonConfig(config_file, config))
		{
			LoadYamlConfig(config_file, config);
		}
		return config;
	}

	void BuildCaffeExecutor(const cOptimizerExecutor::tConfig& config, std::shared_ptr<cOptimizerExecutor>& out_executor)
	{
		caffe::SolverParameter param;
		caffe::ReadProtoFromTextFileOrDie(config.mPolicyCheckpoint, &param);
		caffe::Caffe::set_mode(caffe::Caffe::CPU);
		const std::string optimizer = ToLower(config.mOptimizer);

		if (optimizer == "sgd")
		{
			out_executor = std::shared_ptr<cOptimizerExecutor>(new cCaffeOptimizerExecutor<caffe::SGDSolver<cNeuralNet::tNNData>>(param));
		}
		else if (optimizer == "nesterov")
		{
			out_executor = std::shared_ptr<cOptimizerExecutor>(new cCaffeOptimizerExecutor<caffe::NesterovSolver<cNeuralNet::tNNData>>(param));
		}
		else if (optimizer == "adagrad")
		{
			out_executor = std::shared_ptr<cOptimizerExecutor>(new cCaffeOptimizerExecutor<caffe::AdaGradSolver<cNeuralNet::tNNData>>(param));
		}
		else if (optimizer == "rmsprop")
		{
			out_executor = std::shared_ptr<cOptimizerExecutor>(new cCaffeOptimizerExecutor<caffe::RMSPropSolver<cNeuralNet::tNNData>>(param));
		}
		else if (optimizer == "adadelta")
		{
			out_executor = std::shared_ptr<cOptimizerExecutor>(new cCaffeOptimizerExecutor<caffe::AdaDeltaSolver<cNeuralNet::tNNData>>(param));
		}
		else if (optimizer == "adam")
		{
			out_executor = std::shared_ptr<cOptimizerExecutor>(new cCaffeOptimizerExecutor<caffe::AdamSolver<cNeuralNet::tNNData>>(param));
		}
		else
		{
			LOG(FATAL) << "Unknown optimizer: " << config.mOptimizer;
		}
	}
}

cOptimizerExecutor::tConfig::tConfig()
	: mBackend("caffe"), mOptimizer("sgd"), mPolicyCheckpoint("")
{
}

bool cOptimizerExecutor::tConfig::IsValid() const
{
	return !mBackend.empty() && !mOptimizer.empty() && !mPolicyCheckpoint.empty();
}

void cOptimizerExecutor::BuildExecutor(const std::string& config_file, std::shared_ptr<cOptimizerExecutor>& out_executor)
{
	auto config = ParseConfig(config_file);
	if (!config.IsValid())
	{
		LOG(FATAL) << "Invalid optimizer config file: " << config_file;
	}

	if (ToLower(config.mBackend) == "caffe")
	{
		BuildCaffeExecutor(config, out_executor);
	}
	else
	{
		LOG(FATAL) << "Unsupported optimizer backend: " << config.mBackend;
	}
}

cOptimizerExecutor::cOptimizerExecutor()
{
}

cOptimizerExecutor::~cOptimizerExecutor()
{
}
