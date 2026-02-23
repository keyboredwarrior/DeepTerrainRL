#pragma once

#include <vector>
#include <Eigen/Dense>

#include <pytorch/blob.hpp>

class cMinibatchAdapter
{
public:
	typedef double tNNData;

	static void StageMatrix(const Eigen::MatrixXd& mat, int rows, int cols, std::vector<tNNData>& out_data);
	static void StageNormalizedMatrix(const Eigen::MatrixXd& mat, int rows, int cols,
									 const Eigen::VectorXd& offset, const Eigen::VectorXd& scale,
									 std::vector<tNNData>& out_data);
	static void CopyToBlob(const std::vector<tNNData>& data, pytorch::Blob<tNNData>& out_blob);
};
