#include "MinibatchAdapter.h"

#include <algorithm>
#include <cassert>

void cMinibatchAdapter::StageMatrix(const Eigen::MatrixXd& mat, int rows, int cols, std::vector<tNNData>& out_data)
{
	assert(rows >= 0);
	assert(cols >= 0);
	assert(mat.rows() >= rows);
	assert(mat.cols() == cols);

	const int count = rows * cols;
	out_data.resize(count);

	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			out_data[i * cols + j] = mat(i, j);
		}
	}
}

void cMinibatchAdapter::StageNormalizedMatrix(const Eigen::MatrixXd& mat, int rows, int cols,
											 const Eigen::VectorXd& offset, const Eigen::VectorXd& scale,
											 std::vector<tNNData>& out_data)
{
	assert(offset.size() == cols);
	assert(scale.size() == cols);

	StageMatrix(mat, rows, cols, out_data);
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			tNNData& val = out_data[i * cols + j];
			val += offset[j];
			val *= scale[j];
		}
	}
}

void cMinibatchAdapter::CopyToBlob(const std::vector<tNNData>& data, caffe::Blob<tNNData>& out_blob)
{
	assert(out_blob.count() == static_cast<int>(data.size()));
	std::copy(data.begin(), data.end(), out_blob.mutable_cpu_data());
}
