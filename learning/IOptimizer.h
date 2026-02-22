#pragma once

#include <memory>

class IOptimizer
{
public:
	virtual ~IOptimizer() = default;

	virtual void Step(int steps) = 0;
	virtual void Reset() = 0;
	virtual void ZeroGrad() = 0;
	virtual void Update() = 0;
	virtual double ForwardBackward() = 0;
};

using tOptimizerPtr = std::shared_ptr<IOptimizer>;
