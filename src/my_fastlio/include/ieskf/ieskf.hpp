#pragma once

#include "manifold/so3.hpp"
#include <Eigen/Core>
#include <manifold/manifold.hpp>
#include <manifold/so3.hpp>
#include <manifold/s2.hpp>
#include <manifold/vector.hpp>

namespace ieskf{

template<typename StateType>
class IESKF
{
public:
    StateType state;
private:
    void predict();
    void update();
};

}
