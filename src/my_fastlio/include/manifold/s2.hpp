#pragma  once


#include <Eigen/Core>
#include "manifold.hpp"

namespace manifold {

class S2;

template<>
struct ManifoldTraits<S2>
{
    static constexpr int DOF = 2;
    static constexpr int DIM = 3;
};

class S2 : public ManifoldBase<S2>, public Eigen::Matrix<double, 3, 1>
{
public:
    using Base = Eigen::Matrix<double, 3, 1>;
    using Tangent = Eigen::Matrix<double, 2, 1>;

    S2(const Base& data): Base(data){}
    S2(): Base(){}

    // TODO:
    S2 boxplus_impl(const Tangent& tangent)
    {
        return S2(Base::Identity());
    }
};

}



