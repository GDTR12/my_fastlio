#pragma  once

#include "manifold.hpp"
#include "sophus/so3.hpp"

namespace manifold {

class SO3;

template<>
struct ManifoldTraits<SO3>
{
    static constexpr int DOF = Sophus::SO3<double>::DoF;
    static constexpr int DIM = Sophus::SO3<double>::num_parameters;
};

class SO3 : public ManifoldBase<SO3>, public Sophus::SO3<double>
{
public:
    using Base = Sophus::SO3<double>;
    using Tangent = Eigen::Matrix<double, 3, 1>;

    SO3()
      : Sophus::SO3<double>(Eigen::Matrix3d::Identity())
    {}

    SO3(const Sophus::SO3d& other)
      : Sophus::SO3<double>(other)
    {}

    void boxplus_impl(const Eigen::Matrix<double, DOF, 1>& delta)
    {
        *this = SO3((*this) * Base::exp(delta));
    }

    static SO3 Identity(){return Base();}

};

}
