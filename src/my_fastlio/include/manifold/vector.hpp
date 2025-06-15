#include "manifold.hpp"
#include <Eigen/Core>

namespace manifold {

template<int N>
class Vector;

template<int N>
struct ManifoldTraits<Vector<N>>
{
    static constexpr int DOF = N;
    static constexpr int DIM = N;
};

template<int N>
class Vector : public ManifoldBase<Vector<N>>, Eigen::Matrix<double, N, 1>
{
public:
    using Base = Eigen::Matrix<double, N, 1>;
    using Tangent = Eigen::Matrix<double, N, 1>;

    Vector<N>():Base(){}

    Vector<N>(const Base& initial)
    {
        *this = initial;
    }

    Vector<N> boxplus_impl(const Base& delta)
    {
        return *this + delta;
    }

};


}