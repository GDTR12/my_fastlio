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
class Vector : public ManifoldBase<Vector<N>>, public Eigen::Matrix<double, N, 1>
{
public:
    using Base = Eigen::Matrix<double, N, 1>;
    using Tangent = Eigen::Matrix<double, N, 1>;

    Vector<N>():Base(){}

    Vector<N>(const Base& initial)
        : Base(initial)
    {}

    void boxplus_impl(const Base& delta)
    {
        *this = Vector(*this + delta);
    }

};


}