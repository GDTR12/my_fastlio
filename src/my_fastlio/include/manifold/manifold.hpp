#pragma once

#include <Eigen/Core>
#include <tuple>


namespace manifold{

template<typename T>
struct ManifoldTraits;

template<typename Derived>
class ManifoldBase
{
public:
    static constexpr int DOF = ManifoldTraits<Derived>::DOF;
    static constexpr int DIM = ManifoldTraits<Derived>::DIM;

    Derived boxplus(const Eigen::Matrix<double, DOF, 1>& tangent) const
    {
        static_cast<const Derived*>(this)->boxplus_impl(tangent);
    }

    static Derived Identity(){return Derived::Identity();}

};

template<typename... Args>
class Manifolds
{
public: 
    std::tuple<Args...> data;
    Manifolds()
    {
        data = std::make_tuple(Args{}...);
    }
    
};

}
