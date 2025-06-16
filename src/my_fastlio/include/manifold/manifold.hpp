#pragma once

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>


namespace manifold{

template<typename T>
struct ManifoldTraits;

template<typename Derived>
class ManifoldBase
{
public:
    static constexpr int DOF = ManifoldTraits<Derived>::DOF;
    static constexpr int DIM = ManifoldTraits<Derived>::DIM;

    void boxplus(const Eigen::Matrix<double, DOF, 1>& tangent)
    {
        static_cast<Derived*>(this)->boxplus_impl(tangent);
    }

    static Derived Identity(){return Derived::Identity();}

};

template<typename... Args>
class Manifolds
{
public: 
    Manifolds()
    {
        data = std::make_tuple(Args{}...);
    }

    static constexpr int DIM = (ManifoldTraits<Args>::DIM + ...);
    static constexpr int DOF = (ManifoldTraits<Args>::DOF + ...);

    void boxplus(const Eigen::Matrix<double, DOF, 1>& delta)
    {
        boxplus_impl<0>(delta, 0);
    }

    template<std::size_t N>
    static decltype(auto) get(std::tuple<Args...>& tuple)
    {
        return std::get<N>(tuple);
    }

    std::tuple<Args...> data;
private:

    static constexpr std::array<int, sizeof...(Args)> createOffset()
    {
        constexpr std::array<int, sizeof...(Args)> dofs;
        constexpr std::array<int, sizeof...(Args)> offset;
        offset[0] = 0;
        for(int i = 1; i < sizeof...(Args); i++)
        {
            offset[i] = offset[i - 1] + dofs[i - 1];
        }
        return offset;
    }

    template<std::size_t I>
    void boxplus_impl(const Eigen::Matrix<double, DOF, 1>& delta, int offset)
    {
        if constexpr (I == sizeof...(Args)){
            return;
        }else{
            using Type = std::tuple_element_t<I, std::tuple<Args...>>;
            constexpr std::size_t dof = Type::DOF;
            auto tangent = delta.template segment<dof>(offset);
            std::get<I>(data).boxplus(tangent);
            boxplus_impl<I+1>(delta, offset + dof);
        }
    }

    static constexpr std::array<int, sizeof...(Args)> IDX_LST = createOffset();
    
};

}
