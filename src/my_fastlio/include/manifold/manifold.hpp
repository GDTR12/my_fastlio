#pragma once

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include "Eigen/src/Core/Matrix.h"
#include "common_define.hpp"


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

    Eigen::Matrix<double, DOF, 1> boxminus(const Derived& other) const
    {
        return static_cast<const Derived*>(this)->boxminus_impl(other);
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

    // this \boxminus other
    Eigen::Matrix<double, DOF, 1> boxminus(const Manifolds<Args...>& other)
    {
        Eigen::Matrix<double, DOF, 1> ret;
        boxminus_impl<0>(other, ret, 0);
        return ret;
    }

    typedef Eigen::Matrix<double, DOF, DOF> JacAplusDminusX;
    void jacobianDelta_AplusDeltaminusX(const Manifolds<Args...>& A, JacAplusDminusX& jacobian)
    {
        jacobian.setZero();
        jacobianDelta_AplusDeltaminusX_impl<0>(A, jacobian, 0);
    }

    void invJacobianDelta_AplusDeltaminusX(const Manifolds<Args...>& A, JacAplusDminusX& jacobian)
    {
        jacobian.setZero();
        invJacobianDelta_AplusDeltaminusX_impl<0>(A, jacobian, 0);
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


    template<std::size_t I>
    void boxminus_impl(const Manifolds<Args...>& other, Eigen::Matrix<double, DOF, 1>& delta, int offset)
    {
        if constexpr (I == sizeof...(Args)){
            return;
        }else{
            using Type = std::tuple_element_t<I, std::tuple<Args...>>;
            constexpr std::size_t dof = Type::DOF;
            const Type& Y = std::get<I>(data);
            const Type& X = std::get<I>(other.data);
            Eigen::Matrix<double, dof, 1> delta_seg = Y.boxminus(X);
            delta.template segment<dof>(offset) = delta_seg;
            boxminus_impl<I+1>(other, delta, offset + dof);
        }
    }

    template<std::size_t I>
    void jacobianDelta_AplusDeltaminusX_impl(const Manifolds<Args...>& A, Eigen::Matrix<double, DOF, DOF>& jacobian, int offset_jac)
    {
        if constexpr (I == sizeof...(Args)){
            return;
        }else{
            using Type = std::tuple_element_t<I, std::tuple<Args...>>;
            constexpr std::size_t dof = Type::DOF;
            Type x = std::get<I>(data);
            Type a = std::get<I>(A.data);
            x.jacobianDelta_AplusDeltaminusX(a, jacobian.block(offset_jac, offset_jac, dof, dof));
            jacobianDelta_AplusDeltaminusX_impl<I+1>(A, jacobian, offset_jac + dof);
        }
    }

    template<std::size_t I>
    void invJacobianDelta_AplusDeltaminusX_impl(const Manifolds<Args...>& A, Eigen::Matrix<double, DOF, DOF>& jacobian, int offset_jac)
    {
        if constexpr (I == sizeof...(Args)){
            return;
        }else{
            using Type = std::tuple_element_t<I, std::tuple<Args...>>;
            constexpr std::size_t dof = Type::DOF;
            Type x = std::get<I>(data);
            Type a = std::get<I>(A.data);
            x.invJacobianDelta_AplusDeltaminusX(a, jacobian.block(offset_jac, offset_jac, dof, dof));
            invJacobianDelta_AplusDeltaminusX_impl<I+1>(A, jacobian, offset_jac + dof);
        }
    }


    static constexpr std::array<int, sizeof...(Args)> IDX_LST = createOffset();
    
};

}
