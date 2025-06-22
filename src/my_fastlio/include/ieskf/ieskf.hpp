#pragma once

#include "manifold/so3.hpp"
#include <Eigen/Core>
#include <functional>
#include <manifold/manifold.hpp>
#include <manifold/so3.hpp>
#include <manifold/s2.hpp>
#include <manifold/vector.hpp>
#include <stdexcept>
#include <iostream>

namespace ieskf{


// namespace mfd = manifold;
// typedef mfd::Manifolds<mfd::SO3, mfd::Vector<3>, mfd::Vector<3>, mfd::Vector<3>, mfd::Vector<3>,
//                         mfd::S2<10,10,1>, mfd::SO3, mfd::Vector<3>> StateType;
// typedef mfd::Manifolds<mfd::SO3, mfd::Vector<3>, mfd::Vector<3>, mfd::Vector<3>, mfd::Vector<3>,
//                         mfd::S2<10,10,1>, mfd::SO3, mfd::Vector<3>> ControlType;
template<typename StateType, typename ControlType, int DIM_NOISE>
class IESKF
{
public:
    // static constexpr int DIM_NOISE = 12;
    static constexpr int DIM_ERROR_STATE = StateType::DOF;
    typedef Eigen::Matrix<double, DIM_ERROR_STATE, 1> ErrorStateType;
    typedef Eigen::Matrix<double, DIM_NOISE, 1> NoiseType;
    typedef Eigen::Matrix<double, DIM_ERROR_STATE, DIM_ERROR_STATE> ErrorPropagateFx;
    typedef Eigen::Matrix<double, DIM_ERROR_STATE, DIM_NOISE> ErrorPropagateFw;
    typedef Eigen::Matrix<double, DIM_ERROR_STATE, DIM_ERROR_STATE> ErrorStateConvarianceType;
    typedef Eigen::Matrix<double, DIM_NOISE, DIM_NOISE> NoiseConvarianceType;
    typedef Eigen::Matrix<double, Eigen::Dynamic, DIM_ERROR_STATE> ObserveMatrix;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ObserveCovarianceType;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> ObserveResult;

    IESKF()
    {
        error_cov.setIdentity();
        noise_cov.setIdentity();
        error_state.setZero();
    }

    void predict(const ErrorStateType& delta_x, const ControlType& u, const double dt)
    {
        computeFxAndFw(fx, fw, normal_state, error_state, u, dt);
        normal_state.boxplus(delta_x * dt);
        if (nullptr == computeFxAndFw){
            throw std::runtime_error("Function computeFxAndFw didn't initialized!");
        }
        error_state = fx * error_state + fw * noise;
        error_cov = fx * error_cov * fx.transpose() + fw * noise_cov * fw.transpose();
        error_state.setZero();
    }

    void update()
    {
        StateType& Xk = normal_state;
        StateType X0 = normal_state;
        ErrorStateType& deltaX = error_state;
        ErrorStateConvarianceType& Pk = error_cov;

        std::cout << std::get<7>(Xk.data).transpose() << std::endl;
        Xk.boxplus(deltaX.setOnes() * 0.01);
        std::cout << std::get<7>(Xk.data).transpose() << std::endl;
        for (int i = 0; i < NUM_MAX_ITERATIONS; i++)
        {
            ObserveMatrix Hk; 
            ObserveResult zk;
            ObserveCovarianceType R;
            typename StateType::JacAplusDminusX Jk_inv;
            if (i == 0){
                Jk_inv.setIdentity();
            }else{
                X0.invJacobianDelta_AplusDeltaminusX(Xk, Jk_inv);
                Pk = Jk_inv * Pk * Jk_inv.transpose();
            }
            
            // if (nullptr == computeHxAndR){
            //     throw std::runtime_error("Function computeHxAndR didn't initialized!");
            // }
            // computeHxAndR(zk, Hk, R, Xk, X0);
            ErrorStateType error = Xk.boxminus(X0);
            std::cout << error.transpose() << std::endl;
            exit(0);

        }
    }

    StateType normal_state;
    ErrorStateType error_state;
    NoiseType noise;
    ErrorStateConvarianceType error_cov;
    NoiseConvarianceType noise_cov;
    std::function<void(ErrorPropagateFx& fx, ErrorPropagateFw& fw, const StateType& X, const ErrorStateType& delta_x, const ControlType& u, const double dt)> computeFxAndFw;
    std::function<void(ObserveResult& zk, ObserveMatrix& H, ObserveCovarianceType& R, const StateType& Xk,const StateType& X)> computeHxAndR;
    ErrorPropagateFx fx;
    ErrorPropagateFw fw;

    int NUM_MAX_ITERATIONS = 8;
};

}
