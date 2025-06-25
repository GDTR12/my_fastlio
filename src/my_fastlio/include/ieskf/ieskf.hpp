#pragma once

#include "manifold/so3.hpp"
#include <Eigen/Core>
#include <functional>
#include <manifold/manifold.hpp>
#include <manifold/so3.hpp>
#include <manifold/s2.hpp>
#include <manifold/vector.hpp>
#include <stdexcept>
#include <chrono>
#include <iostream>

namespace ieskf{



struct IESKFUpdateInfo{
    int iter_times;
    double cost_time;
    double begin_cost;
    double end_cost;
    double avarave_iter_time;
};

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
        error_cov = 1e-5 * ErrorStateConvarianceType::Identity();
        noise_cov = 1e-5 * NoiseConvarianceType::Identity();
        error_state.setZero();
    }

    void predict(const ErrorStateType& delta_x, const ControlType& u, const double dt)
    {
        computeFxAndFw(fx, fw, normal_state, error_state, u, dt);
        normal_state.boxplus(delta_x * dt);
        if (nullptr == computeFxAndFw){
            throw std::runtime_error("Function computeFxAndFw didn't initialized!");
        }
        // std::cout << fx << std::endl;
        error_state = fx * error_state + fw * noise;
        // std::cout << "fx: \n" << fx << "\n" << std::endl;
        error_cov = fx * error_cov * fx.transpose() + fw * noise_cov * fw.transpose();
        // error_state.setZero();
    }

    IESKFUpdateInfo update(int max_iter, double epsilon)
    {
        StateType Xk = normal_state;
        StateType last_Xk;
        StateType X0 = normal_state;
        ErrorStateType& deltaX = error_state;
        ErrorStateConvarianceType Pk = error_cov;
        Eigen::Matrix<double, StateType::DOF, Eigen::Dynamic> K;
        ObserveMatrix Hk; 
        // std::cout << std::get<7>(Xk.data).transpose() << std::endl;
        // Xk.boxplus(deltaX.setOnes() * 0.01);
        // std::cout << std::get<7>(Xk.data).transpose() << std::endl;
        // std::cout << Pk << std::endl;
        // exit(0);

        // std::cout << "es: \n" << error_state.transpose() << std::endl;
        // std::cout << "es cov: \n" << error_cov << std::endl;
        // exit(0);
        IESKFUpdateInfo info;
        int count = 0;
        double total_time = 0;
        for (int i = 0; i < max_iter; i++)
        {
            ObserveResult zk;
            ObserveCovarianceType R_inv;
            typename StateType::JacAplusDminusX Jk_inv;
            if (i == 0){
                Jk_inv.setIdentity();
            }else{
                X0.invJacobianDelta_AplusDeltaminusX(Xk, Jk_inv);
                // std::cout << Jk_inv << std::endl;
                Pk = Jk_inv * error_cov * Jk_inv.transpose();
            }
            
            if (nullptr == computeHxAndRinv){
                throw std::runtime_error("Function computeHxAndR didn't initialized!");
            }
            computeHxAndRinv(zk, Hk, R_inv, Xk, X0);
            if (i == 0){
                info.begin_cost = zk.norm();
            }
            
            auto t1 = std::chrono::high_resolution_clock::now();
            // std::cout << "Pk inv: \n" << Pk << std::endl << std::endl;
            K.resize(StateType::DOF, Hk.rows());
            K.setZero();
            K = (Hk.transpose() * R_inv * Hk + Pk.inverse()).ldlt().solve(Hk.transpose() * R_inv);
            // K = (Hk.transpose() * Hk + (Pk / 0.001).inverse()).ldlt().solve(Hk.transpose());
            // K = (Hk.transpose() * R_inv * Hk + Pk.inverse()).inverse() * Hk.transpose() * R_inv;
            // K = (Hk.transpose() * R_inv * Hk + Pk.inverse()).inverse();
            // std::cout << K.leftCols(2) << std::endl;
            // std::cout << "K: " << K << std::endl << std::endl;
            last_Xk = Xk;

            // std::cout << K << std::endl;
            // std::cout << Jk_inv << std::endl;
            // std::cout << Pk << std::endl;
            // std::cout << Xk.boxminus(X0).transpose() << std::endl;
            // std::cout << Jk_inv * (Xk.boxminus(X0)) << std::endl;
            Eigen::Matrix<double, StateType::DOF, 1> delta = (-K * zk - (Eigen::Matrix<double, DIM_ERROR_STATE, DIM_ERROR_STATE>::Identity() - K * Hk) * Jk_inv * (Xk.boxminus(X0)));
            // delta.template tail<2>() = Eigen::Vector2d::Zero();
            // std::cout << Xk.boxminus(X0).transpose() << std::endl;
            // std::cout << "K: " << K.rows() << ", " << K.cols() << std::endl;
            // std::cout << "j dx: " << (Jk_inv * (Xk.boxminus(X0))).transpose() << std::endl;
            // std::cout << "k zk: " << (K * zk).transpose() << std::endl;
            Xk.boxplus(delta);

            auto t2 = std::chrono::high_resolution_clock::now();
            auto duration = float(std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()) / float(1e3);      
            total_time += duration;
            count++;

            // if (i > 0 && Xk.boxminus(last_Xk).norm() < epsilon || i == max_iter - 1){
            if (i == max_iter - 1){
                std::cout << "iter finish" << std::endl;
                info.end_cost = zk.norm();
                break;
            }
        }
        normal_state = Xk;
        // std::cout << K * Hk << std::endl;
        error_cov = (ErrorStateConvarianceType::Identity() - K * Hk) * Pk;

        // std::cout << Hk << std::endl;
        // exit(0);
        // std::cout << error_cov << std::endl << std::endl;

        info.avarave_iter_time = total_time / count;
        info.iter_times = count;
        info.cost_time = total_time;

        return info;
    }



    StateType normal_state;
    ErrorStateType error_state;
    NoiseType noise;
    ErrorStateConvarianceType error_cov;
    NoiseConvarianceType noise_cov;
    std::function<void(ErrorPropagateFx& fx, ErrorPropagateFw& fw, const StateType& X, const ErrorStateType& delta_x, const ControlType& u, const double dt)> computeFxAndFw;
    std::function<void(ObserveResult& zk, ObserveMatrix& H, ObserveCovarianceType& R, const StateType& Xk,const StateType& X)> computeHxAndRinv;
    ErrorPropagateFx fx;
    ErrorPropagateFw fw;

};

}
