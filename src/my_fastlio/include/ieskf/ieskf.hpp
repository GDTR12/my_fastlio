#pragma once

#include <Eigen/SparseCholesky>
#include "Eigen/src/SparseCore/SparseMatrix.h"
#include "Eigen/src/SparseCore/SparseUtil.h"
#include "manifold/so3.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cstdlib>
#include <functional>
#include <manifold/manifold.hpp>
#include <manifold/so3.hpp>
#include <manifold/s2.hpp>
#include <manifold/vector.hpp>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <iostream>

namespace ieskf{



struct IESKFUpdateInfo{
    int iter_times;
    double cost_time;
    size_t H_rows = 0;
    double measure_time;
    double begin_cost;
    double end_cost;
    double is_converged = false;
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
    // typedef Eigen::Matrix<double, Eigen::Dynamic, DIM_ERROR_STATE> ObserveMatrix;
    typedef Eigen::SparseMatrix<double> ObserveMatrix;
    // typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ObserveCovarianceType;
    typedef Eigen::SparseMatrix<double> ObserveCovarianceType;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> ObserveResult;

    IESKF()
    {
        error_cov = 1e-5 * ErrorStateConvarianceType::Identity();
        noise_cov = 1e-5 * NoiseConvarianceType::Identity();
        error_state.setZero();
        Eigen::initParallel();
        Eigen::setNbThreads(8); // 或设为固定线程数，如 4
    }

    void predict(const ErrorStateType& delta_x, const ControlType& u, const double dt)
    {
        computeFxAndFw(fx, fw, normal_state, error_state, u, dt);
        normal_state.boxplus(delta_x * dt);
        if (nullptr == computeFxAndFw){
            throw std::runtime_error("Function computeFxAndFw didn't initialized!");
        }
        error_cov = fx * error_cov * fx.transpose() + fw * noise_cov * fw.transpose();
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

        IESKFUpdateInfo info;
        int count = 0;
        double total_time = 0;
        double total_measure_time = 0;
        double last_cost;
        for (int i = 0; i < max_iter; i++)
        {
            Hk.resize(0, 0);
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
            auto t0 = std::chrono::high_resolution_clock::now();

            computeHxAndRinv(zk, Hk, R_inv, Xk, X0);
            if (i == 0){
                last_cost = info.begin_cost = zk.norm();
            }
            
            auto t1 = std::chrono::high_resolution_clock::now();
            K.resize(StateType::DOF, Hk.rows());
            K.setZero();

            // Eigen::SparseMatrix<double> Hk_sp = Hk.sparseView();
            // Eigen::SparseMatrix<double> Rinv_sp = R_inv.sparseView();

            Eigen::SparseMatrix<double> HkT_Rinv = Hk.transpose() * R_inv;
            // Eigen::Matrix<double, DIM_ERROR_STATE, Eigen::Dynamic> HkT_Rinv = HkT_Rinv.eval();
            // HkT_Rinv.resize(DIM_ERROR_STATE, Hk.rows());
            // HkT_Rinv = Hk.transpose() * R_inv;

            ErrorStateConvarianceType HkT_Rinv_Hk = (HkT_Rinv * Hk).eval();
            // std::cout << "Pk: " << Pk.inverse() << std::endl;
            K = (HkT_Rinv_Hk + Pk.inverse()).ldlt().solve(HkT_Rinv.toDense());
            ErrorStateConvarianceType I_KH = ErrorStateConvarianceType::Identity() - K * Hk;
            ErrorStateType delta = (-K * zk - I_KH * Jk_inv * (Xk.boxminus(X0)));
            // std::cout << "j dx: " << (Jk_inv * (Xk.boxminus(X0))).transpose() << std::endl;
            // std::cout << "k zk: " << (K * zk).transpose() << std::endl;
            Xk.boxplus(delta);

            auto t2 = std::chrono::high_resolution_clock::now();
            auto duration = float(std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()) / float(1e3);      
            auto duration_meas = float(std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()) / float(1e3);
            total_time += duration;
            total_measure_time += duration_meas;
            count++;

            double current_cost = zk.norm();

            if (current_cost < last_cost){
                normal_state = Xk;
                error_cov = I_KH * Pk;
            }

            if (i > 0 && delta.norm() < epsilon || i == max_iter - 1){
                info.is_converged = true;
                info.end_cost = current_cost;
                info.H_rows = Hk.rows();
                break;
            }
            last_cost = current_cost;

        }
        // if (info.begin_cost > info.end_cost){
        //     normal_state = Xk;
        //     error_cov = (ErrorStateConvarianceType::Identity() - K * Hk) * Pk;
        // }

        info.avarave_iter_time = total_time / count;
        info.iter_times = count;
        info.cost_time = total_time;
        info.measure_time = total_measure_time / count;
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
