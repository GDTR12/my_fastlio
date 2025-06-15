#pragma once

#include "manifold/so3.hpp"
#include <Eigen/Core>
#include <manifold/manifold.hpp>
#include <manifold/so3.hpp>
#include <manifold/s2.hpp>
#include <manifold/vector.hpp>

namespace mfd = manifold;
namespace ieskf{


typedef mfd::Manifolds<mfd::SO3, mfd::Vector<3>, mfd::Vector<3>, mfd::Vector<3>, mfd::Vector<3>,
                        mfd::S2, mfd::SO3, mfd::Vector<3>> StateType;
// template<typename StateType, int DIM_NOISE>
class IESKF
{
public:
    static constexpr int DIM_NOISE = 12;
    static constexpr int DIM_ERROR_STATE = StateType::DOF;
    typedef Eigen::Matrix<double, DIM_ERROR_STATE, 1> ErrorStateType;
    typedef Eigen::Matrix<double, DIM_NOISE, 1> NoiseType;
    typedef Eigen::Matrix<double, DIM_ERROR_STATE, DIM_ERROR_STATE> ErrorPropagateFx;
    typedef Eigen::Matrix<double, DIM_ERROR_STATE, DIM_NOISE> ErrorPropagateFw;
    typedef Eigen::Matrix<double, DIM_ERROR_STATE, DIM_ERROR_STATE> ErrorStateConvarianceType;
    typedef Eigen::Matrix<double, DIM_NOISE, DIM_NOISE> NoiseConvarianceType;
    typedef Eigen::Matrix<double, Eigen::Dynamic, DIM_ERROR_STATE> ObserveMatrix;

    IESKF()
    {
        error_cov.setIdentity();
        noise_cov.setIdentity();
        error_state.setZero();
    }
    void computeFxAndFw(const ErrorPropagateFx& fx, const ErrorPropagateFw& fw);

    StateType normal_state;
    ErrorStateType error_state;
    NoiseType noise;
    ErrorStateConvarianceType error_cov;
    static NoiseConvarianceType noise_cov;
    
    ErrorPropagateFx fx;
    ErrorPropagateFw fw;
private:
    void predict(const ErrorStateType& delta_x)
    {
        normal_state.boxplus(delta_x);
        computeFxAndFw(fx, fw);
        error_state = fx * error_state + fw * noise;
        error_cov = fx * error_cov * fx.transpose() + fw * noise_cov * fw.transpose();
        error_state.setZero();
    }

    void update(const ObserveMatrix& H, const Eigen::VectorXd& z, const Eigen::MatrixXd& v)
    {
        
    }
};

}
