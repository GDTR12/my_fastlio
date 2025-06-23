#pragma  once


#include <Eigen/Core>
#include <sophus/so3.hpp>
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Geometry/Quaternion.h"
#include "manifold.hpp"
#include "iostream"

namespace manifold {

template <int Len0, int Len1, int S2_Type>
class S2;

enum S2Type{
    CLOSE_X,
    CLOSE_Y,
    CLOSE_Z,
};

template<int Len0, int Len1, int S2_Type>
struct ManifoldTraits<S2<Len0, Len1, S2_Type>>
{
    static constexpr int DOF = 2;
    static constexpr int DIM = 3;
};

template<int Len0, int Len1, int S2_Type>
class S2 : public ManifoldBase<S2<Len0, Len1, S2_Type>>, public Eigen::Matrix<double, 3, 1>
{
public:
    using Base = Eigen::Matrix<double, 3, 1>;
    using Tangent = Eigen::Matrix<double, 2, 1>;
    static constexpr double len = static_cast<double>(Len0) / static_cast<double>(Len1);
    static constexpr double tolerance = 1e-8;

    S2(const Base& data)
    {
        *this = len * data.normalized(); 
    }

    S2()
    {
        if (S2_Type == CLOSE_X) {
            *this = Base::UnitX() * len;
        }else if (S2_Type == CLOSE_Y){
            *this = Base::UnitY() * len;
        }else{
            *this = Base::UnitZ() * len;
        }
    }

    void boxplus_impl(const Tangent& tangent)
    {
        auto Bx = computeBx(*this);
        *this = SO3Exp(Bx * tangent) * (*this);
    }

    // Y:this X:other y \minus x 
    Eigen::Matrix<double, 2, 1> boxminus_impl(const S2& other) const
    {
        Eigen::Vector3d Y = *this;
        Eigen::Vector3d X = other;
        Eigen::Matrix<double, 2, 3> BxT = computeBx(X).transpose();
        auto theta = Eigen::Quaterniond::FromTwoVectors(X, Y);

        // std::cout << theta.coeffs().transpose() << std::endl;
        auto Bxu = Sophus::SO3d(theta).log();
        // std::cout << Bxu << std::endl;
        return Eigen::Vector2d(BxT.row(0) * Bxu, BxT.row(1) * Bxu);
    }


    S2& operator=(const Base& other)
    {
        Base::operator=(len * other.normalized());
        return *this;
    }

    void jacobianDelta_AplusDeltaminusX(const S2& A, Eigen::Ref<Eigen::Matrix<double, 2, 2>> mat)
    {
        mat.setIdentity();
    }

    void invJacobianDelta_AplusDeltaminusX(const S2& A, Eigen::Ref<Eigen::Matrix<double, 2, 2>> mat)
    {
        mat.setIdentity();
    }

    Eigen::Matrix<double, 3, 2> boxplusJacobian(const Tangent& u) const
    {
        Eigen::Matrix<double, 3, 2> Bx = computeBx((*this));
        Eigen::Matrix<double, 3, 1> Bu = Bx * u;
        return -SO3Exp(Bu).matrix() * SO3Hat(*this) * SO3Jr(Bu) * Bx;
    }

    S2 operator-(const S2& other) = delete;
    S2 operator+(const S2& other) = delete;

private:

    Eigen::Matrix<double, 3, 2> computeBx(const Eigen::Vector3d x) const
    {
        Eigen::Matrix<double, 3, 2> res;
        Eigen::Vector3d tmp(0,0,1);
        if (x.dot(tmp) < 1e-2){
            tmp = Eigen::Vector3d(1, 0 ,0);
        }
        Eigen::Vector3d tmp1 = (tmp - x * (x.transpose() * tmp)).normalized();
        res.leftCols(1) = tmp1;
        res.rightCols(1) = x.cross(tmp1).normalized();
        // std::cout << "bx" << res << std::endl;
        return res;
        // auto vec = this->data();
        // if (S2_Type == CLOSE_Y){
        //     if (vec[1] + len > tolerance){
        //         res << len - vec[0]*vec[0]/(len+vec[1]), -vec[0]*vec[2]/(len+vec[1]),
        //                 -vec[0], -vec[2],
        //                 -vec[0]*vec[2]/(len+vec[1]), len-vec[2]*vec[2]/(len+vec[1]);
        //         res /= len;
        //     }else {
        //         res = Eigen::Matrix<double, 3, 2>::Zero();
        //         res(1, 1) = -1;
        //         res(2, 0) = 1;
        //     }
        // }else if (S2_Type == CLOSE_Z){
        //     res = Eigen::Matrix<double, 3,2>::Zero();


		// 	// if(std::abs(std::abs(vec[2]) - len) > tolerance)
		// 	// {
				
		// 	// 	res << len - vec[0]*vec[0]/(len+vec[2]), -vec[0]*vec[1]/(len+vec[2]),
		// 	// 			-vec[0]*vec[1]/(len+vec[2]), len-vec[1]*vec[1]/(len+vec[2]),
		// 	// 			-vec[0], -vec[1];
		// 	// 	res /= len;
		// 	// }else{
		// 	// 	res = Eigen::Matrix<double, 3, 2>::Zero();
		// 		res(0, 0) = 1;
		// 		res(1, 1) = 1;
		// 	// }
        // }else{
        //     if(vec[0] + len > tolerance)
        //     {
                
        //         res << -vec[1], -vec[2],
        //                     len - vec[1]*vec[1]/(len+vec[0]), -vec[2]*vec[1]/(len+vec[0]),
        //                     -vec[2]*vec[1]/(len+vec[0]), len-vec[2]*vec[2]/(len+vec[0]);
        //         res /= len;
        //     }
        //     else
        //     {
        //         res = Eigen::Matrix<double, 3, 2>::Zero();
        //         res(1, 1) = -1;
        //         res(2, 0) = 1;
        //     }
        // }
        // return res;
    }
};

}



