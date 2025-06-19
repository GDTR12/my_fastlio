#pragma  once


#include <Eigen/Core>
#include <sophus/so3.hpp>
#include "manifold.hpp"

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
            *this = Base::UnitY() * len;
        }else if (S2_Type == CLOSE_Y){
            *this = Base::UnitZ() * len;
        }else{
            *this = Base::UnitX() * len;
        }
    }

    void boxplus_impl(const Tangent& tangent)
    {
        auto Bx = computeBx();
        *this = Sophus::SO3d::exp(Bx * tangent) * (*this);
    }

    S2& operator=(const Base& other)
    {
        Base::operator=(len * other.normalized());
        return *this;
    }

    Eigen::Matrix<double, 3, 2> boxplusJacobian(const Tangent& u) const
    {
        Eigen::Matrix<double, 3, 2> Bx = computeBx();
        Eigen::Matrix<double, 3, 1> Bu = Bx * u;
        return Sophus::SO3d::exp(Bu).matrix() * Sophus::SO3d::hat(*this) * Sophus::SO3d::leftJacobianInverse(Bu) * Bx;
    }

    S2 operator-(const S2& other) = delete;
    S2 operator+(const S2& other) = delete;

private:

    Eigen::Matrix<double, 3, 2> computeBx() const
    {
        Eigen::Matrix<double, 3, 2> res;
        auto vec = this->data();
        if (S2_Type == CLOSE_X){
            if (vec[1] + len > tolerance){
                res << len - vec[0]*vec[0]/(len+vec[1]), -vec[0]*vec[2]/(len+vec[1]),
                        -vec[0], -vec[2],
                        -vec[0]*vec[2]/(len+vec[1]), len-vec[2]*vec[2]/(len+vec[1]);
                res /= len;
            }else {
                res = Eigen::Matrix<double, 3, 2>::Zero();
                res(1, 1) = -1;
                res(2, 0) = 1;
            }
        }else if (S2_Type == CLOSE_Y){
			if(vec[2] + len > tolerance)
			{
				
				res << len - vec[0]*vec[0]/(len+vec[2]), -vec[0]*vec[1]/(len+vec[2]),
						-vec[0]*vec[1]/(len+vec[2]), len-vec[1]*vec[1]/(len+vec[2]),
						-vec[0], -vec[1];
				res /= len;
			}else{
				res = Eigen::Matrix<double, 3, 2>::Zero();
				res(1, 1) = -1;
				res(2, 0) = 1;
			}
        }else{
            if(vec[0] + len > tolerance)
            {
                
                res << -vec[1], -vec[2],
                            len - vec[1]*vec[1]/(len+vec[0]), -vec[2]*vec[1]/(len+vec[0]),
                            -vec[2]*vec[1]/(len+vec[0]), len-vec[2]*vec[2]/(len+vec[0]);
                res /= len;
            }
            else
            {
                res = Eigen::Matrix<double, 3, 2>::Zero();
                res(1, 1) = -1;
                res(2, 0) = 1;
            }
        }
        return res;
    }
};

}



