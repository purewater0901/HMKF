#include "distribution/two_dimensional_normal_distribution.h"

TwoDimensionalNormalDistribution::TwoDimensionalNormalDistribution(const Eigen::Vector2d& mean,
                                                                   const Eigen::Matrix2d& covariance)
                                                                   : mean_(mean), covariance_(covariance)
{
    if(!checkPositiveDefiniteness(covariance_)) {
        throw std::runtime_error("Possibly non semi-positive definitie matrix!");
    }

    if(!initializeData()) {
        throw std::runtime_error("Failed to do Eigen Decomposition.");
    }

    // enable initialization
    initialization_ = true;
}

void TwoDimensionalNormalDistribution::setValues(const Eigen::Vector2d &mean, const Eigen::Matrix2d &covariance)
{
    // Set Value
    mean_ = mean;
    covariance_ = covariance;

    if(!checkPositiveDefiniteness(covariance_)) {
        throw std::runtime_error("Possibly non semi-positive definitie matrix!");
    }

    if(!initializeData()) {
        throw std::runtime_error("Failed to do Eigen Decomposition.");
    }

    initialization_ = true;
}

bool TwoDimensionalNormalDistribution::checkPositiveDefiniteness(const Eigen::Matrix2d& covariance)
{
    Eigen::LLT<Eigen::MatrixXd> lltOfA(covariance_); // compute the Cholesky decomposition of A
    if(lltOfA.info() == Eigen::NumericalIssue) {
        return false;
    }

    return true;
}

bool TwoDimensionalNormalDistribution::initializeData()
{
    if(std::fabs(covariance_(0, 1)) < 1e-10)
    {
        // Two Variables are independent
        eigen_values_ << 1.0/covariance_(0, 0), 1.0/covariance_(1, 1);
        T_ = Eigen::Matrix2d::Identity();
        independent_ = true;
    }
    else
    {
        // Inverse Matrix
        const Eigen::Matrix2d inv_covariance = covariance_.inverse();

        // Eigen Decomposition
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(inv_covariance);
        if(es.info() != Eigen::Success) {
            return false;
        }

        eigen_values_ = es.eigenvalues();
        T_ = es.eigenvectors();
        independent_ = false;
    }
    return true;
}

double TwoDimensionalNormalDistribution::calc_mean(const int dim)
{
    if(dim > 1) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    return mean_(dim);
}

double TwoDimensionalNormalDistribution::calc_covariance(const int dim)
{
    if(dim > 1) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    return covariance_(dim, dim);
}

double TwoDimensionalNormalDistribution::calc_moment(const int dim, const int moment)
{
    if(dim > 1) {
        throw std::invalid_argument("Dim is larger than the size of the distribution");
    }
    NormalDistribution dist(mean_(dim), covariance_(dim, dim));
    return dist.calc_moment(moment);
}

double TwoDimensionalNormalDistribution::calc_xy_cos_y_sin_y_moment(const int x_moment, const int y_moment, const int cos_moment, const int sin_moment)
{
    if(!initialization_) {
        throw std::runtime_error("Need To Initialize two dimensional normal distribution");
    }

    if(independent_) {
        NormalDistribution normal_x(mean_(0), covariance_(0, 0));
        NormalDistribution normal_y(mean_(1), covariance_(1, 1));

        return normal_x.calc_moment(x_moment) * normal_y.calc_x_cos_sin_moment(y_moment, cos_moment, sin_moment);
    }

    const auto y_mean = T_.transpose()*mean_;
    const double t11 = T_(0, 0);
    const double t12 = T_(0, 1);
    const double t21 = T_(1, 0);
    const double t22 = T_(1, 1);

    const double l1_mean = t21*y_mean(0);
    const double l1_cov = t21*t21/eigen_values_(0);
    NormalDistribution l1(l1_mean, l1_cov);

    const double l2_mean = t22*y_mean(1);
    const double l2_cov = t22*t22/eigen_values_(1);
    NormalDistribution l2(l2_mean, l2_cov);

    double result = 0.0;
    for(int k=0; k<=x_moment; ++k) {
        for(int l=0; l<=y_moment; ++l) {
            const double x_coeff = nCr(x_moment, k) * std::pow(t11, k) / std::pow(t21, k);
            const double y_coeff = nCr(y_moment, l) * std::pow(t12, x_moment - k) / std::pow(t22, x_moment-k);
            for(int c=0; c<=cos_moment; ++c) {
                for(int s=0; s<= sin_moment; ++s) {
                    const double cos_coeff = nCr(cos_moment, c) * std::pow(-1, cos_moment-c);
                    const double sin_coeff = nCr(sin_moment, s);
                    const double coeff = x_coeff * y_coeff * cos_coeff * sin_coeff;
                    const double l1_moment = l1.calc_x_cos_sin_moment(k+l, sin_moment+c-s, cos_moment+s-c);
                    const double l2_moment = l2.calc_x_cos_sin_moment(x_moment+y_moment-k-l, c+s, cos_moment+sin_moment-c-s);
                    result += coeff * l1_moment * l2_moment;
                }
            }
        }
    }

    return result;
}
