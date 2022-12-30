#ifndef UNCERTAINTY_PROPAGATION_TWO_DIMENSIONAL_NORMAL_DISTRIBUTION_H
#define UNCERTAINTY_PROPAGATION_TWO_DIMENSIONAL_NORMAL_DISTRIBUTION_H

#include <iostream>
#include <vector>
#include <Eigen/Eigen>
#include <complex>

#include "distribution/normal_distribution.h"
#include "utilities.h"

class TwoDimensionalNormalDistribution
{
public:
    TwoDimensionalNormalDistribution() = default;
    TwoDimensionalNormalDistribution(const Eigen::Vector2d& mean, const Eigen::Matrix2d& covariance);

    bool checkPositiveDefiniteness(const Eigen::Matrix2d& covariance);
    bool initializeData();

    void setValues(const Eigen::Vector2d& mean, const Eigen::Matrix2d& covariance);

    double calc_mean(const int dim);
    double calc_covariance(const int dim);
    double calc_moment(const int dim, const int moment);
    double calc_xy_cos_y_sin_y_moment(const int x_moment, const int y_moment, const int cos_moment, const int sin_moment);

    Eigen::Vector2d mean_;
    Eigen::Matrix2d covariance_;

    Eigen::Vector2d eigen_values_;
    Eigen::Matrix2d T_;
    bool independent_ = false;
    bool initialization_ = false;
};

#endif //UNCERTAINTY_PROPAGATION_TWO_DIMENSIONAL_NORMAL_DISTRIBUTION_H
