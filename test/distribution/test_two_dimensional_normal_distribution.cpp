#include <gtest/gtest.h>
#include <random>
#include <iostream>

#include "distribution/two_dimensional_normal_distribution.h"

TEST(TwoDimensionalNormalDistribution, NonPositiveDefinite)
{
    const double epsilon = 0.01;

    // No Initialization
    {
        TwoDimensionalNormalDistribution dist;
        EXPECT_THROW(dist.calc_xy_cos_y_sin_y_moment(1, 1, 0, 0), std::runtime_error);
    }

    {
        const Eigen::Vector2d mean{5.0, M_PI/4.0};
        Eigen::Matrix2d covariance;
        covariance << std::pow(2.0, 2), 5.5,
                5.5, std::pow(M_PI/8.0, 2);

        EXPECT_THROW(TwoDimensionalNormalDistribution(mean, covariance), std::runtime_error);
    }

    {
        const Eigen::Vector2d mean{5.0, M_PI/4.0};
        Eigen::Matrix2d covariance;
        covariance << std::pow(2.0, 2), 0.5,
                      0.5, std::pow(M_PI/8.0, 2);

        EXPECT_NO_THROW(TwoDimensionalNormalDistribution(mean, covariance));
    }
}

TEST(TwoDimensionalNormalDistribution, FIRST_ORDER)
{
    const double epsilon = 0.01;

    const Eigen::Vector2d mean{5.0, M_PI/4.0};
    Eigen::Matrix2d covariance;
    covariance << std::pow(2.0, 2), 0.5,
            0.5, std::pow(M_PI/8.0, 2);

    TwoDimensionalNormalDistribution dist(mean, covariance);
    // mean and covariance
    for(int i=0; i<2; ++i) {
        const auto exact_mean = dist.calc_mean(i);
        const auto exact_cov = dist.calc_covariance(i);
        EXPECT_NEAR(exact_mean, mean(i), epsilon);
        EXPECT_NEAR(exact_cov, covariance(i, i), epsilon);
    }

    // moment
    for(int i=0; i<2; ++i) {
        NormalDistribution ans_dist(mean(i), covariance(i, i));
        for(int moment=1; moment<5; ++moment) {
            const auto exact_moment = dist.calc_moment(i, moment);
            EXPECT_NEAR(exact_moment, ans_dist.calc_moment(moment), epsilon);
        }
        if(i == 0)
            EXPECT_NEAR(dist.calc_xy_cos_y_sin_y_moment(3, 0, 0, 0), ans_dist.calc_moment(3), epsilon);
        else
            EXPECT_NEAR(dist.calc_xy_cos_y_sin_y_moment(0, 3, 0, 0), ans_dist.calc_moment(3), epsilon);
    }
}

TEST(TwoDimensionalNormalDistribution, SECOND_ORDER)
{
    const double epsilon = 0.01;

    const Eigen::Vector2d mean{5.0, M_PI/4.0};
    Eigen::Matrix2d covariance;
    covariance << std::pow(2.0, 2), 0.5,
                  0.5, std::pow(M_PI/8.0, 2);

    TwoDimensionalNormalDistribution dist(mean, covariance);

    // E[XY]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(1, 1, 0, 0);
        EXPECT_NEAR(exact_moment, 4.427334, epsilon);
    }

    // E[Xsin(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(1, 0, 0, 1);
       EXPECT_NEAR(exact_moment, 3.6007520, epsilon);
    }

    // E[Xcos(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(1, 0, 1, 0);
        EXPECT_NEAR(exact_moment, 2.9459, epsilon);
    }
}

TEST(TwoDimensionalNormalDistribution, SECOND_ORDER_INDEPENDENT)
{
    const double epsilon = 0.001;

    const Eigen::Vector2d mean{5.0, M_PI/4.0};
    Eigen::Matrix2d covariance;
    covariance << std::pow(2.0, 2), 0.0,
                  0.0, std::pow(M_PI/8.0, 2);

    TwoDimensionalNormalDistribution dist(mean, covariance);

    // E[XY]
    {
        const auto ans = 5.0 * M_PI/4.0;
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(1, 1, 0, 0);
        EXPECT_NEAR(exact_moment, ans, epsilon);
    }

    // E[Xsin(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(1, 0, 0, 1);
        EXPECT_NEAR(exact_moment, 3.273071163, epsilon);
    }

    // E[Xcos(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(1, 0, 1, 0);
        EXPECT_NEAR(exact_moment, 3.27286161, epsilon);
    }
}

TEST(TwoDimensionalNormalDistribution, THIRD_ORDER)
{
    const double epsilon = 0.01;

    const Eigen::Vector2d mean{5.0, M_PI/4.0};
    Eigen::Matrix2d covariance;
    covariance << std::pow(2.0, 2), 0.5,
                  0.5, std::pow(M_PI/8.0, 2);

    TwoDimensionalNormalDistribution dist(mean, covariance);

    // E[XXY]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(2, 1, 0, 0);
        EXPECT_NEAR(exact_moment, 27.778, epsilon);
    }

    // E[XYY]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(1, 2, 0, 0);
        EXPECT_NEAR(exact_moment, 4.6411128, epsilon);
    }

    // E[XXsin(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(2, 0, 0, 1);
        EXPECT_NEAR(exact_moment, 22.096993326, epsilon);
    }

    // E[XXcos(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(2, 0, 1, 0);
        EXPECT_NEAR(exact_moment, 15.54862050, epsilon);
    }

    // E[XYsin(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(1, 1, 0, 1);
        EXPECT_NEAR(exact_moment, 3.6097508, epsilon);
    }

    // E[XYcos(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(1, 1, 1, 0);
        EXPECT_NEAR(exact_moment, 2.0859199, epsilon);
    }

}

TEST(TwoDimensionalNormalDistribution, THIRD_ORDER_INDEPENDENT)
{
    const double epsilon = 0.01;

    const Eigen::Vector2d mean{5.0, M_PI/4.0};
    Eigen::Matrix2d covariance;
    covariance << std::pow(2.0, 2), 0.0,
                  0.0, std::pow(M_PI/8.0, 2);

    TwoDimensionalNormalDistribution dist(mean, covariance);

    // E[XXY]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(2, 1, 0, 0);
        EXPECT_NEAR(exact_moment, 22.775529781007346, epsilon);
    }

    // E[XYY]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(1, 2, 0, 0);
        EXPECT_NEAR(exact_moment, 3.85534762284069, epsilon);
    }

    // E[XXsin(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(2, 0, 0, 1);
        EXPECT_NEAR(exact_moment, 18.983258184618798, epsilon);
    }

    // E[XXcos(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(2, 0, 1, 0);
        EXPECT_NEAR(exact_moment, 18.98155516034286, epsilon);
    }

    // E[XYsin(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(1, 1, 0, 1);
        EXPECT_NEAR(exact_moment, 3.075471848944058, epsilon);
    }

    // E[XYcos(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(1, 1, 1, 0);
        EXPECT_NEAR(exact_moment, 2.0658474494695342, epsilon);
    }
}

TEST(TwoDimensionalNormalDistribution, FOURTH_ORDER)
{
    const double epsilon = 0.01;

    const Eigen::Vector2d mean{5.0, M_PI/4.0};
    Eigen::Matrix2d covariance;
    covariance << std::pow(2.0, 2), 0.5,
                  0.5, std::pow(M_PI/8.0, 2);

    TwoDimensionalNormalDistribution dist(mean, covariance);

    // E[XXYY]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(2, 2, 0, 0);
        EXPECT_NEAR(exact_moment, 30.71297264, epsilon);
    }

    // E[XXcos(Y)cos(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(2, 0, 2, 0);
        EXPECT_NEAR(exact_moment, 10.826096268554286, epsilon);
    }

    // E[XXsin(Y)sin(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(2, 0, 0, 2);
        EXPECT_NEAR(exact_moment, 18.16853514471178, epsilon);
    }

    // E[XXcos(Y)sin(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(2, 0, 1, 1);
        EXPECT_NEAR(exact_moment, 10.284920731528517, epsilon);
    }

    // E[Xsin(Y)sin(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(1, 0, 0, 2);
        EXPECT_NEAR(exact_moment, 2.8671537809845917, epsilon);
    }

    // E[Xcos(Y)cos(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(1, 0, 2, 0);
        EXPECT_NEAR(exact_moment, 2.1328447016537084, epsilon);
    }

    // E[Xcos(Y)sin(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(1, 0, 1, 1);
        EXPECT_NEAR(exact_moment, 1.8366293429964922, epsilon);
    }

    // E[XXXcos(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(3, 0, 1, 0);
        EXPECT_NEAR(exact_moment, 90.25668275632225, epsilon);
    }

    // E[XXXsin(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(3, 0, 0, 1);
        EXPECT_NEAR(exact_moment, 147.043554590238, epsilon);
    }

    // E[Xcos(Y)cos(Y)cos(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(1, 0, 3, 0);
        EXPECT_NEAR(exact_moment, 1.63520185069028, epsilon);
    }

    // E[Xsin(Y)sin(Y)sin(Y)]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(1, 0, 0, 3);
        EXPECT_NEAR(exact_moment, 2.3915735952085293, epsilon);
    }
}

TEST(TwoDimensionalNormalDistribution, FOURTH_ORDER_INDEPENDENT)
{
    const double epsilon = 0.01;

    // E[XXYY]
    {
        const Eigen::Vector2d mean{5.0, M_PI/4.0};
        Eigen::Matrix2d covariance;
        covariance << std::pow(2.0, 2), 0.0,
                0.0, std::pow(M_PI/8.0, 2);
        TwoDimensionalNormalDistribution dist(mean, covariance);

        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(2, 2, 0, 0);
        EXPECT_NEAR(exact_moment, 22.360585709185308, epsilon);
    }
}

TEST(TwoDimensionalNormalDistribution, HIGHER_ORDER_INDEPENDENT)
{
    const double epsilon = 0.01;
    const Eigen::Vector2d mean{0.2, 0.5};
    Eigen::Matrix2d covariance;
    covariance << std::pow(0.1, 2), 0.05*0.05,
            0.05*0.05, std::pow(0.1, 2);
    TwoDimensionalNormalDistribution dist(mean, covariance);

    // E[X^3Y^3]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(3, 3, 0, 0);
        EXPECT_NEAR(exact_moment, 0.0022635750534243286, epsilon);
    }

    // E[X^3Y^4]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(3, 4, 0, 0);
        EXPECT_NEAR(exact_moment, 0.0013110546012807863, epsilon);
    }

    // E[X^4Y^3]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(4, 3, 0, 0);
        EXPECT_NEAR(exact_moment, 0.000716658203829099, epsilon);
    }

    // E[X^4Y^4]
    {
        const auto exact_moment = dist.calc_xy_cos_y_sin_y_moment(4, 4, 0, 0);
        EXPECT_NEAR(exact_moment, 0.0004188272681060508, epsilon);
    }
}
