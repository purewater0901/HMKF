#include <gtest/gtest.h>
#include <random>
#include <iostream>

#include "distribution/exponential_distribution.h"

const double epsilon = 0.01;

namespace {
    std::vector<double> getExponentialDistributionSamples(const double lambda,
                                                     const int num_sample)
    {
        std::default_random_engine generator;
        std::exponential_distribution<double> distribution(lambda);

        std::vector<double> samples(num_sample);
        for(int i=0; i<num_sample; ++i) {
            samples.at(i) = distribution(generator);
        }

        return samples;
    }
} // namespace

const double lambda = 10.0;

// Exact Distribution
ExponentialDistribution dist(lambda);

// Monte Carlo Simulation
const int num_sample = 10000*10000;
const auto samples = getExponentialDistributionSamples(lambda, num_sample);

TEST(ExponentialDistribution, X_MOMENT)
{
    // First Order
    {
        // exact
        const auto exact_moment = dist.calc_moment(1);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            sum += samples.at(i);
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    // Second Order
    {
        // exact
        const auto exact_moment = dist.calc_moment(2);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += x*x;
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    // Third Order
    {
        // exact
        const auto exact_moment = dist.calc_moment(3);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += x*x*x;
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    // Fourth Order
    {
        // exact
        const auto exact_moment = dist.calc_moment(4);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<num_sample; ++i) {
            const double x = samples.at(i);
            sum += x*x*x*x;
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, 0.01);
    }
}

TEST(ExponentialDistribution, TRIGONOMETRIC_MOMENT)
{
    // First Order
    {
        {
            // exact
            const auto exact_moment = dist.calc_sin_moment(1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_cos_moment(1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }
    }

    // Second Order
    {
        {
            // exact
            const auto exact_moment = dist.calc_sin_moment(2);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::sin(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_cos_moment(2);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x) * std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_cos_sin_moment(1, 1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }
    }

    // Third Order
    {
        {
            // exact
            const auto exact_moment = dist.calc_sin_moment(3);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::sin(x) * std::sin(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_cos_moment(3);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x) * std::cos(x) * std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_cos_sin_moment(2, 1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x) * std::cos(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }


        // exact
        {
            const auto exact_moment = dist.calc_cos_sin_moment(1, 2);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x) * std::sin(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }
    }

    // Fourth Order
    // Third Order
    {
        {
            // exact
            const auto exact_moment = dist.calc_sin_moment(4);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::sin(x) * std::sin(x) * std::sin(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_cos_moment(4);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x) * std::cos(x) * std::cos(x) * std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_cos_sin_moment(2, 2);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x) * std::cos(x) * std::sin(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }


        {
            // exact
            const auto exact_moment = dist.calc_cos_sin_moment(1, 3);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x) * std::sin(x) * std::sin(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_cos_sin_moment(3, 1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += std::cos(x) * std::cos(x) * std::cos(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }
    }
}

TEST(ExponentialDistribution, MIXED_TRIGONOMETRIC_MOMENT)
{
    // Second Order
    {
        {
            // exact
            const auto exact_moment = dist.calc_x_sin_moment(1, 1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x*std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_x_cos_moment(1, 1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }
    }

    // Third Order
    {
        {
            // exact
            const auto exact_moment = dist.calc_x_sin_moment(2, 1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * x * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }


        {
            // exact
            const auto exact_moment = dist.calc_x_sin_moment(1, 2);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * std::sin(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_x_cos_moment(2, 1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * x * std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }


        {
            // exact
            const auto exact_moment = dist.calc_x_cos_moment(1, 2);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * std::cos(x) * std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }
    }

    // Fourth Order
    {
        {
            // exact
            const auto exact_moment = dist.calc_x_sin_moment(3, 1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * x * x *std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }


        {
            // exact
            const auto exact_moment = dist.calc_x_sin_moment(1, 3);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * std::sin(x) * std::sin(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_x_cos_moment(3, 1);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * x * x * std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_x_cos_moment(1, 3);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * std::cos(x) * std::cos(x) * std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_x_sin_moment(2, 2);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * x * std::sin(x) * std::sin(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }

        {
            // exact
            const auto exact_moment = dist.calc_x_cos_moment(2, 2);

            // monte carlo
            double sum = 0.0;
            for(int i=0; i<samples.size(); ++i) {
                const double x = samples.at(i);
                sum += x * x * std::cos(x) * std::cos(x);
            }
            const double monte_carlo_moment = sum / num_sample;

            EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
        }
    }
}

TEST(ExponentialDistribution, HIGHER_ORDER)
{
    {
        // exact
        const auto exact_moment = dist.calc_moment(8);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += std::pow(x, 8);
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    {
        // exact
        const auto exact_moment = dist.calc_x_sin_moment(4, 4);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += std::pow(x, 4) * std::pow(std::sin(x), 4);
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    {
        // exact
        const auto exact_moment = dist.calc_cos_sin_moment(3, 4);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += std::pow(std::cos(x), 3) * std::pow(std::sin(x), 4);
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }
}

TEST(ExponentialDistribution, ExponentialMoment)
{
    {
        // exact
        const auto exact_moment = dist.calc_exp_moment(0);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += std::exp(0);
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    {
        // exact
        const auto exact_moment = dist.calc_exp_moment(1);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += std::exp(x);
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    {
        // exact
        const auto exact_moment = dist.calc_exp_moment(2);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += std::exp(2*x);
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    {
        // exact
        const auto exact_moment = dist.calc_exp_moment(8);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += std::exp(8*x);
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }
}

TEST(ExponentialDistribution, ExponentialTrigonoemtricMoment)
{
    {
        // exact
        const auto exact_moment = dist.calc_exp_cos_moment(3, 4);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += std::exp(3*x)*std::pow(std::cos(x), 4);
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    {
        // exact
        const auto exact_moment = dist.calc_exp_cos_moment(1, 0);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += std::exp(x);
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    {
        // exact
        const auto exact_moment = dist.calc_exp_cos_moment(0, 1);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += std::cos(x);
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    {
        // exact
        const auto exact_moment = dist.calc_exp_sin_moment(3, 4);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += std::exp(3*x)*std::pow(std::sin(x), 4);
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    {
        // exact
        const auto exact_moment = dist.calc_exp_sin_moment(1, 0);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += std::exp(x);
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    {
        // exact
        const auto exact_moment = dist.calc_exp_sin_moment(0, 1);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += std::sin(x);
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    {
        // exact
        const auto exact_moment = dist.calc_exp_x_cos_sin_moment(2, 3, 4, 5);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += std::exp(2*x) * std::pow(x, 3) * std::pow(std::cos(x), 4) * std::pow(std::sin(x), 5);
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    {
        // exact
        const auto exact_moment = dist.calc_exp_x_cos_sin_moment(10, 2, 1, 1);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += std::exp(10*x) * std::pow(x, 2) * std::pow(std::cos(x), 1) * std::pow(std::sin(x), 1);
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    {
        // exact
        const auto exact_moment = dist.calc_exp_x_cos_sin_moment(1, 1, 2, 1);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += std::exp(1*x) * std::pow(x, 1) * std::pow(std::cos(x), 2) * std::pow(std::sin(x), 1);
        }
        const double monte_carlo_moment = sum / num_sample;

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }

    {
        // exact
        const auto exact_moment = dist.calc_exp_x_cos_sin_moment(2, 3, 0, 0);

        // monte carlo
        double sum = 0.0;
        for(int i=0; i<samples.size(); ++i) {
            const double x = samples.at(i);
            sum += std::exp(2*x) * std::pow(x, 3);
        }
        const double monte_carlo_moment = sum / num_sample;

        std::cout << "exact: " << exact_moment << std::endl;
        std::cout << "montecarlo: " << monte_carlo_moment << std::endl;
        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }
}
