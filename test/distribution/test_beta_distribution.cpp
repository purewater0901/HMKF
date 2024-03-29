#include <gtest/gtest.h>
#include <random>
#include <iostream>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "distribution/beta_distribution.h"

namespace {
    std::vector<double> getBetaDistributionSamples(const double alpha,
                                                  const double beta,
                                                  const int num_sample)
    {
        size_t seed = 1234567890;
        boost::random::mt19937 engine(seed);
        boost::function<double()> rand_beta =
                boost::bind(boost::random::beta_distribution<>(alpha, beta), engine);

        std::vector<double> samples(num_sample);
        for(int i=0; i<num_sample; ++i) {
            samples.at(i) = rand_beta();
        }

        return samples;
    }
} // namespace

const double epsilon = 0.0001;

// Monte Carlo Simulation
const int num_sample = 10000*10000;

TEST(BetaDistribution, X_MOMENT)
{
    const double alpha = 0.5;
    const double beta = 0.5;
    BetaDistribution dist(alpha, beta);
    const auto samples = getBetaDistributionSamples(alpha, beta, num_sample);

    // First Order
    {
        // exact
        const auto exact_moment = dist.calc_moment(1);
        EXPECT_NEAR(exact_moment, alpha / (alpha + beta), epsilon);
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

TEST(BetaDistribution, TRIGONOMETRIC_MOMENT)
{
    const double alpha = 5.0;
    const double beta = 0.5;
    BetaDistribution dist(alpha, beta);
    const auto samples = getBetaDistributionSamples(alpha, beta, num_sample);

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

TEST(BetaDistribution, MIXED_TRIGONOMETRIC_MOMENT)
{
    const double alpha = 4.0;
    const double beta = 2.1;
    BetaDistribution dist(alpha, beta);
    const auto samples = getBetaDistributionSamples(alpha, beta, num_sample);

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

TEST(BetaDistribution, ExponentialMoment)
{
    const double alpha = 2.0;
    const double beta = 2.0;
    BetaDistribution dist(alpha, beta);
    const auto samples = getBetaDistributionSamples(alpha, beta, num_sample);

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

        EXPECT_NEAR(exact_moment, monte_carlo_moment, 0.1);
    }
}

TEST(BetaDistribution, ExponentialTrigonoemtricMoment)
{
    const double alpha = 2.0;
    const double beta = 2.0;
    BetaDistribution dist(alpha, beta);
    const auto samples = getBetaDistributionSamples(alpha, beta, num_sample);

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

        EXPECT_NEAR(exact_moment, monte_carlo_moment, 0.1);
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

        EXPECT_NEAR(exact_moment, monte_carlo_moment, epsilon);
    }
}