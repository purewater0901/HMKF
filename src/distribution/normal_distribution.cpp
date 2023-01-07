#include "distribution/normal_distribution.h"

NormalDistribution::NormalDistribution(const double mean, const double variance) : mean_(mean), variance_(variance)
{
}

double NormalDistribution::calc_mean()
{
    return mean_;
}

double NormalDistribution::calc_variance()
{
    return  variance_;
}

std::complex<double> NormalDistribution::calc_characteristic(const int t)
{
    if(t == 0)
    {
        return {1.0, 0.0};
    }

    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);
    return std::exp(i*t_double*mean_ - variance_*std::pow(t_double, 2)*0.5);
}

std::complex<double> NormalDistribution::calc_first_diff_characteristic(const int t)
{
    if(t == 0)
    {
        return {0.0, mean_};
    }

    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);
    const auto tmp = i*mean_ - variance_*t_double;
    return tmp * calc_characteristic(t);
}

std::complex<double> NormalDistribution::calc_diff_characteristic(const int t, const int order)
{
    if(order==0) {
        return calc_characteristic(t);
    }

    if(order == 1) {
        return calc_first_diff_characteristic(t);
    }

    const std::complex<double> i(0.0, 1.0);
    const auto t_double = static_cast<double>(t);
    const auto tmp = (i*mean_ - variance_*t_double);

    return -(order-1) * variance_ * calc_diff_characteristic(t, order-2) + tmp* calc_diff_characteristic(t, order-1);
}
