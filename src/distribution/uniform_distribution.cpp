#include "distribution/uniform_distribution.h"

UniformDistribution::UniformDistribution(const double l, const double u) : BaseDistribution(), l_(l), u_(u) {}

double UniformDistribution::calc_mean()
{
    return 0.5 * (l_ + u_);
}

double UniformDistribution::calc_variance()
{
    return  (u_ - l_) * (u_ - l_) / 12.0;
}

std::complex<double> UniformDistribution::calc_characteristic(const std::complex<double>& t)
{
    if(std::abs(t) < 1e-6)
    {
        return {1.0, 0.0};
    }

    const std::complex<double> i(0.0, 1.0);
    return (std::exp(i*t*u_) - std::exp(i*t*l_))/ (i*t*(u_-l_));
}

std::complex<double> UniformDistribution::calc_diff_characteristic(const std::complex<double>& t, const int order)
{
    if(order==0) {
        return calc_characteristic(t);
    }

    const std::complex<double> i(0.0, 1.0);
    const auto order_double = static_cast<double>(order);

    if(std::abs(t) < 1e-6) {
        return std::pow(i, order) * (std::pow(u_, order+1) - std::pow(l_, order+1)) / ((order+1)*(u_-l_));
    }

    const auto deno = std::pow(i, order-1) * (std::pow(u_, order)*std::exp(i*t*u_) - std::pow(l_, order)*std::exp(i*t*l_));
    const auto nume = t* (u_ - l_);

    return deno/nume - order_double * calc_diff_characteristic(t, order-1) / t;
}
