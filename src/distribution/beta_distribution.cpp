#include "distribution/beta_distribution.h"
#include <cmath>

constexpr double TOLERANCE = 1.0e-10;
std::complex<double> hypergeom(const double a, const double b, const double t)
{
    const std::complex<double> i(0.0, 1.0);

    double alpha = a;
    double beta = b;
    double n = 1.0;
    auto term = alpha * (i*t) / beta;
    auto value = 1.0 + term;

    while ( std::abs( term ) > TOLERANCE )
    {
        alpha += 1.0;
        beta += 1.0;
        n += 1.0;
        term *= alpha * (i*t) / beta / n;
        value += term;
    }

    return value;
}

std::complex<double> first_diff_hypergeom(const double a, const double b, const double t)
{
    const std::complex<double> i(0.0, 1.0);
    return a / b * i * hypergeom(a+1.0, b+1.0, t);
}

std::complex<double> second_diff_hypergeom(const double a, const double b, const double t)
{
    return -(a*(a+1.0)) / (b*(b+1.0)) * hypergeom(a+2.0, b+2.0, t);
}

std::complex<double> third_diff_hypergeom(const double a, const double b, const double t)
{
    const std::complex<double> i(0.0, 1.0);
    return -i * (a*(a+1.0)*(a+2.0)) / (b*(b+1.0)*(b+2.0)) * hypergeom(a+3.0, b+3.0, t);
}

std::complex<double> fourth_diff_hypergeom(const double a, const double b, const double t)
{
    return (a*(a+1.0)*(a+2.0)*(a+3.0)) / (b*(b+1.0)*(b+2.0)*(b+3.0)) * hypergeom(a+4.0, b+4.0, t);
}

BetaDistribution::BetaDistribution(const double alpha, const double beta) : alpha_(alpha), beta_(beta)
{
}

double BetaDistribution::calc_mean()
{
    return alpha_/(alpha_+beta_);
}

double BetaDistribution::calc_variance()
{
    return  (alpha_*beta_)/((alpha_+beta_)*(alpha_+beta_)*(alpha_+beta_+1.0));
}

std::complex<double> BetaDistribution::calc_characteristic(const int t)
{
    return hypergeom(alpha_, alpha_+beta_, t);
}

std::complex<double> BetaDistribution::calc_first_diff_characteristic(const int t)
{
    return first_diff_hypergeom(alpha_, alpha_+beta_, t);
}

std::complex<double> BetaDistribution::calc_second_diff_characteristic(const int t)
{
    return second_diff_hypergeom(alpha_, alpha_+beta_, t);
}

std::complex<double> BetaDistribution::calc_third_diff_characteristic(const int t)
{
    return third_diff_hypergeom(alpha_, alpha_+beta_, t);
}

std::complex<double> BetaDistribution::calc_fourth_diff_characteristic(const int t)
{
    return fourth_diff_hypergeom(alpha_, alpha_+beta_, t);
}
