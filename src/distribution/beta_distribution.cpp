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

std::complex<double> diff_hypergeom(const double a, const double b, const double t, const int order)
{
    const std::complex<double> i(0.0, 1.0);
    double a_coeff = a;
    double b_coeff = b;
    for(size_t n=1; n<order; ++n) {
        a_coeff *= a+static_cast<double>(n);
        b_coeff *= b+static_cast<double>(n);
    }
    return a_coeff / b_coeff * std::pow(i, order) * hypergeom(a+order, b+order, t);
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

std::complex<double> BetaDistribution::calc_diff_characteristic(const int t, const int order)
{
    if(order==0) {
        return calc_characteristic(order);
    }

    return diff_hypergeom(alpha_, alpha_+beta_, t, order);
}
