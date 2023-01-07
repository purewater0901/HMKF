#include "distribution/base_distribution.h"
#include "utilities.h"

double BaseDistribution::calc_moment(const int order)
{
    const auto result = calc_diff_characteristic(0, order);

    std::complex<double> ideal(0.0, 1.0);
    std::complex<double> ideal_coeff(1.0, 0.0);
    for(int i=1; i<=order; ++i)
        ideal_coeff *= ideal;

    return (result/ideal_coeff).real();
}

double BaseDistribution::calc_cos_moment(const int order)
{
    double result = 0.0;
    for(int k=0; k<=order; ++k)
    {
        const auto coeff = static_cast<double>(nCr(order, k));
        const int char_arg = 2*k - order;
        result += coeff * calc_characteristic(char_arg).real();
    }

    return result / std::pow(2.0, order);
}

double BaseDistribution::calc_sin_moment(const int order)
{
    std::complex<double> result(0.0, 0.0);
    for(int k=0; k<=order; ++k)
    {
        const double coeff = static_cast<double>(nCr(order, k)) * std::pow(-1.0, order - k);
        const int char_arg = 2*k - order;
        result += coeff * calc_characteristic(char_arg);
    }

    std::complex<double> ideal(0.0, -0.5);
    std::complex<double> ideal_coeff(1.0, 0.0);
    for(int i=1; i<=order; ++i)
        ideal_coeff *= ideal;

    return (ideal_coeff * result).real();
}

double BaseDistribution::calc_cos_sin_moment(const int cos_order, const int sin_order)
{
    std::complex<double> result(0.0, 0.0);
    for(int k1=0; k1<=cos_order; ++k1)
    {
        for(int k2=0; k2<=sin_order; ++k2)
        {
            const double coeff_cos = nCr(cos_order, k1);
            const double coeff_sin = nCr(sin_order, k2) * std::pow(-1.0, sin_order - k2);
            const int char_arg = 2*(k1+k2)-cos_order-sin_order;
            result += coeff_cos * coeff_sin * calc_characteristic(char_arg);
        }
    }

    std::complex<double> ideal(0.0, -1.0);
    std::complex<double> ideal_coeff(1.0, 0.0);
    for(int i=1; i<=sin_order; ++i)
        ideal_coeff *= ideal;
    const double real_coeff = std::pow(2.0, cos_order+sin_order);

    return ((ideal_coeff / real_coeff) * result).real();
}

double BaseDistribution::calc_x_cos_moment(const int x_order, const int cos_order)
{
    std::complex<double> result(0.0, 0.0);
    for(int k=0; k<=cos_order; ++k)
    {
        const auto coeff = static_cast<double>(nCr(cos_order, k));
        const int char_arg = 2*k - cos_order;
        result += coeff * calc_diff_characteristic(char_arg, x_order);

    }

    std::complex<double> ideal(0.0, 1.0);
    std::complex<double> ideal_coeff(1.0, 0.0);
    for(int i=1; i<=x_order; ++i)
        ideal_coeff *= ideal;
    const double real_coeff = std::pow(2.0, cos_order);

    return (result / (real_coeff*ideal_coeff)).real();
}

double BaseDistribution::calc_x_sin_moment(const int x_order, const int sin_order)
{
    std::complex<double> result(0.0, 0.0);
    for(int k=0; k<=sin_order; ++k)
    {
        const auto coeff = static_cast<double>(nCr(sin_order, k)) * std::pow(-1.0, sin_order-k);
        const int char_arg = 2*k - sin_order;
        result += coeff * calc_diff_characteristic(char_arg, x_order);
    }

    std::complex<double> i(0.0, 1.0);
    std::complex<double> ideal_coeff = std::pow(i, x_order+sin_order);
    const double real_coeff = std::pow(2.0, sin_order);

    return (result / (real_coeff*ideal_coeff)).real();
}

double BaseDistribution::calc_x_cos_sin_moment(const int x_order, const int cos_order, const int sin_order)
{
    std::complex<double> result(0.0, 0.0);
    for(int k1=0; k1<=cos_order; ++k1)
    {
        for(int k2=0; k2<=sin_order; ++k2)
        {
            const auto cos_coeff = static_cast<double>(nCr(cos_order, k1));
            const auto sin_coeff = static_cast<double>(nCr(sin_order, k2)) * std::pow(-1.0, sin_order-k2);
            const int char_arg = 2*(k1+k2) - cos_order - sin_order;
            result += cos_coeff * sin_coeff * calc_diff_characteristic(char_arg, x_order);
        }
    }

    std::complex<double> ideal(0.0, 1.0);
    std::complex<double> ideal_coeff(1.0, 0.0);
    for(int i=1; i<=x_order+sin_order; ++i)
        ideal_coeff *= ideal;
    const double real_coeff = std::pow(2.0, cos_order + sin_order);

    return (result / (real_coeff*ideal_coeff)).real();
}
