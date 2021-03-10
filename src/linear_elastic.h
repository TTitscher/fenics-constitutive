#pragma once
#include "interfaces.h"

template <Constraint TC>
M<TC> C(double E, double nu);

template <>
M<UNIAXIAL_STRAIN> C<UNIAXIAL_STRAIN>(double E, double nu)
{
    return M<UNIAXIAL_STRAIN>::Constant(E);
}

template <>
M<UNIAXIAL_STRESS> C<UNIAXIAL_STRESS>(double E, double nu)
{
    return M<UNIAXIAL_STRAIN>::Constant(E);
}

template <>
M<PLANE_STRESS> C<PLANE_STRESS>(double E, double nu)
{
    const double C11 = E / (1 - nu * nu);
    const double C12 = nu * C11;
    const double C33 = (1 - nu) * 0.5 * C11;
    M<PLANE_STRAIN> c;
    c << C11, C12, 0, C12, C11, 0, 0, 0, C33;
    return c;
}

template <>
M<PLANE_STRAIN> C<PLANE_STRAIN>(double E, double nu)
{
    const double l = E * nu / (1 + nu) / (1 - 2 * nu);
    const double m = E / (2.0 * (1 + nu));
    M<PLANE_STRAIN> c = V<PLANE_STRAIN>({2 * m, 2 * m, m}).asDiagonal();
    c.block<2, 2>(0, 0) += Eigen::Matrix2d::Constant(l);
    return c;
}

template <>
M<FULL> C<FULL>(double E, double nu)
{
    const double l = E * nu / (1 + nu) / (1 - 2 * nu);
    const double m = E / (2.0 * (1 + nu));

    V<FULL> diagonal;
    diagonal.segment<3>(0) = Eigen::Vector3d::Constant(2 * m);
    diagonal.segment<3>(3) = Eigen::Vector3d::Constant(m);
    M<FULL> c = diagonal.asDiagonal();
    c.block<3, 3>(0, 0) += Eigen::Matrix3d::Constant(l);
    return c;
}

Eigen::MatrixXd C(double E, double nu, Constraint c)
{
    if (c == UNIAXIAL_STRAIN)
        return C<UNIAXIAL_STRAIN>(E, nu);
    if (c == UNIAXIAL_STRESS)
        return C<UNIAXIAL_STRESS>(E, nu);
    if (c == PLANE_STRAIN)
        return C<PLANE_STRAIN>(E, nu);
    if (c == PLANE_STRESS)
        return C<PLANE_STRESS>(E, nu);
    if (c == FULL)
        return C<FULL>(E, nu);
    throw std::runtime_error("Stuffy");
}

class LinearElastic : public MechanicsLaw
{
public:
    LinearElastic(double E, double nu, Constraint constraint)
        : MechanicsLaw(constraint)
    {
        _C = C(E, nu, constraint);
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> Evaluate(const Eigen::VectorXd& strain, int i = 0) override
    {
        return {_C * strain, _C};
    }

private:
    Eigen::MatrixXd _C;
};

