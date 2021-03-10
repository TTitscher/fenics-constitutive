#pragma once
#include "linear_elastic.h"
#include <cmath>

struct DamageLawInterface
{
    virtual std::pair<double, double> Evaluate(double kappa) const = 0;
};

struct StrainNormInterface
{
    virtual std::pair<double, Eigen::VectorXd> Evaluate(Eigen::VectorXd strain) const = 0;
};

class DamageLawExponential : public DamageLawInterface
{
public:
    DamageLawExponential(double k0, double alpha, double beta)
        : _k0(k0)
        , _a(alpha)
        , _b(beta)
    {
    }

    std::pair<double, double> Evaluate(double k) const override
    {
        if (k <= _k0)
            return {0., 0.};
        const double omega = 1 - _k0 / k * (1 - _a + _a * std::exp(_b * (_k0 - k)));
        const double domega = _k0 / k * ((1 / k + _b) * _a * std::exp(_b * (_k0 - k)) + (1 - _a) / k);
        return {omega, domega};
    }

private:
    const double _k0;
    const double _a;
    const double _b;
};

std::pair<double, V<FULL>> InvariantI1(V<FULL> v)
{
    const double I1 = v[0] + v[1] + v[2];
    V<FULL> dI1 = V<FULL>::Zero();
    dI1.segment<3>(0) = Eigen::Vector3d::Ones();
    return {I1, dI1};
}

std::pair<double, V<FULL>> InvariantJ2(V<FULL> v)
{
    const double J2 =
            ((v[0] - v[1]) * (v[0] - v[1]) + (v[1] - v[2]) * (v[1] - v[2]) + (v[2] - v[0]) * (v[2] - v[0])) / 6. +
            0.25 * (v[3] * v[3] + v[4] * v[4] + v[5] * v[5]);
    V<FULL> dJ2;
    dJ2[0] = (2. * v[0] - v[1] - v[2]) / 3.;
    dJ2[1] = (2. * v[1] - v[2] - v[0]) / 3.;
    dJ2[2] = (2. * v[2] - v[0] - v[1]) / 3.;
    dJ2[3] = 0.5 * v[3];
    dJ2[4] = 0.5 * v[4];
    dJ2[5] = 0.5 * v[5];
    return {J2, dJ2};
}

Eigen::Matrix<double, 6, Eigen::Dynamic> T3D(double nu, Constraint c)
{
    Eigen::Matrix<double, 6, Eigen::Dynamic> T(6, Dim::Q(c));
    T.setZero();
    switch (c)
    {
    case Constraint::UNIAXIAL_STRAIN:
    {
        T(0, 0) = 1;
        break;
    }
    case Constraint::UNIAXIAL_STRESS:
    {
        T(0, 0) = 1;
        T(1, 0) = -nu;
        T(2, 0) = -nu;
        break;
    }
    case Constraint::PLANE_STRAIN:
    {
        T(0, 0) = 1;
        T(1, 1) = 1;
        T(5, 2) = 1;
        break;
    }
    case Constraint::PLANE_STRESS:
    {
        T(0, 0) = 1;
        T(1, 1) = 1;
        T(2, 0) = nu / (nu - 1.);
        T(2, 1) = nu / (nu - 1.);
        T(5, 2) = 1;
        break;
    }
    case Constraint::FULL:
    {
        T = M<FULL>::Identity();
        break;
    }
    }
    return T;
}

class ModMisesEeq : public StrainNormInterface
{
public:
    ModMisesEeq(double k, double nu, Constraint c)
        : _K1((k - 1.0) / (2.0 * k * (1.0 - 2.0 * nu)))
        , _K2(3.0 / (k * (1.0 + nu) * (1.0 + nu)))
        , _nu(nu)
        , _c(c)
        , _T3D(T3D(_nu, _c))
    {
    }

    std::pair<double, Eigen::VectorXd> Evaluate(Eigen::VectorXd strain) const
    {
        // transformation to 3D and invariants
        const V<FULL> strain3D = _T3D * strain;
        double I1, J2;
        V<FULL> dI1, dJ2;
        std::tie(I1, dI1) = InvariantI1(strain3D);
        std::tie(J2, dJ2) = InvariantJ2(strain3D);

        // actual modified mises norm
        const double A = std::sqrt(_K1 * _K1 * I1 * I1 + _K2 * J2) + 1.e-14;
        const double eeq = _K1 * I1 + A;
        const double deeq_dI1 = _K1 + _K1 * _K1 * I1 / A;
        const double deeq_dJ2 = _K2 / (2 * A);
        //
        //// derivative in 3D and transformation back
        V<FULL> deeq = deeq_dI1 * dI1 + deeq_dJ2 * dJ2;
        return {eeq, _T3D.transpose() * deeq};
    }

private:
    const double _K1;
    const double _K2;
    const double _nu;
    const Constraint _c;
    const Eigen::Matrix<double, 6, Eigen::Dynamic> _T3D;
};


class LocalDamage : public MechanicsLaw
{
public:
    LocalDamage(double E, double nu, Constraint c, std::shared_ptr<DamageLawInterface> omega,
                std::shared_ptr<StrainNormInterface> strain_norm)
        : MechanicsLaw(c)
        , _C(C(E, nu, c))
        , _omega(omega)
        , _strain_norm(strain_norm)
        , _kappa(1)
    {
    }

    void Resize(int n) override
    {
        _kappa.Resize(n);
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> Evaluate(const Eigen::VectorXd& strain, int i) override
    {
        double kappa, dkappa, omega, domega, eeq;
        Eigen::VectorXd deeq;

        std::tie(eeq, deeq) = _strain_norm->Evaluate(strain);
        std::tie(kappa, dkappa) = EvaluateKappa(eeq, _kappa.GetScalar(i));
        std::tie(omega, domega) = _omega->Evaluate(kappa);

        return {(1. - omega) * _C * strain, (1. - omega) * _C - _C * strain * domega * dkappa * deeq.transpose()};
    }

    std::pair<double, double> EvaluateKappa(double eeq, double kappa) const
    {
        if (eeq >= kappa)
            return {eeq, 1.};
        else
            return {kappa, 0};
    }

    virtual void Update(const Eigen::VectorXd& strain, int i) override
    {
        const double eeq = _strain_norm->Evaluate(strain).first;
        const double kappa = EvaluateKappa(eeq, _kappa.GetScalar(i)).first;
        _kappa.Set(kappa, i);
    }

    Eigen::VectorXd Kappa() const
    {
        return _kappa.data;
    }


private:
    Eigen::MatrixXd _C;
    std::shared_ptr<DamageLawInterface> _omega;
    std::shared_ptr<StrainNormInterface> _strain_norm;
    QValues _kappa;
};

class GradientDamage : public LawInterface
{
public:
    GradientDamage(double E, double nu, Constraint c, std::shared_ptr<DamageLawInterface> omega,
                   std::shared_ptr<StrainNormInterface> strain_norm)
        : _C(C(E, nu, c))
        , _omega(omega)
        , _strain_norm(strain_norm)
        , _kappa(1)
    {
    }

    void DefineOutputs(std::vector<QValues>& out) const override
    {
        const int q = _C.rows();
        out[EEQ] = QValues(1);
        out[DEEQ] = QValues(q);
        out[SIGMA] = QValues(q);
        out[DSIGMA_DE] = QValues(q);
        out[DSIGMA_DEPS] = QValues(q, q);
    }

    void DefineInputs(std::vector<QValues>& input) const override
    {
        const int q = _C.rows();
        input[E] = QValues(1);
        input[EPS] = QValues(q);
    }

    void Resize(int n) override
    {
        _kappa.Resize(n);
    }

    void Evaluate(const std::vector<QValues>& input, std::vector<QValues>& out, int i) override
    {
        double kappa, dkappa, omega, domega, eeq;
        Eigen::VectorXd deeq;
        auto strain = input[EPS].Get(i);

        std::tie(kappa, dkappa) = EvaluateKappa(input[E].GetScalar(i), _kappa.GetScalar(i));
        std::tie(omega, domega) = _omega->Evaluate(kappa);
        std::tie(eeq, deeq) = _strain_norm->Evaluate(strain);

        out[EEQ].Set(eeq, i);
        out[SIGMA].Set((1. - omega) * _C * strain, i);
        out[DEEQ].Set(deeq, i);
        out[DSIGMA_DE].Set(-_C * strain * domega * dkappa, i);
        out[DSIGMA_DEPS].Set((1. - omega) * _C, i);
    }

    std::pair<double, double> EvaluateKappa(double eeq, double kappa) const
    {
        if (eeq >= kappa)
            return {eeq, 1.};
        else
            return {kappa, 0};
    }

    void Update(const std::vector<QValues>& input, int i) override
    {
        _kappa.Set(EvaluateKappa(input[E].GetScalar(i), _kappa.GetScalar(i)).first, i);
    }

    Eigen::VectorXd Kappa() const
    {
        return _kappa.data;
    }


private:
    Eigen::MatrixXd _C;
    std::shared_ptr<DamageLawInterface> _omega;
    std::shared_ptr<StrainNormInterface> _strain_norm;

    // history values
    QValues _kappa;
};

