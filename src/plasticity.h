#pragma once
#include "interfaces.h"
#include <tuple>

class NormVM
{
public:
    NormVM(Constraint c)
    {
        _q = Dim::Q(c);
        _P.setZero(_q, _q);

        const double f = 6.;

        if (_q == 1)
        {
            _P(0, 0) = 2. / 3.;
        }
        else if (_q == 3)
        {
            _P << 2, -1, 0, -1, 2, 0, 0, 0, f;
            _P *= 1. / 3.;
        }
        else if (_q == 4)
        {
            _P << 2, -1, -1, 0, -1, 2, -1, 0, -1, -1, 2, 0, 0, 0, 0, f;
            _P *= 1. / 3.;
        }
        else if (_q == 6)
        {
            // kein bock.
        }
    }

    std::pair<double, Eigen::VectorXd> Call(Eigen::VectorXd ss) const
    {
        if (_q == 1)
        {
            Eigen::VectorXd m;
            m.resize(_q);
            m[0] = 1;
            if (ss[0] < 0.)
                m[0] = -1;
            return {std::abs(ss[0]), m};
        }
        else
        {
            assert(ss.rows() == _q);
            double se = std::sqrt(1.5 * ss.transpose() * _P * ss);
            if (se == 0)
            {
                return {se, Eigen::MatrixXd::Zero(_q, _q)};
            }
            else
            {
                return {se, 1.5 / se * _P * ss};
            }
        }
    }

    int _q;
    Eigen::MatrixXd _P;
};

class RateIndependentHistory
{
public:
    // Attributes
    Eigen::MatrixXd _p;
    Eigen::MatrixXd _dp_dsig;
    Eigen::MatrixXd _dp_dk;
    
    // Constructor
    RateIndependentHistory()
    {
    _p.setOnes(1, 1);
    _dp_dsig.setZero(1, 1);
    _dp_dk.setZero(1, 1);
    }
    
    // Call
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> Call(Eigen::VectorXd sigma, double kappa)
    {
    return {_p, _dp_dsig, _dp_dk};
    }
};