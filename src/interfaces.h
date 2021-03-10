#pragma once
#include <eigen3/Eigen/Core>
#include <exception>
#include <vector>
#include <numeric>
#include <memory>

enum Constraint
{
    UNIAXIAL_STRAIN,
    UNIAXIAL_STRESS,
    PLANE_STRAIN,
    PLANE_STRESS,
    FULL
};

enum Q
{
    EPS,
    E,
    SIGMA,
    DSIGMA_DEPS,
    DSIGMA_DE,
    EEQ,
    DEEQ,
    KAPPA,
    LAST
};

class QValues
{
public:
    QValues() = default;

    //! @brief stores n x rows x cols values where n is the number of IPs
    QValues(int rows, int cols = 1)
        : _rows(rows)
        , _cols(cols)
    {
    }

    void Resize(int n)
    {
        data.setZero(n * _rows * _cols);
    }

    void Set(double value, int i)
    {
        assert(_rows == 1);
        assert(_cols == 1);
        data[i] = value;
    }

    void Set(Eigen::MatrixXd value, int i)
    {
        assert(value.rows() == _rows);
        assert(value.cols() == _cols);
        data.segment(_rows * _cols * i, _rows * _cols) = Eigen::Map<Eigen::VectorXd>(value.data(), value.size());
    }

    double GetScalar(int i) const
    {
        assert(_rows == 1);
        assert(_cols == 1);
        return data[i];
    }

    Eigen::MatrixXd Get(int i) const
    {
        Eigen::VectorXd ip_values = data.segment(_rows * _cols * i, _rows * _cols);
        return Eigen::Map<Eigen::MatrixXd>(ip_values.data(), _rows, _cols);
    }

    bool IsUsed() const
    {
        return _rows != 0;
    }


    // private:
    int _rows = 0;
    int _cols = 0;
    Eigen::VectorXd data;
};

struct Dim
{
    static constexpr int G(Constraint c)
    {
        if (c == UNIAXIAL_STRAIN)
            return 1;
        if (c == UNIAXIAL_STRESS)
            return 1;
        if (c == PLANE_STRAIN)
            return 2;
        if (c == PLANE_STRESS)
            return 2;
        if (c == FULL)
            return 3;
        static_assert(true, "Constraint type not supported.");
        return -1;
    }

    static constexpr int Q(Constraint c)
    {
        if (c == UNIAXIAL_STRAIN)
            return 1;
        if (c == UNIAXIAL_STRESS)
            return 1;
        if (c == PLANE_STRAIN)
            return 3;
        if (c == PLANE_STRESS)
            return 3;
        if (c == FULL)
            return 6;
        static_assert(true, "Constraint type not supported.");
        return -1;
    }
};


template <Constraint TC>
using V = Eigen::Matrix<double, Dim::Q(TC), 1>;

template <Constraint TC>
using M = Eigen::Matrix<double, Dim::Q(TC), Dim::Q(TC)>;


struct LawInterface
{
    virtual void DefineOutputs(std::vector<QValues>& out) const = 0;
    virtual void DefineInputs(std::vector<QValues>& input) const = 0;
    virtual void Evaluate(const std::vector<QValues>& input, std::vector<QValues>& out, int i) = 0;
    virtual void Update(const std::vector<QValues>& input, int i)
    {
    }
    virtual void Resize(int n)
    {
    }
};

class MechanicsLaw
{
public:
    MechanicsLaw(Constraint constraint)
        : _constraint(constraint)
    {
    }

    virtual std::pair<Eigen::VectorXd, Eigen::MatrixXd> Evaluate(const Eigen::VectorXd& strain, int i = 0) = 0;

    virtual void Update(const Eigen::VectorXd& strain, int i = 0)
    {
    }

    virtual void Resize(int n)
    {
    }

    const Constraint _constraint;
};

class MechanicsLawAdapter : public LawInterface
{
public:
    MechanicsLawAdapter(std::shared_ptr<MechanicsLaw> law)
        : _law(law)
    {
    }

    void DefineOutputs(std::vector<QValues>& out) const override
    {
        const int q = Dim::Q(_law->_constraint);
        out[SIGMA] = QValues(q);
        out[DSIGMA_DEPS] = QValues(q, q);
    }

    void DefineInputs(std::vector<QValues>& input) const override
    {
        input[EPS] = QValues(Dim::Q(_law->_constraint));
    }

    void Evaluate(const std::vector<QValues>& input, std::vector<QValues>& out, int i) override
    {
        auto eval = _law->Evaluate(input[EPS].Get(i), i);
        out[SIGMA].Set(eval.first, i);
        out[DSIGMA_DEPS].Set(eval.second, i);
    }
    void Update(const std::vector<QValues>& input, int i) override
    {
        _law->Update(input[EPS].Get(i), i);
    }
    void Resize(int n) override
    {
        _law->Resize(n);
    }

private:
    std::shared_ptr<MechanicsLaw> _law;
};


class IpLoop
{
public:
    IpLoop()
    {
        _outputs.resize(Q::LAST);
        _inputs.resize(Q::LAST);
    }

    void AddLaw(std::shared_ptr<LawInterface> law, std::vector<int> ips)
    {
        _laws.push_back(law);
        _ips.push_back(ips);
        law->DefineInputs(_inputs);
        law->DefineOutputs(_outputs);

        if (_n != 0)
            Resize(_n);
    }

    void AddLaw(std::shared_ptr<MechanicsLaw> law, std::vector<int> ips)
    {
        auto law_interface = std::make_shared<MechanicsLawAdapter>(law);
        AddLaw(law_interface, ips);
    }

    virtual void Resize(int n)
    {
        _n = n;
        for (auto& qvalues : _outputs)
            qvalues.Resize(n);

        for (auto& law : _laws)
            law->Resize(_n);
    }

    Eigen::VectorXd Get(Q what)
    {
        return _outputs.at(what).data;
    }

    std::vector<Q> RequiredInputs() const
    {
        std::vector<Q> required;
        for (unsigned iQ = 0; iQ < _inputs.size(); ++iQ)
        {
            Q q = static_cast<Q>(iQ);
            if (_inputs[q].IsUsed())
                required.push_back(q);
        }
        return required;
    }

    virtual void Evaluate(const Eigen::VectorXd& all_strains, const Eigen::VectorXd& all_neeq)
    {
        FixIPs();

        _inputs[E].data = all_neeq;
        _inputs[EPS].data = all_strains;
        for (unsigned iLaw = 0; iLaw < _laws.size(); ++iLaw)
            for (int ip : _ips[iLaw])
                _laws[iLaw]->Evaluate(_inputs, _outputs, ip);
    }

    virtual void Update(const Eigen::VectorXd& all_strains, const Eigen::VectorXd& all_neeq)
    {
        _inputs[E].data = all_neeq;
        _inputs[EPS].data = all_strains;
        for (unsigned iLaw = 0; iLaw < _laws.size(); ++iLaw)
            for (int ip : _ips[iLaw])
                _laws[iLaw]->Update(_inputs, ip);
    }

    std::vector<std::shared_ptr<LawInterface>> _laws;
    std::vector<std::vector<int>> _ips;
    std::vector<QValues> _outputs;
    std::vector<QValues> _inputs;
    int _n = 0;

private:
    void FixIPs()
    {
        // Actually, there is only one case to fix:
        if (_laws.size() == 1 and _ips[0].empty())
        {
            auto& v = _ips[0];
            v.resize(_n);
            std::iota(v.begin(), v.end(), 0);
        }

        // The rest are checks.
        int total_num_ips = 0;
        for (const auto& v : _ips)
            total_num_ips += v.size();
        if (total_num_ips != _n)
            throw std::runtime_error("The IPs numbers don't match!");

        // complete check if all IPs have a law.
        std::vector<bool> all(_n, false);
        for (const auto& v : _ips)
        {
            for (int ip : v)
            {
                if (all[ip])
                    throw std::runtime_error("Ip is there at least twice!");

                all[ip] = true;
            }
        }
        for (int ip = 0; ip < _n; ++ip)
        {
            if (not all[ip])
            {
                throw std::runtime_error("Ip has no law!");
            }
        }
    }
};

