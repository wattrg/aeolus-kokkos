#ifndef CFL_H
#define CFL_H

#include <util/numeric_types.h>

#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class CflSchedule {
public:
    CflSchedule() {}

    virtual ~CflSchedule() {}

    virtual Ibis::real eval(Ibis::real t) = 0;

    virtual bool residual_based() const { return false; }
};

class ConstantSchedule : public CflSchedule {
public:
    ConstantSchedule() {}

    ConstantSchedule(Ibis::real cfl);

    Ibis::real eval(Ibis::real t);

private:
    Ibis::real cfl_;
};

class LinearSchedule : public CflSchedule {
public:
    LinearSchedule() {}

    LinearSchedule(json schedule);

    Ibis::real eval(Ibis::real t);

private:
    std::vector<Ibis::real> times_;
    std::vector<Ibis::real> cfls_;
};

class ResidualBasedCfl : public CflSchedule {
public:
    ResidualBasedCfl() {}

    ResidualBasedCfl(Ibis::real threshold, Ibis::real power,
                     Ibis::real start_cfl, Ibis::real max_cfl);

    ResidualBasedCfl(json config);

    Ibis::real eval(Ibis::real t);

    bool residual_based() const { return true; }

private:
    Ibis::real threshold_;
    Ibis::real power_;
    Ibis::real start_cfl_;
    Ibis::real max_cfl_;

    Ibis::real previous_cfl_;
    Ibis::real previous_residual_;
};

std::unique_ptr<CflSchedule> make_cfl_schedule(json config);

#endif
