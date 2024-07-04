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

std::unique_ptr<CflSchedule> make_cfl_schedule(json config);

#endif
