#ifndef JFNK_H
#define JFNK_H

#include <finite_volume/conserved_quantities.h>
#include <gas/flow_state.h>
#include <gas/transport_properties.h>
#include <io/io.h>
#include <linear_algebra/linear_system.h>
#include <solvers/cfl.h>
#include <util/numeric_types.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <linear_algebra/gmres.h>

using json = nlohmann::json;

class Jfnk {
public:
    Jfnk(std::unique_ptr<CflSchedule>&& cfl, size_t max_steps);

    Jfnk(json config);
    
    void step(std::shared_ptr<LinearSystem> system, ConservedQuantities<Ibis::dual>& cq);

    void solve(std::shared_ptr<LinearSystem> system, Sim<Ibis::dual>& sim);

    size_t max_steps() const { return max_steps_; }

private:
    size_t max_steps_;
    std::unique_ptr<CflSchedule> cfl_;

    Gmres gmres_;

    Ibis::Vector<Ibis::real> x_;
};

#endif
