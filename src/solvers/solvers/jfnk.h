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
    Jfnk() {}
    
    Jfnk(std::shared_ptr<LinearSystem> system, std::unique_ptr<CflSchedule>&&, json config);
    
    void step(ConservedQuantities<Ibis::dual>& cq,  FlowStates<Ibis::dual>& fs);

    void solve(Sim<Ibis::dual>& sim);

    size_t max_steps() const { return max_steps_; }

private:
    size_t max_steps_;

    std::shared_ptr<LinearSystem> system_;

    std::unique_ptr<CflSchedule> cfl_;

    Gmres gmres_;

    Ibis::Vector<Ibis::real> x_;
};

#endif
