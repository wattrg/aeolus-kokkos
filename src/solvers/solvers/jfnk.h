#ifndef JFNK_H
#define JFNK_H

#include <finite_volume/conserved_quantities.h>
#include <gas/flow_state.h>
#include <gas/transport_properties.h>
#include <io/io.h>
#include <linear_algebra/gmres.h>
#include <linear_algebra/linear_system.h>
#include <solvers/cfl.h>
#include <solvers/transient_linear_system.h>
#include <util/numeric_types.h>

#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class Jfnk {
public:
    Jfnk() {}

    Jfnk(std::shared_ptr<PseudoTransientLinearSystem> system,
         std::unique_ptr<CflSchedule>&&, json config);

    int initialise();

    GmresResult step(std::shared_ptr<Sim<Ibis::dual>>& sim,
                     ConservedQuantities<Ibis::dual>& cq, FlowStates<Ibis::dual>& fs);

    void solve(std::shared_ptr<Sim<Ibis::dual>>& sim);

    size_t max_steps() const { return max_steps_; }

    Ibis::real pseudo_time_step_size() const { return stable_dt_; };

    Ibis::real global_residual() const { return std::numeric_limits<Ibis::real>::max(); };

    Ibis::real target_residual() const { return tolerance_; };

private:
    std::shared_ptr<PseudoTransientLinearSystem> system_;
    std::unique_ptr<CflSchedule> cfl_;
    Gmres gmres_;
    Ibis::Vector<Ibis::real> dU_;

    size_t max_steps_;
    Ibis::real tolerance_;
    Ibis::real global_residual_;
    Ibis::real stable_dt_;

public:  // this is public to appease NVCC
    void apply_update_(std::shared_ptr<Sim<Ibis::dual>>& sim,
                       ConservedQuantities<Ibis::dual>& cq, FlowStates<Ibis::dual>& fs);
};

#endif
