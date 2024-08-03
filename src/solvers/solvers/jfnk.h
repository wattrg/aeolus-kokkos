#ifndef JFNK_H
#define JFNK_H

#include <finite_volume/conserved_quantities.h>
#include <gas/flow_state.h>
#include <gas/transport_properties.h>
#include <io/io.h>
#include <linear_algebra/gmres.h>
#include <linear_algebra/linear_system.h>
#include <solvers/cfl.h>
#include <util/numeric_types.h>

#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class Jfnk {
public:
    Jfnk() {}

    Jfnk(std::unique_ptr<LinearSystem>&& system, std::unique_ptr<CflSchedule>&&,
         json config);

    void step(ConservedQuantities<Ibis::dual>& cq);

    void solve(Sim<Ibis::dual>& sim);

    size_t max_steps() const { return max_steps_; }

    Ibis::real pseudo_time_step_size() const;

    Ibis::real global_residual() const;

    Ibis::real target_residual() const;

private:
    std::unique_ptr<LinearSystem> system_;
    std::unique_ptr<CflSchedule> cfl_;
    Gmres gmres_;
    Ibis::Vector<Ibis::real> dU_;

    size_t max_steps_;
    Ibis::real target_residual_;
    Ibis::real global_residual_;

public:  // this is public to appease NVCC
    void apply_update_(ConservedQuantities<Ibis::dual>& cq);
};

#endif
