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
         std::unique_ptr<CflSchedule>&&,
         std::shared_ptr<ConservedQuantities<Ibis::dual>> resiudals, json config);

    int initialise();

    LinearSolveResult step(std::shared_ptr<Sim<Ibis::dual>>& sim,
                           ConservedQuantities<Ibis::dual>& cq,
                           FlowStates<Ibis::dual>& fs);

    void solve(std::shared_ptr<Sim<Ibis::dual>>& sim);

    size_t max_steps() const { return max_steps_; }

    Ibis::real pseudo_time_step_size() const { return stable_dt_; }

    ConservedQuantitiesNorm<Ibis::dual> residual_norms() const { return residual_norms_; }

    ConservedQuantitiesNorm<Ibis::dual> relative_residual_norms() const {
        return residual_norms_ / initial_residual_norms_;
    }

    Ibis::real target_residual() const { return tolerance_; }

    const LinearSolveResult& last_gmres_result() const { return last_gmres_result_; }

private:
    std::shared_ptr<PseudoTransientLinearSystem> system_;
    std::shared_ptr<PseudoTransientLinearSystem> preconditioner_;
    std::unique_ptr<CflSchedule> cfl_;
    std::unique_ptr<IterativeLinearSolver> gmres_;
    Ibis::Vector<Ibis::real> dU_;

    size_t max_steps_;
    Ibis::real tolerance_;
    Ibis::real stable_dt_;

    std::shared_ptr<ConservedQuantities<Ibis::dual>> residuals_;
    ConservedQuantitiesNorm<Ibis::dual> residual_norms_;
    ConservedQuantitiesNorm<Ibis::dual> initial_residual_norms_;
    LinearSolveResult last_gmres_result_;

    void set_pseudo_time_step_size(Ibis::real dt_star);

public:  // this is public to appease NVCC
    void apply_update_(std::shared_ptr<Sim<Ibis::dual>>& sim,
                       ConservedQuantities<Ibis::dual>& cq, FlowStates<Ibis::dual>& fs);
};

#endif
