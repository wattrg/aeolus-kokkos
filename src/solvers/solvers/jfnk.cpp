#include <solvers/jfnk.h>
#include "finite_volume/conserved_quantities.h"

Jfnk::Jfnk(std::unique_ptr<CflSchedule>&& cfl, size_t max_steps)
    : max_steps_(max_steps), cfl_(std::move(cfl)) {}

Jfnk::Jfnk(json config) : max_steps_(config.at("max_steps")) {
    cfl_ = make_cfl_schedule(config.at("cfl"));
}

void Jfnk::step(std::shared_ptr<LinearSystem> system, ConservedQuantities<Ibis::dual>& cq) {
    gmres_->solve(system, x_);
}
