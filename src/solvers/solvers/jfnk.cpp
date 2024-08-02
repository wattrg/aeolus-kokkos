#include <solvers/jfnk.h>
#include "finite_volume/conserved_quantities.h"

Jfnk::Jfnk(const std::shared_ptr<LinearSystem> system, json config) {
    max_steps_ = config.at("max_steps");
    gmres_ = Gmres(system, config.at("gmres"));
    cfl_ = make_cfl_schedule(config.at("cfl"));
}

void Jfnk::step(ConservedQuantities<Ibis::dual>& cq) {
    x_.zero();
    gmres_.solve(system_, x_);
}
