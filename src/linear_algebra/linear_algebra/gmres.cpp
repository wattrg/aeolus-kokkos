#include <doctest/doctest.h>
#include <linear_algebra/gmres.h>

#include "linear_algebra/linear_system.h"

Gmres::Gmres(const SystemLinearisation& system, const size_t max_iters, Ibis::real tol) {
    tol_ = tol;
    max_iters_ = max_iters;
    num_vars_ = system.num_vars();

    H0_ = Ibis::Matrix<Ibis::real>("Gmres::H0", max_iters_ + 1, max_iters_);
    H1_ = Ibis::Matrix<Ibis::real>("Gmres::H1", max_iters_ + 1, max_iters_);
    Gamma_ = Ibis::Matrix<Ibis::real>("Gmres::Gamma", max_iters_ + 1, max_iters_ + 1);
    krylov_vectors_ =
        Ibis::Matrix<Ibis::real>("Gmres::krylov_vectors", num_vars_, max_iters_);
    r0_ = Ibis::Vector<Ibis::real>("Gmres::r0", num_vars_);
    
}

Gmres::Gmres(const SystemLinearisation& system, json config)
    : Gmres(system, config.at("max_iters"), config.at("tolerance")) {}

GmresResult Gmres::solve(SystemLinearisation& system, Ibis::Vector<Ibis::real>& x0,
                          Ibis::Vector<Ibis::real>& b) {
    // r0_ = 
}

void Gmres::compute_r0_(SystemLinearisation& system, Ibis::Vector<Ibis::real>& x0) {
    // system.matrix_vector_product(, , , , , )
}

TEST_CASE("GMRES") {
    //
}
