#ifndef GMRES_H
#define GMRES_H

#include <linear_algebra/linear_solver.h>
#include <util/numeric_types.h>
#include <util/types.h>

#include <Kokkos_Core.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct GmresResult {
    bool succes;
    size_t n_iters;
    Ibis::real tol;
};

class Gmres {
public:
    Gmres(const SystemLinearisation& system, const size_t max_iters, Ibis::real tol);

    Gmres(const SystemLinearisation& system, json config);

    GmresResult solve(SystemLinearisation& system, Ibis::Vector<Ibis::real>& x0,
                      Ibis::Vector<Ibis::real>& b);

private:
    // configuration
    size_t max_iters_;
    size_t tol_;
    size_t num_vars_;

    // memory
    Ibis::Matrix<Ibis::real> krylov_vectors_;
    Ibis::Vector<Ibis::real> r0_;

    // least squares problem
    Ibis::Matrix<Ibis::real> H0_;
    Ibis::Matrix<Ibis::real> H1_;
    Ibis::Matrix<Ibis::real> Q0_;
    Ibis::Matrix<Ibis::real> Q1_;
    Ibis::Matrix<Ibis::real> Gamma_;

    // implementation
    void compute_r0_(SystemLineariastion& system, Ibis::Vector<Ibis::real>& x0);
};

class FGmres {
public:
    FGmres(const SystemLinearisation& system, const size_t max_iters, Ibis::real tol);

    GmresResult solve(SystemLinearisation& system);

private:
    size_t max_iters_;
    size_t tol_;
};

#endif
