#ifndef GMRES_H
#define GMRES_H

// #include <linear_algebra/linear_solver.h>
#include <linear_algebra/linear_system.h>
#include <linear_algebra/vector.h>
#include <util/numeric_types.h>

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
    Gmres(const std::shared_ptr<LinearSystem> system, const size_t max_iters,
          Ibis::real tol);

    Gmres(const std::shared_ptr<LinearSystem> system, json config);

    GmresResult solve(std::shared_ptr<LinearSystem> system, Ibis::Vector<Ibis::real>& x0);

private:
    // configuration
    size_t max_iters_;
    size_t tol_;
    size_t num_vars_;

public:  // this has to be public to access from inside kernels
    // memory
    Ibis::Matrix<Ibis::real> krylov_vectors_;
    Ibis::Vector<Ibis::real> v_;
    // Ibis::Vector<Ibis::real> z_;
    Ibis::Vector<Ibis::real> r0_;
    Ibis::Vector<Ibis::real> w_;

    // least squares problem
    Ibis::Matrix<Ibis::real> H0_;
    Ibis::Matrix<Ibis::real> H1_;
    Ibis::Matrix<Ibis::real> Q0_;
    Ibis::Matrix<Ibis::real> Q1_;
    Ibis::Matrix<Ibis::real> Gamma_;

    // implementation
    void compute_r0_(std::shared_ptr<LinearSystem> system, Ibis::Vector<Ibis::real>& x0);
};

#endif
