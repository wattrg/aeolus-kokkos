#ifndef GMRES_H
#define GMRES_H

// #include <linear_algebra/linear_solver.h>
#include <linear_algebra/linear_system.h>
#include <linear_algebra/vector.h>
#include <util/numeric_types.h>

#include <Kokkos_Core.hpp>
#include <nlohmann/json.hpp>
#include "util/types.h"

using json = nlohmann::json;

struct GmresResult {
    GmresResult(bool success, size_t n_iters, Ibis::real tol, Ibis::real residual);
    
    bool succes;
    size_t n_iters;
    Ibis::real tol;
    Ibis::real residual;
};

class Gmres {
public:
    using MemSpace = Ibis::DefaultMemSpace;
    using HostMemSpace = Ibis::DefaultHostMemSpace;
    using ArrayLayout = Ibis::DefaultArrayLayout;
    using HostArrayLayout = Ibis::DefaultHostArrayLayout;

    
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
    Ibis::Matrix<Ibis::real, HostMemSpace, HostArrayLayout> H0_;
    Ibis::Matrix<Ibis::real, HostMemSpace, HostArrayLayout> H1_;
    Ibis::Matrix<Ibis::real, HostMemSpace, HostArrayLayout> Q0_;
    Ibis::Matrix<Ibis::real, HostMemSpace, HostArrayLayout> Q1_;
    Ibis::Matrix<Ibis::real, HostMemSpace, HostArrayLayout> Omega_;
    Ibis::Vector<Ibis::real, HostArrayLayout, HostMemSpace> g0_;
    Ibis::Vector<Ibis::real, HostArrayLayout, HostMemSpace> g1_;
    Ibis::Vector<Ibis::real, HostArrayLayout, HostMemSpace> h_rotated_;

    // implementation
    void compute_r0_(std::shared_ptr<LinearSystem> system, Ibis::Vector<Ibis::real>& x0);
};

#endif
