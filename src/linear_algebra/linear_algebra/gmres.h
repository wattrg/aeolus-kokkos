#ifndef GMRES_H
#define GMRES_H

// #include <linear_algebra/linear_solver.h>
#include <linear_algebra/dense_linear_algebra.h>
#include <linear_algebra/linear_system.h>
#include <util/numeric_types.h>
#include <util/types.h>

#include <Kokkos_Core.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct GmresResult {
    GmresResult();
    GmresResult(bool success, size_t n_iters, Ibis::real tol, Ibis::real residual);

    bool success;
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
    using HostExecSpace = Ibis::DefaultHostExecSpace;

public:
    Gmres(){};

    Gmres(std::shared_ptr<LinearSystem> system, const size_t max_iters, Ibis::real tol);

    Gmres(std::shared_ptr<LinearSystem> system, json config);

    GmresResult solve(std::shared_ptr<LinearSystem> system, Ibis::Vector<Ibis::real>& x0);

private:
    // configuration
    size_t max_iters_;
    size_t num_vars_;
    Ibis::real tol_;

public:  // this has to be public to access from inside kernels
    // memory
    Ibis::Matrix<Ibis::real> krylov_vectors_;
    Ibis::Vector<Ibis::real> v_;
    // Ibis::Vector<Ibis::real> z_;
    Ibis::Vector<Ibis::real> r0_;
    Ibis::Vector<Ibis::real> w_;

    // least squares problem
    Ibis::Matrix<Ibis::real, HostExecSpace> H0_;
    Ibis::Matrix<Ibis::real, HostExecSpace> H1_;
    Ibis::Matrix<Ibis::real, HostExecSpace> Q0_;
    Ibis::Matrix<Ibis::real, HostExecSpace> Q1_;
    Ibis::Matrix<Ibis::real, HostExecSpace> Omega_;
    Ibis::Vector<Ibis::real, HostExecSpace> g0_;
    Ibis::Vector<Ibis::real, HostExecSpace> g1_;
    Ibis::Vector<Ibis::real, HostExecSpace> ym_host_;
    Ibis::Vector<Ibis::real> ym_;
    Ibis::Vector<Ibis::real, HostExecSpace> h_rotated_;

    // implementation
    void compute_r0_(std::shared_ptr<LinearSystem> system, Ibis::Vector<Ibis::real>& x0);
    void apply_rotations_to_hessenberg_(size_t j);
};

#endif
