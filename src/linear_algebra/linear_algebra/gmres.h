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

struct LinearSolveResult {
    LinearSolveResult();
    LinearSolveResult(bool success, size_t n_iters, Ibis::real tol, Ibis::real residual);

    bool success;
    size_t n_iters;
    Ibis::real tol;
    Ibis::real residual;
};

class IterativeLinearSolver {
public:
    IterativeLinearSolver() {}
    virtual ~IterativeLinearSolver() {}

    virtual LinearSolveResult solve(Ibis::Vector<Ibis::real>& x) = 0;
};

class Gmres : public IterativeLinearSolver {
public:
    using MemSpace = Ibis::DefaultMemSpace;
    using HostMemSpace = Ibis::DefaultHostMemSpace;
    using ArrayLayout = Ibis::DefaultArrayLayout;
    using HostArrayLayout = Ibis::DefaultHostArrayLayout;
    using HostExecSpace = Ibis::DefaultHostExecSpace;

public:
    Gmres() {}

    ~Gmres() {}

    Gmres(std::shared_ptr<LinearSystem> system, const size_t max_iters, Ibis::real tol);

    Gmres(std::shared_ptr<LinearSystem> system, json config);

    LinearSolveResult solve(Ibis::Vector<Ibis::real>& x0);

private:
    // configuration
    size_t max_iters_;
    size_t num_vars_;
    Ibis::real tol_;
    std::shared_ptr<LinearSystem> system_;

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
};

class FGmres : public IterativeLinearSolver {
public:
    using HostExecSpace = Ibis::DefaultHostExecSpace;

public:
    FGmres() {}

    ~FGmres() {}

    FGmres(std::shared_ptr<LinearSystem> system, const size_t max_iters, Ibis::real tol,
           std::shared_ptr<LinearSystem> precondition_system,
           const size_t max_precondition_iters, Ibis::real inner_tol);

    FGmres(std::shared_ptr<LinearSystem> system,
           std::shared_ptr<LinearSystem> preconditioner, json config);

    LinearSolveResult solve(Ibis::Vector<Ibis::real>& x);

private:
    // configuration
    size_t max_iters_;
    size_t num_vars_;
    Ibis::real tol_;

    std::shared_ptr<LinearSystem> system_;
    std::shared_ptr<LinearSystem> precondition_system_;

    // the inner gmres
    Gmres preconditioner_;

public:  // this has to be public to access from inside kernels
    // memory
    Ibis::Matrix<Ibis::real> krylov_vectors_;
    Ibis::Matrix<Ibis::real> preconditioned_krylov_vectors_;
    Ibis::Vector<Ibis::real> v_;
    Ibis::Vector<Ibis::real> z_;
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
};

#endif
