#include <doctest/doctest.h>
#include <linear_algebra/dense_linear_algebra.h>
#include <linear_algebra/gmres.h>
#include <linear_algebra/linear_system.h>

#include "util/types.h"

using HostExecSpace = Ibis::DefaultHostExecSpace;

void compute_r0_(std::shared_ptr<LinearSystem> system, Ibis::Vector<Ibis::real>& x0,
                 Ibis::Vector<Ibis::real> r0, Ibis::Vector<Ibis::real> w) {
    system->matrix_vector_product(x0, w);

    auto rhs = system->rhs();
    size_t num_vars = system->num_vars();
    Kokkos::parallel_for(
        "Gmres::b-Ax0", num_vars,
        KOKKOS_LAMBDA(const size_t i) { r0(i) = rhs(i) - w(i); });
}

void apply_rotations_to_hessenberg_(Ibis::Matrix<Ibis::real, HostExecSpace> H0,
                                    Ibis::Matrix<Ibis::real, HostExecSpace> H1,
                                    Ibis::Matrix<Ibis::real, HostExecSpace> Q0,
                                    Ibis::Matrix<Ibis::real, HostExecSpace> Q1,
                                    Ibis::Matrix<Ibis::real, HostExecSpace> Omega,
                                    Ibis::Vector<Ibis::real, HostExecSpace> g0,
                                    Ibis::Vector<Ibis::real, HostExecSpace> g1,
                                    Ibis::Vector<Ibis::real, HostExecSpace> hr,
                                    size_t j) {
    // progressively rotate the Hessenberg into the QR factorisation using
    // plane rotations, on the cpu
    if (j != 0) {
        // apply previous rotations to the new column of the Hessenberg
        auto Q_sub = Q0.sub_matrix(0, j + 1, 0, j + 1);
        auto h_col_j = H0.sub_matrix(0, j + 1, 0, j + 1).column(j);
        auto h_rotated = hr.sub_vector(0, j + 1);
        Ibis::gemv(Q_sub, h_col_j, h_rotated);
        h_col_j.deep_copy_layout(h_rotated);
    }

    // build the rotation matrix for this step
    Omega.set_to_identity();
    Ibis::real denom = Ibis::sqrt(H0(j, j) * H0(j, j) + H0(j + 1, j) * H0(j + 1, j));
    Ibis::real si = H0(j + 1, j) / denom;
    Ibis::real ci = H0(j, j) / denom;
    Omega(j, j) = ci;
    Omega(j, j + 1) = si;
    Omega(j + 1, j) = -si;
    Omega(j + 1, j + 1) = ci;

    // rotate the hessenberg matrix and the right hand side
    auto H_old = H0.sub_matrix(0, j + 2, 0, j + 2);
    auto H_new = H1.sub_matrix(0, j + 2, 0, j + 2);
    auto Omega_sub = Omega.sub_matrix(0, j + 2, 0, j + 2);

    auto g = g0.sub_vector(0, j + 2);
    auto g_new = g1.sub_vector(0, j + 2);
    Ibis::gemm(Omega_sub, H_old, H_new);
    Ibis::gemv(Omega_sub, g, g_new);

    // and update the global rotation matrix
    auto Q_new = Q1.sub_matrix(0, j + 2, 0, j + 2);
    auto Q_old = Q0.sub_matrix(0, j + 2, 0, j + 2);
    if (j == 0) {
        Q_new.deep_copy(Omega_sub);
    } else {
        Ibis::gemm(Omega_sub, Q_old, Q_new);
    }

    g.deep_copy_layout(g_new);
    Q_old.deep_copy(Q_new);
    H_old.deep_copy(H_new);
}

LinearSolveResult::LinearSolveResult(bool success_, size_t n_iters_, Ibis::real tol_,
                                     Ibis::real residual_)
    : success(success_), n_iters(n_iters_), tol(tol_), residual(residual_) {}

LinearSolveResult::LinearSolveResult() : LinearSolveResult(false, 0, -1.0, -1.0) {}

Gmres::Gmres(std::shared_ptr<LinearSystem> system, const size_t max_iters,
             Ibis::real tol) {
    tol_ = tol;
    num_vars_ = system->num_vars();
    max_iters_ = Kokkos::min(num_vars_, max_iters);

    // least squares problem
    H0_ =
        Ibis::Matrix<Ibis::real, HostExecSpace>("Gmres::H0", max_iters_ + 1, max_iters_);
    H1_ =
        Ibis::Matrix<Ibis::real, HostExecSpace>("Gmres::H1", max_iters_ + 1, max_iters_);
    Q0_ = Ibis::Matrix<Ibis::real, HostExecSpace>("Gmres::Q0", max_iters_ + 1,
                                                  max_iters_ + 1);
    Q1_ = Ibis::Matrix<Ibis::real, HostExecSpace>("Gmres::Q1", max_iters_ + 1,
                                                  max_iters_ + 1);
    Omega_ = Ibis::Matrix<Ibis::real, HostExecSpace>("Gmres::Gamma", max_iters_ + 1,
                                                     max_iters_ + 1);
    g0_ = Ibis::Vector<Ibis::real, HostExecSpace>("Gmres::g0", max_iters_ + 1);
    g1_ = Ibis::Vector<Ibis::real, HostExecSpace>("Gmres::g1", max_iters_ + 1);
    h_rotated_ =
        Ibis::Vector<Ibis::real, HostExecSpace>("Gmres::h_rotated", max_iters_ + 1);
    ym_host_ = Ibis::Vector<Ibis::real, HostExecSpace>("Gmres::ym_h", max_iters_ + 1);
    ym_ = Ibis::Vector<Ibis::real>("Gmres::ym_d", max_iters_ + 1);

    // Krylov subspace and memory for Arnoldi procedure
    krylov_vectors_ =
        Ibis::Matrix<Ibis::real>("Gmres::krylov_vectors", num_vars_, max_iters_ + 1);
    r0_ = Ibis::Vector<Ibis::real>("Gmres::r0", num_vars_);
    w_ = Ibis::Vector<Ibis::real>("Gmres::w", num_vars_);
    v_ = Ibis::Vector<Ibis::real>("Gmres::v", num_vars_);

    system_ = system;
}

Gmres::Gmres(std::shared_ptr<LinearSystem> system, json config)
    : Gmres(system, config.at("max_iters"), config.at("tol")) {}

LinearSolveResult Gmres::solve(Ibis::Vector<Ibis::real>& x0) {
    // zero out (or set to identity matrix) memory
    Q0_.set_to_identity();
    Q1_.set_to_identity();
    Omega_.set_to_identity();
    H0_.set_to_zero();
    H1_.set_to_zero();
    g0_.zero();
    g1_.zero();

    // initialise the intial residuals and first krylov vector
    compute_r0_(system_, x0, r0_, w_);
    Ibis::real beta = Ibis::norm2(r0_);
    if (beta < tol_) {
        // the initial guess is within the tolerance
        return LinearSolveResult{true, 0, tol_, beta};
    }
    g0_(0) = beta;
    Ibis::scale(r0_, v_, 1.0 / beta);
    krylov_vectors_.column(0).deep_copy_layout(v_);

    LinearSolveResult result{false, 0, tol_, beta};
    for (size_t j = 0; j < max_iters_; j++) {
        // build the next krylov vector and entries in the Hessenberg matrix
        system_->matrix_vector_product(v_, w_);
        for (size_t i = 0; i < j + 1; i++) {
            auto vi = krylov_vectors_.column(i);
            H0_(i, j) = Ibis::dot(w_, vi);
            Ibis::add_scaled_vector(w_, vi, -H0_(i, j));
        }
        H0_(j + 1, j) = Ibis::norm2(w_);
        Ibis::scale(w_, v_, 1.0 / H0_(j + 1, j));
        krylov_vectors_.column(j + 1).deep_copy_layout(v_);

        // progressively rotate the Hessenberg into upper-triangular form
        // so we can calculate the residual of this step, and later solve
        // the least squares problem
        apply_rotations_to_hessenberg_(H0_, H1_, Q0_, Q1_, Omega_, g0_, g1_, h_rotated_,
                                       j);

        // check convergence
        Ibis::real residual = Ibis::abs(g0_(j + 1));
        result.residual = residual;
        result.n_iters = j + 1;
        if (residual < tol_) {
            result.success = true;
            break;
        }
    }

    // return the guess, even if we didn't converge
    size_t n_vectors = result.n_iters;
    auto H = H0_.sub_matrix(0, n_vectors, 0, n_vectors);
    auto V = krylov_vectors_.columns(0, n_vectors);
    auto g = g0_.sub_vector(0, n_vectors);
    auto ym_host = ym_host_.sub_vector(0, n_vectors);
    auto ym = ym_.sub_vector(0, n_vectors);
    Ibis::upper_triangular_solve(H, ym_host, g);
    ym.deep_copy_space(ym_host);
    Ibis::gemv(V, ym, w_);
    Ibis::add_scaled_vector(x0, w_, 1.0);

    return result;
}

FGmres::FGmres(std::shared_ptr<LinearSystem> system, const size_t max_iters,
               Ibis::real tol, std::shared_ptr<LinearSystem> precondition_system,
               const size_t max_precondition_iters, Ibis::real precondition_tol) {
    tol_ = tol;
    num_vars_ = system->num_vars();
    max_iters_ = Kokkos::min(num_vars_, max_iters);

    // least squares problem
    H0_ =
        Ibis::Matrix<Ibis::real, HostExecSpace>("FGmres::H0", max_iters_ + 1, max_iters_);
    H1_ =
        Ibis::Matrix<Ibis::real, HostExecSpace>("FGmres::H1", max_iters_ + 1, max_iters_);
    Q0_ = Ibis::Matrix<Ibis::real, HostExecSpace>("FGmres::Q0", max_iters_ + 1,
                                                  max_iters_ + 1);
    Q1_ = Ibis::Matrix<Ibis::real, HostExecSpace>("FGmres::Q1", max_iters_ + 1,
                                                  max_iters_ + 1);
    Omega_ = Ibis::Matrix<Ibis::real, HostExecSpace>("FGmres::Gamma", max_iters_ + 1,
                                                     max_iters_ + 1);
    g0_ = Ibis::Vector<Ibis::real, HostExecSpace>("FGmres::g0", max_iters_ + 1);
    g1_ = Ibis::Vector<Ibis::real, HostExecSpace>("FGmres::g1", max_iters_ + 1);
    h_rotated_ =
        Ibis::Vector<Ibis::real, HostExecSpace>("FGmres::h_rotated", max_iters_ + 1);
    ym_host_ = Ibis::Vector<Ibis::real, HostExecSpace>("FGmres::ym_h", max_iters_ + 1);
    ym_ = Ibis::Vector<Ibis::real>("FGmres::ym_d", max_iters_ + 1);

    // Krylov subspace and memory for Arnoldi procedure
    krylov_vectors_ =
        Ibis::Matrix<Ibis::real>("FGmres::krylov_vectors", num_vars_, max_iters_ + 1);
    preconditioned_krylov_vectors_ = Ibis::Matrix<Ibis::real>(
        "FGmres::preconditioned_krylov_vectors", num_vars_, max_iters_ + 1);
    r0_ = Ibis::Vector<Ibis::real>("FGmres::r0", num_vars_);
    w_ = Ibis::Vector<Ibis::real>("FGmres::w", num_vars_);
    v_ = Ibis::Vector<Ibis::real>("FGmres::v", num_vars_);
    z_ = Ibis::Vector<Ibis::real>("FFGmres::z", num_vars_);

    // The linear system of equations
    system_ = system;

    // The preconditioner system of equations, and gmres to solve it
    precondition_system_ = precondition_system;
    preconditioner_ =
        Gmres(precondition_system, max_precondition_iters, precondition_tol);
}

FGmres::FGmres(std::shared_ptr<LinearSystem> system,
               std::shared_ptr<LinearSystem> preconditioner, json config)
    : FGmres(system, config.at("max_iters"), config.at("tol"), preconditioner,
             config.at("max_precondition_iters"), config.at("precondition_tol")) {}

LinearSolveResult FGmres::solve(Ibis::Vector<Ibis::real>& x) {
    // zero out (or set to identity matrix) memory
    Q0_.set_to_identity();
    Q1_.set_to_identity();
    Omega_.set_to_identity();
    H0_.set_to_zero();
    H1_.set_to_zero();
    g0_.zero();
    g1_.zero();

    // initialise the intial residuals and first krylov vector
    compute_r0_(system_, x, r0_, w_);
    Ibis::real beta = Ibis::norm2(r0_);
    if (beta < tol_) {
        // the initial guess is within the tolerance
        return LinearSolveResult{true, 0, tol_, beta};
    }
    g0_(0) = beta;
    Ibis::scale(r0_, v_, 1.0 / beta);
    krylov_vectors_.column(0).deep_copy_layout(v_);

    LinearSolveResult result{false, 0, tol_, beta};
    for (size_t j = 0; j < max_iters_; j++) {
        // solve the precondition system
        precondition_system_->set_rhs(v_);
        preconditioner_.solve(z_);
        preconditioned_krylov_vectors_.column(j).deep_copy_layout(z_);

        // build the next krylov vector and entries in the Hessenberg matrix
        system_->matrix_vector_product(z_, w_);
        for (size_t i = 0; i < j + 1; i++) {
            auto vi = krylov_vectors_.column(i);
            H0_(i, j) = Ibis::dot(w_, vi);
            Ibis::add_scaled_vector(w_, vi, -H0_(i, j));
        }
        H0_(j + 1, j) = Ibis::norm2(w_);
        Ibis::scale(w_, v_, 1.0 / H0_(j + 1, j));
        krylov_vectors_.column(j + 1).deep_copy_layout(v_);

        // progressively rotate the Hessenberg into upper-triangular form
        // so we can calculate the residual of this step, and later solve
        // the least squares problem
        apply_rotations_to_hessenberg_(H0_, H1_, Q0_, Q1_, Omega_, g0_, g1_, h_rotated_,
                                       j);

        // check convergence
        Ibis::real residual = Ibis::abs(g0_(j + 1));
        result.residual = residual;
        result.n_iters = j + 1;
        if (residual < tol_) {
            result.success = true;
            break;
        }
    }

    // return the guess, even if we didn't converge
    size_t n_vectors = result.n_iters;
    auto H = H0_.sub_matrix(0, n_vectors, 0, n_vectors);
    auto Z = preconditioned_krylov_vectors_.columns(0, n_vectors);
    auto g = g0_.sub_vector(0, n_vectors);
    auto ym_host = ym_host_.sub_vector(0, n_vectors);
    auto ym = ym_.sub_vector(0, n_vectors);
    Ibis::upper_triangular_solve(H, ym_host, g);
    ym.deep_copy_space(ym_host);
    Ibis::gemv(Z, ym, w_);
    Ibis::add_scaled_vector(x, w_, 1.0);

    return result;
}

TEST_CASE("GMRES") {
    class TestLinearSystem : public LinearSystem {
    public:
        using ExecSpace = Kokkos::DefaultExecutionSpace;

        TestLinearSystem() {
            matrix_ = Ibis::Matrix<Ibis::real, ExecSpace>("A", 5, 5);
            auto matrix_h = matrix_.host_mirror();
            matrix_h(0, 0) = 2.0;
            matrix_h(0, 1) = -0.5;
            matrix_h(1, 0) = -1.0;
            matrix_h(1, 1) = 2.0;
            matrix_h(1, 2) = -0.5;
            matrix_h(2, 1) = -1.0;
            matrix_h(2, 2) = 2.0;
            matrix_h(2, 3) = -0.5;
            matrix_h(3, 2) = -1.0;
            matrix_h(3, 3) = 2.0;
            matrix_h(3, 4) = -0.5;
            matrix_h(4, 3) = -1.0;
            matrix_h(4, 4) = 2.0;
            matrix_.deep_copy_space(matrix_h);

            rhs_ = Ibis::Vector<Ibis::real, ExecSpace>("rhs", 5);
            auto rhs_h = rhs_.host_mirror();
            rhs_h(0) = 2.0;
            rhs_h(1) = -0.5;
            rhs_h(2) = -2.75;
            rhs_h(3) = 3.75;
            rhs_h(4) = -0.5;
            rhs_.deep_copy_space(rhs_h);
        }

        ~TestLinearSystem() {}

        void eval_rhs() {}

        void set_rhs(Ibis::Vector<Ibis::real>& rhs) {
            precondition_rhs_.deep_copy_space(rhs);
        }

        void matrix_vector_product(Ibis::Vector<Ibis::real>& vec,
                                   Ibis::Vector<Ibis::real>& res) {
            Ibis::gemv(matrix_, vec, res);
        }

        KOKKOS_INLINE_FUNCTION
        Ibis::real& rhs(const size_t i) const { return rhs_(i); }

        KOKKOS_INLINE_FUNCTION
        Ibis::real& rhs(const size_t i, const size_t j) const {
            (void)j;
            return rhs_(i);
        }

        Ibis::Vector<Ibis::real>& rhs() { return rhs_; }

        size_t num_vars() const { return 5; }

    private:
        Ibis::Matrix<Ibis::real, ExecSpace> matrix_;
        Ibis::Vector<Ibis::real, ExecSpace> rhs_;
        Ibis::Vector<Ibis::real, ExecSpace> precondition_rhs_;
    };

    std::shared_ptr<LinearSystem> sys{new TestLinearSystem()};

    Gmres solver{sys, 5, 1e-14};
    Ibis::Vector<Ibis::real> x{"x", 5};
    LinearSolveResult result = solver.solve(x);

    auto x_h = x.host_mirror();
    x_h.deep_copy_space(x);

    CHECK(result.success == true);
    CHECK(x_h(0) == doctest::Approx(1.0));
    CHECK(x_h(1) == doctest::Approx(0.0));
    CHECK(x_h(2) == doctest::Approx(-1.0));
    CHECK(x_h(3) == doctest::Approx(1.5));
    CHECK(x_h(4) == doctest::Approx(0.5));
}

TEST_CASE("FGMRES") {
    class TestLinearSystem : public LinearSystem {
    public:
        using ExecSpace = Kokkos::DefaultExecutionSpace;

        TestLinearSystem(Ibis::Matrix<Ibis::real> matrix) {
            matrix_ = matrix;
            rhs_ = Ibis::Vector<Ibis::real>("rhs", 5);
            auto rhs_h = rhs_.host_mirror();
            rhs_h(0) = 2.0;
            rhs_h(1) = 0.0;
            rhs_h(2) = -3.5;
            rhs_h(3) = 3.5;
            rhs_h(4) = -0.5;
            rhs_.deep_copy_space(rhs_h);
        }

        ~TestLinearSystem() {}

        void eval_rhs() {}

        void set_rhs(Ibis::Vector<Ibis::real>& rhs) { rhs_.deep_copy_space(rhs); }

        void matrix_vector_product(Ibis::Vector<Ibis::real>& vec,
                                   Ibis::Vector<Ibis::real>& res) {
            Ibis::gemv(matrix_, vec, res);
        }

        KOKKOS_INLINE_FUNCTION
        Ibis::real& rhs(const size_t i) const { return rhs_(i); }

        KOKKOS_INLINE_FUNCTION
        Ibis::real& rhs(const size_t i, const size_t j) const {
            (void)j;
            return rhs_(i);
        }

        Ibis::Vector<Ibis::real>& rhs() { return rhs_; }

        size_t num_vars() const { return 5; }

    private:
        Ibis::Matrix<Ibis::real, ExecSpace> matrix_;
        Ibis::Vector<Ibis::real, ExecSpace> rhs_;
    };

    Ibis::Matrix<Ibis::real> matrix("A", 5, 5);
    auto matrix_h = matrix.host_mirror();
    matrix_h(0, 0) = 2.0;
    matrix_h(0, 1) = -1.0;
    matrix_h(1, 0) = -1.0;
    matrix_h(1, 1) = 2.0;
    matrix_h(1, 2) = -1.0;
    matrix_h(2, 1) = -1.0;
    matrix_h(2, 2) = 2.0;
    matrix_h(2, 3) = -1.0;
    matrix_h(3, 2) = -1.0;
    matrix_h(3, 3) = 2.0;
    matrix_h(3, 4) = -1.0;
    matrix_h(4, 3) = -1.0;
    matrix_h(4, 4) = 2.0;
    matrix.deep_copy_space(matrix_h);

    std::shared_ptr<LinearSystem> sys{new TestLinearSystem(matrix)};

    // the preconditioner we'll use is just the diagonal terms
    Ibis::Matrix<Ibis::real> precondition_matrix("P", 5, 5);
    auto preconditioner_h = precondition_matrix.host_mirror();
    preconditioner_h(0, 0) = 2.0;
    preconditioner_h(1, 1) = 2.0;
    preconditioner_h(2, 2) = 2.0;
    preconditioner_h(3, 3) = 2.0;
    preconditioner_h(4, 4) = 2.0;
    precondition_matrix.deep_copy_space(preconditioner_h);
    std::shared_ptr<LinearSystem> preconditioner{
        new TestLinearSystem(precondition_matrix)};

    FGmres solver{sys, 5, 1e-14, preconditioner, 2, 1e-1};
    Ibis::Vector<Ibis::real> x{"x", 5};
    LinearSolveResult result = solver.solve(x);

    auto x_h = x.host_mirror();
    x_h.deep_copy_space(x);

    CHECK(result.success == true);
    CHECK(x_h(0) == doctest::Approx(1.0));
    CHECK(x_h(1) == doctest::Approx(0.0));
    CHECK(x_h(2) == doctest::Approx(-1.0));
    CHECK(x_h(3) == doctest::Approx(1.5));
    CHECK(x_h(4) == doctest::Approx(0.5));
}
