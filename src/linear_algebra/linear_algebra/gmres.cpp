#include <doctest/doctest.h>
#include <linear_algebra/dense_linear_algebra.h>
#include <linear_algebra/gmres.h>
#include <linear_algebra/linear_system.h>

GmresResult::GmresResult(bool success_, size_t n_iters_, Ibis::real tol_,
                         Ibis::real residual_)
    : success(success_), n_iters(n_iters_), tol(tol_), residual(residual_) {}

Gmres::Gmres(const std::shared_ptr<LinearSystem> system, const size_t max_iters,
             Ibis::real tol) {
    tol_ = tol;
    max_iters_ = max_iters;
    num_vars_ = system->num_vars();

    // least squares problem
    H0_ = Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace>(
        "Gmres::H0", max_iters_ + 1, max_iters_);
    H1_ = Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace>(
        "Gmres::H1", max_iters_ + 1, max_iters_);
    Q0_ = Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace>(
        "Gmres::Q0", max_iters_ + 1, max_iters_);
    Q1_ = Ibis::Matrix<Ibis::real, Kokkos::DefaultHostExecutionSpace>(
        "Gmres::Q1", max_iters_ + 1, max_iters_);
    g0_ = Ibis::Vector<Ibis::real, Kokkos::DefaultHostExecutionSpace>("Gmres::g0",
                                                                      max_iters_ + 1);
    g1_ = Ibis::Vector<Ibis::real, Kokkos::DefaultHostExecutionSpace>("Gmres::g1",
                                                                      max_iters_ + 1);
    h_rotated_ = Ibis::Vector<Ibis::real, Kokkos::DefaultHostExecutionSpace>(
        "Gmres::h_rotated", max_iters_ + 1);
    Omega_ = Ibis::Matrix<Ibis::real>("Gmres::Gamma", max_iters_ + 1, max_iters_ + 1);
    ym_host_ = Ibis::Vector<Ibis::real, Kokkos::DefaultHostExecutionSpace>(
        "Gmres::ym_h", max_iters_ + 1);
    ym_host_ = Ibis::Vector<Ibis::real>("Gmres::ym_d", max_iters_ + 1);

    // Krylov subspace and memory for Arnoldi procedure
    krylov_vectors_ =
        Ibis::Matrix<Ibis::real>("Gmres::krylov_vectors", num_vars_, max_iters_);
    r0_ = Ibis::Vector<Ibis::real>("Gmres::r0", num_vars_);
    w_ = Ibis::Vector<Ibis::real>("Gmres::w", num_vars_);
    v_ = Ibis::Vector<Ibis::real>("Gmres::v", num_vars_);
}

Gmres::Gmres(const std::shared_ptr<LinearSystem> system, json config)
    : Gmres(system, config.at("max_iters"), config.at("tolerance")) {}

GmresResult Gmres::solve(std::shared_ptr<LinearSystem> system,
                         Ibis::Vector<Ibis::real>& x0) {
    // initialise the intial residuals and first krylov vector
    compute_r0_(system, x0);
    Ibis::real beta = Ibis::norm2(r0_);
    g0_(0) = beta;
    Ibis::scale(r0_, v_, 1.0 / beta);
    krylov_vectors_.column(0).deep_copy_layout(v_);

    GmresResult result{false, 0, tol_, beta};
    for (size_t j = 0; j < max_iters_; j++) {
        // build the next krylov vector and entries in the Hessenberg matrix
        system->matrix_vector_product(v_, w_);
        for (size_t i = 0; i < j; i++) {
            H0_(i, j) = Ibis::dot(w_, v_);
            Ibis::add_scaled_vector(w_, krylov_vectors_.column(i), -H0_(i, j));
        }
        H0_(j + 1, j) = Ibis::norm2(w_);
        Ibis::scale(w_, v_, 1.0 / H0_(j + 1, j));
        krylov_vectors_.column(j + 1).deep_copy_layout(v_);

        // progressively rotate the Hessenberg into upper-triangular form
        // so we can calculate the residual of this step, and later solve
        // the least squares problem
        apply_rotations_to_hessenberg_(j);

        // check convergence
        Ibis::real residual = Ibis::abs(g0_(j + 1));
        result.residual = residual;
        result.n_iters = j;
        if (residual < tol_) {
            result.success = true;
            break;
        }
    }

    auto H = H0_.sub_matrix(0, result.n_iters, 0, result.n_iters);
    auto V = krylov_vectors_.columns(0, result.n_iters);
    auto g = g0_.sub_vector(0, result.n_iters);
    auto w = w_.sub_vector(0, V.n_rows());
    auto ym_host = ym_host_.sub_vector(0, result.n_iters);
    auto ym = ym_.sub_vector(0, result.n_iters);

    // return the guess, even if we didn't converge
    Ibis::upper_triangular_solve(H, ym_host, g);
    ym.deep_copy_space(ym_host);
    Ibis::gemv(V, g, w);
    Ibis::add_scaled_vector(x0, w_, 1.0);

    return result;
}

void Gmres::apply_rotations_to_hessenberg_(size_t j) {
    // progressively rotate the Hessenberg into the QR factorisation using
    // plane rotations, on the cpu
    if (j != 0) {
        // apply previous rotations to the new column of the Hessenberg
        auto Q_sub = Q0_.sub_matrix(0, j + 1, 0, j + 1);
        auto h_col_j = H0_.sub_matrix(0, j + 1, 0, j + 1).column(j);
        auto h_rotated = h_rotated_.sub_vector(0, j + 1);
        Ibis::gemv(Q_sub, h_col_j, h_rotated);
        h_col_j.deep_copy_layout(h_rotated);
    }

    // build the rotation matrix for this step
    Omega_.set_to_identity();
    Ibis::real denom = Ibis::sqrt(H0_(j, j) * H0_(j, j) + H0_(j + 1, j) * H0_(j + 1, j));
    Ibis::real si = H0_(j + 1, j) / denom;
    Ibis::real ci = H0_(j, j) / denom;
    Omega_(j, j) = ci;
    Omega_(j, j + 1) = si;
    Omega_(j + 1, j) = -si;
    Omega_(j + 1, j + 1) = ci;

    // rotate the hessenberg matrix and the right hand side
    auto H = H0_.sub_matrix(0, j + 2, 0, j + 2);
    auto H_new = H1_.sub_matrix(0, j + 2, 0, j + 2);
    auto Omega = Omega_.sub_matrix(0, j + 2, 0, j + 2);
    auto g = g0_.sub_vector(0, j + 2);
    auto g_new = g1_.sub_vector(0, j + 2);
    Ibis::gemm(Omega, H, H_new);
    Ibis::gemv(Omega, g, g_new);

    // and update the global rotation matrix
    auto Q = Q0_.sub_matrix(0, j + 2, 0, j + 2);
    auto Q_new = Q1_.sub_matrix(0, j + 2, 0, j + 2);
    if (j == 0) {
        Q_new.deep_copy(Omega);
    } else {
        Ibis::gemm(Omega, Q, Q_new);
    }

    g.deep_copy_layout(g_new);
    Q.deep_copy(Q_new);
    H.deep_copy(H_new);
}

void Gmres::compute_r0_(std::shared_ptr<LinearSystem> system,
                        Ibis::Vector<Ibis::real>& x0) {
    system->matrix_vector_product(x0, w_);

    auto r0 = r0_;
    auto w = w_;
    auto rhs = system->rhs();
    Kokkos::parallel_for(
        "Gmres::b-Ax0", num_vars_, KOKKOS_LAMBDA(const int i) { r0(i) = rhs(i) - w(i); });
}

TEST_CASE("GMRES") {
    class TestLinearSystem : public LinearSystem {
    public:
        using ExecSpace = Kokkos::DefaultHostExecutionSpace;

        TestLinearSystem() {
            matrix_ = Ibis::Matrix<Ibis::real, ExecSpace>("A", 5, 5);
            matrix_(0, 0) = 1.0;
            matrix_(0, 1) = 0.5;
            matrix_(1, 0) = 0.5;
            matrix_(1, 1) = 1.0;
            matrix_(1, 2) = 0.5;
            matrix_(2, 1) = 0.5;
            matrix_(2, 2) = 1.0;
            matrix_(2, 3) = 0.5;
            matrix_(3, 2) = 0.5;
            matrix_(3, 3) = 1.0;
            matrix_(3, 4) = 0.5;
            matrix_(4, 3) = 0.5;
            matrix_(4, 4) = 1.0;

            rhs_ = Ibis::Vector<Ibis::real, ExecSpace>("rhs", 5);
            rhs_(0) = 3.0;
            rhs_(1) = 2.0;
            rhs_(2) = 1.5;
            rhs_(3) = 5.0;
            rhs_(4) = -1.0;
        }

        ~TestLinearSystem() {}

        void eval_rhs() {}

        void matrix_vector_product(Ibis::Vector<Ibis::real>& vec,
                                   Ibis::Vector<Ibis::real>& res) {
            Ibis::gemv(matrix_, vec, res);
        }

        Ibis::real& rhs(const size_t i) const { return rhs_(i); }

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

    std::shared_ptr<LinearSystem> sys{new TestLinearSystem()};

    Gmres solver{sys, 5, 1e-10};
    Ibis::Vector<Ibis::real, Kokkos::DefaultHostExecutionSpace> x{"x", 5};
    solver.solve(sys, x);

    CHECK(x(0) == doctest::Approx(0.16666666666666));
    CHECK(x(1) == doctest::Approx(6.33333333333333));
    CHECK(x(2) == doctest::Approx(-8.5));
    CHECK(x(3) == doctest::Approx(13.66666666666666));
    CHECK(x(4) == doctest::Approx(-8.83333333333333));
}
