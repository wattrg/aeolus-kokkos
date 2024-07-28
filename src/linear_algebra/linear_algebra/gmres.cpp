#include <doctest/doctest.h>
#include <linear_algebra/gmres.h>
#include <linear_algebra/dense_linear_algebra.h>

GmresResult::GmresResult(bool success_, size_t n_iters_, Ibis::real tol_, Ibis::real residual_) 
    : succes(success_), n_iters(n_iters_), tol(tol_), residual(residual_) {}

Gmres::Gmres(const std::shared_ptr<LinearSystem> system, const size_t max_iters,
             Ibis::real tol) {
    tol_ = tol;
    max_iters_ = max_iters;
    num_vars_ = system->num_vars();

    H0_ = Ibis::Matrix<Ibis::real>("Gmres::H0", max_iters_ + 1, max_iters_);
    H1_ = Ibis::Matrix<Ibis::real>("Gmres::H1", max_iters_ + 1, max_iters_);
    Omega_ = Ibis::Matrix<Ibis::real>("Gmres::Gamma", max_iters_ + 1, max_iters_ + 1);
    krylov_vectors_ =
        Ibis::Matrix<Ibis::real>("Gmres::krylov_vectors", num_vars_, max_iters_);
    r0_ = Ibis::Vector<Ibis::real>("Gmres::r0", num_vars_);
    w_ = Ibis::Vector<Ibis::real>("Gmres::w", num_vars_);
}

Gmres::Gmres(const std::shared_ptr<LinearSystem> system, json config)
    : Gmres(system, config.at("max_iters"), config.at("tolerance")) {}

GmresResult Gmres::solve(std::shared_ptr<LinearSystem> system,
                         Ibis::Vector<Ibis::real>& x0) {
    // initialise the intial residuals and first krylov vector
    compute_r0_(system, x0);
    Ibis::real beta = Ibis::norm2(r0_);
    Ibis::scale(r0_, v_, 1.0 / beta);
    Ibis::deep_copy_vector(krylov_vectors_.column(0), v_);

    for (size_t j = 0; j < max_iters_; j++) {
        // build the next krylov vector and entries in the Hessenberg matrix
        system->matrix_vector_product(v_, w_);
        for (size_t i = 0; i < j; i++) {
            H0_(i, j) = Ibis::dot(w_, v_);
            Ibis::add_scaled_vector(w_, krylov_vectors_.column(i), -H0_(i, j));
        }
        H0_(j + 1, j) = Ibis::norm2(w_);
        Ibis::scale(w_, v_, 1.0 / H0_(j + 1, j));
        Ibis::deep_copy_vector(krylov_vectors_.column(j + 1), v_);

        // progressively rotate the Hessenberg into the QR factorisation using
        // plane rotations, on the cpu
        if (j != 0) {
            // apply previous rotations to the new column of the Hessenberg
            auto Q_sub = Q0_.sub_matrix(0, j + 1, 0, j + 1);
            auto h_col_j = H0_.sub_matrix(0, j, 0, j).column(j);
            Ibis::gemv(Q_sub, h_col_j, h_rotated_);
            Ibis::deep_copy_vector(h_col_j, h_rotated_);
        }

        // build the rotation matrix for this step
        Omega_.set_to_identity();
        Ibis::real denom = 
            Ibis::sqrt(H0_(j, j) * H0_(j, j) + H0_(j + 1, j) * H0_(j + 1, j));
        Ibis::real si = H0_(j + 1, j) / denom;
        Ibis::real ci = H0_(j ,j)  / denom;
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
        Ibis::gemm(Omega, Q, Q_new);

        Ibis::deep_copy_vector(g, g_new);
        Q.deep_copy(Q_new);
        H.deep_copy(H_new);
    }

    return GmresResult(true, max_iters_, -1.0, -1.0);
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
    //
}
