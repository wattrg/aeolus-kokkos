#include <doctest/doctest.h>
#include <linear_algebra/gmres.h>
#include <linear_algebra/vector.h>

Gmres::Gmres(const std::shared_ptr<LinearSystem> system, const size_t max_iters,
             Ibis::real tol) {
    tol_ = tol;
    max_iters_ = max_iters;
    num_vars_ = system->num_vars();

    H0_ = Ibis::Matrix<Ibis::real>("Gmres::H0", max_iters_ + 1, max_iters_);
    H1_ = Ibis::Matrix<Ibis::real>("Gmres::H1", max_iters_ + 1, max_iters_);
    Gamma_ = Ibis::Matrix<Ibis::real>("Gmres::Gamma", max_iters_ + 1, max_iters_ + 1);
    krylov_vectors_ =
        Ibis::Matrix<Ibis::real>("Gmres::krylov_vectors", num_vars_, max_iters_);
    r0_ = Ibis::Vector<Ibis::real>("Gmres::r0", num_vars_);
    w_ = Ibis::Vector<Ibis::real>("Gmres::w", num_vars_);
}

Gmres::Gmres(const std::shared_ptr<LinearSystem> system, json config)
    : Gmres(system, config.at("max_iters"), config.at("tolerance")) {}

GmresResult Gmres::solve(std::shared_ptr<LinearSystem> system,
                         Ibis::Vector<Ibis::real>& x0) {
    compute_r0_(system, x0);
    Ibis::real beta = Ibis::norm2(r0_);
    Ibis::scale(r0_, v_, 1.0 / beta);

    for (size_t j = 0; j < max_iters_; j++) {
        system->matrix_vector_product(v_, w_);
        for (size_t i = 0; i < j; i++) {
            H0_(i, j) = Ibis::dot(w_, v_);  // need to think about where H0_ lives
            Ibis::add_scaled_vector(w_, krylov_vectors_.column(i), -H0_(i, j));
        }
        H0_(j + 1, j) = Ibis::norm2(w_);
        Ibis::scale(w_, v_, 1.0 / H0_(j + 1, j));
        Ibis::deep_copy_vector(krylov_vectors_.column(j + 1), v_);
    }
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
