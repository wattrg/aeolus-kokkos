#ifndef GMRES_H
#define GMRES_H

#include <linear_algebra/linear_solver.h>
#include <util/field.h>
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
    Gmres(const SystemLinearisation& system, const size_t max_iters, Ibis::real tol);

    Gmres(const SystemLinearisation& system, json config);

    GmresResult solve(SystemLinearisation& system, Field<Ibis::real>& solution);

private:
    size_t max_iters_;
    size_t tol_;
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
