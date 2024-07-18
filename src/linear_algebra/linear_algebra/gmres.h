#ifndef GMRES_H
#define GMRES_H

#include <linear_algebra/linear_solver.h>

struct GmresResult {
    bool succes;
    bool n_iters;
    bool tol;
};

class Gmres {
    GmresResult solve(LinearSystem& system);
};

class FGmres {
    
};

#endif
