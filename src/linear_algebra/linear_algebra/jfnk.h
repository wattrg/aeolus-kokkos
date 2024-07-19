#ifndef JFNK_H
#define JFNK_H

#include <finite_volume/conserved_quantities.h>
#include <gas/flow_state.h>
#include <gas/transport_properties.h>
#include <io/io.h>
#include <solver.h>
#include <solvers/cfl.h>
#include <util/numeric_types.h>

#include <memory>

class Jfnk {
public:
    void solve(EquationSystem& system);
    // something
};

#endif
