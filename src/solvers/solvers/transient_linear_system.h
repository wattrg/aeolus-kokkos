#ifndef TRANSIENT_LINEAR_SYSTEM_H
#define TRANSIENT_LINEAR_SYSTEM_H

#include <linear_algebra/linear_system.h>

// This provides an interface for a pseudo-transient linear system
// This is essentially the interface for a linear system, but with
// an extra method to set the pseudo time step size.
class PseudoTransientLinearSystem : public LinearSystem {
public:
    PseudoTransientLinearSystem() {}

    virtual ~PseudoTransientLinearSystem() {}

    virtual void set_pseudo_time_step(Ibis::real dt_star) = 0;
};

#endif
