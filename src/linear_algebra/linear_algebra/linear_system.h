#ifndef LINEAR_SYSTEM_H
#define LINEAR_SYSTEM_H

#include <finite_volume/finite_volume.h>
#include <gas/gas_model.h>
#include <gas/transport_properties.h>
#include <util/field.h>
#include <util/numeric_types.h>
#include "finite_volume/conserved_quantities.h"

class SystemLinearisation {
public:
    SystemLinearisation(){};

    virtual ~SystemLinearisation() {}

    virtual void matrix_vector_product(FiniteVolume<Ibis::dual>& fv,
                                       ConservedQuantities<Ibis::dual>& cq,
                                       const GridBlock<Ibis::dual>& grid,
                                       IdealGas<Ibis::dual>& gas_model,
                                       TransportProperties<Ibis::dual>& trans_prop,
                                       Field<Ibis::real>& vec) = 0;

    virtual void eval_rhs(FiniteVolume<Ibis::dual>& fv, FlowStates<Ibis::dual>&,
                          const GridBlock<Ibis::dual>& grid,
                          IdealGas<Ibis::dual>& gas_model,
                          TransportProperties<Ibis::dual>& trans_prop,
                          ConservedQuantities<Ibis::dual>& residuals,
                          Field<Ibis::real>& vec) = 0;
};

#endif
