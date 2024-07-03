#include <finite_volume/conserved_quantities.h>
#include <finite_volume/flux_calc.h>
#include <gas/flow_state.h>
#include <util/numeric_types.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

template <typename T>
void Ldfss2<T>::compute_flux(const FlowStates<T>& left, const FlowStates<T>& right,
                             ConservedQuantities<T>& flux, IdealGas<T>& gm,
                             bool three_d) {
    double delta = delta_;
    Kokkos::parallel_for(
        "ldfss", flux.size(), KOKKOS_LAMBDA(const int i) {
            // unpack left flow state
            T rL = left.gas.rho(i);
            T pL = left.gas.pressure(i);
            T pLrL = pL / rL;
            T uL = left.vel.x(i);
            T vL = left.vel.y(i);
            T wL = left.vel.z(i);
            T eL = gm.internal_energy(left.gas, i);
            T aL = gm.speed_of_sound(left.gas, i);
            T keL = 0.5 * (uL * uL + vL * vL + wL * wL);
            T HL = eL + pLrL + keL;

            // unpack right flow state
            T rR = right.gas.rho(i);
            T pR = right.gas.pressure(i);
            T pRrR = pR / rR;
            T uR = right.vel.x(i);
            T vR = right.vel.y(i);
            T wR = right.vel.z(i);
            T eR = gm.internal_energy(right.gas, i);
            T aR = gm.speed_of_sound(right.gas, i);
            T keR = 0.5 * (uR * uR + vR * vR + wR * wR);
            T HR = eR + pRrR + keR;

            // common sound speed, and mach numbers
            T am = 0.5 * (aL + aR);
            T ML = uL / am;
            T MR = uR / am;

            // split mach numbers
            T MpL = 0.25 * (ML + 1.0) * (ML + 1.0);
            T MmR = -0.25 * (MR - 1.0) * (MR - 1.0);

            // parameters to provide correct sonic-point transition behaviour
            T alphaL = 0.5 * (1.0 + Ibis::copysign(T(1.0), ML));
            T alphaR = 0.5 * (1.0 - Ibis::copysign(T(1.0), MR));

            // equation 17
            T betaL = -Ibis::max(0.0, 1.0 - Ibis::floor(Ibis::abs(ML)));
            T betaR = -Ibis::max(0.0, 1.0 - Ibis::floor(Ibis::abs(MR)));

            // subsonic pressure splitting (eqn 12)
            T PL = 0.25 * (ML + 1.0) * (ML + 1.0) * (2.0 - ML);
            T PR = 0.25 * (MR - 1.0) * (MR - 1.0) * (2.0 + MR);

            // eqn 11
            T DL = alphaL * (1.0 + betaL) - betaL * PL;
            T DR = alphaR * (1.0 + betaR) - betaR * PR;

            T Mhalf = 0.25 * betaL * betaR *
                      Ibis::pow((Ibis::sqrt(0.5 * (ML * ML + MR * MR)) - 1.0), T(2.0));

            T MhalfL =
                Mhalf *
                (1.0 - ((pL - pR) / (pL + pR) + delta * (Ibis::abs(pL - pR) / pL)));
            T MhalfR =
                Mhalf *
                (1.0 + ((pL - pR) / (pL + pR) - delta * (Ibis::abs(pL - pR) / pR)));

            // C parameter for LDFSS (2) (eqn 13 & eqn 14 & eqn 26 & eqn 27)
            T CL = alphaL * (1.0 + betaL) * ML - betaL * MpL - MhalfL;
            T CR = alphaR * (1.0 + betaR) * MR - betaR * MmR + MhalfR;

            T ru_half = am * rL * CL + am * rR * CR;
            T ru2_half = am * rL * CL * uL + am * rR * CR * uR;
            T p_half = DL * pL + DR * pR;
            flux.mass(i) = ru_half;
            flux.momentum_x(i) = ru2_half + p_half;
            flux.momentum_y(i) = am * rL * CL * vL + am * rR * CR * vR;
            if (three_d) {
                flux.momentum_z(i) = am * rL * CL * wL + am * rR * CR * wR;
            }
            flux.energy(i) = am * rL * CL * HL + am * rR * CR * HR;
        });
}
template class Ldfss2<Ibis::real>;
template class Ldfss2<Ibis::dual>;
