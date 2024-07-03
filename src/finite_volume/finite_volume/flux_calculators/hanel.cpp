#include <finite_volume/conserved_quantities.h>
#include <finite_volume/flux_calc.h>
#include <gas/flow_state.h>

#include <Kokkos_Core.hpp>

template <typename T>
void Hanel<T>::compute_flux(const FlowStates<T>& left, const FlowStates<T>& right,
                            ConservedQuantities<T>& flux, IdealGas<T>& gm, bool three_d) {
    Kokkos::parallel_for(
        "Flux::hanel", flux.size(), KOKKOS_LAMBDA(const int i) {
            // unpack left gas state
            T rL = left.gas.rho(i);
            T pL = left.gas.pressure(i);
            T eL = gm.internal_energy(left.gas, i);
            T aL = gm.speed_of_sound(left.gas, i);
            T uL = left.vel.x(i);
            T vL = left.vel.y(i);
            T wL = left.vel.z(i);
            T keL = 0.5 * (uL * uL + vL * vL + wL * wL);
            T pLrL = pL / rL;
            T HL = eL + pLrL + keL;

            // unpack right gas state
            T rR = right.gas.rho(i);
            T pR = right.gas.pressure(i);
            T eR = gm.internal_energy(right.gas, i);
            T aR = gm.speed_of_sound(right.gas, i);
            T uR = right.vel.x(i);
            T vR = right.vel.y(i);
            T wR = right.vel.z(i);
            T keR = 0.5 * (uR * uR + vR * vR + wR * wR);
            T pRrR = pR / rR;
            T HR = eR + pRrR + keR;

            // pressure and velocity splitting (eqn. 7 and 9)
            T pLplus, uLplus;
            if (Ibis::abs(uL) <= aL) {
                uLplus = 1.0 / (4.0 * aL) * (uL + aL) * (uL + aL);
                pLplus = pL * uLplus * (1.0 / aL * (2.0 - uL / aL));
            } else {
                uLplus = 0.5 * (uL + Ibis::abs(uL));
                pLplus = pL * uLplus * (1.0 / uL);
            }

            T pRminus, uRminus;
            if (Ibis::abs(uR) <= aR) {
                uRminus = -1.0 / (4.0 * aR) * (uR - aR) * (uR - aR);
                pRminus = pR * uRminus * (1.0 / aR * (-2.0 - uR / aR));
            } else {
                uRminus = 0.5 * (uR - Ibis::abs(uR));
                pRminus = pR * uRminus * (1.0 / uR);
            }

            // the final fluxes
            T p_half = pLplus + pRminus;
            flux.mass(i) = uLplus * rL + uRminus * rR;
            flux.momentum_x(i) = uLplus * rL * uL + uRminus * rR * uR + p_half;
            flux.momentum_y(i) = uLplus * rL * vL + uRminus * rR * vR;
            if (three_d) {
                flux.momentum_z(i) = uLplus * rL * wL + uRminus * rR * wR;
            }
            flux.energy(i) = uLplus * rL * HL + uRminus * rR * HR;
        });
}
template class Hanel<Ibis::real>;
template class Hanel<Ibis::dual>;
