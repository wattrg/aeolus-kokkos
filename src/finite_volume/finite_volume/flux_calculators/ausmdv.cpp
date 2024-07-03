#include <finite_volume/conserved_quantities.h>
#include <finite_volume/flux_calc.h>
#include <gas/flow_state.h>
#include <util/numeric_types.h>

#include <Kokkos_Core.hpp>

template <typename T>
void Ausmdv<T>::compute_flux(const FlowStates<T>& left, const FlowStates<T>& right,
                             ConservedQuantities<T>& flux, IdealGas<T>& gm,
                             bool three_d) {
    Kokkos::parallel_for(
        "ausmdv", flux.size(), KOKKOS_LAMBDA(const int i) {
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

            // This is the main part of the flux calculator.
            // Weighting parameters (eqn 32) for velocity splitting.
            T alphaL = 2.0 * pLrL / (pLrL + pRrR);
            T alphaR = 2.0 * pRrR / (pLrL + pRrR);

            // Common sound speed (eqn 33) and Mach doubles.
            T am = Ibis::max(aL, aR);
            T ML = uL / am;
            T MR = uR / am;

            // Left state:
            // pressure splitting (eqn 34)
            // and velocity splitting (eqn 30)
            T pLplus, uLplus;
            T duL = 0.5 * (uL + Ibis::abs(uL));
            if (Ibis::abs(ML) <= 1.0) {
                pLplus = pL * (ML + 1.0) * (ML + 1.0) * (2.0 - ML) * 0.25;
                uLplus = alphaL * ((uL + am) * (uL + am) / (4.0 * am) - duL) + duL;
            } else {
                pLplus = pL * duL / uL;
                uLplus = duL;
            }

            // Right state:
            // pressure splitting (eqn 34)
            // and velocity splitting (eqn 31)
            T pRminus, uRminus;
            T duR = 0.5 * (uR - Ibis::abs(uR));
            if (Ibis::abs(MR) <= 1.0) {
                pRminus = pR * (MR - 1.0) * (MR - 1.0) * (2.0 + MR) * 0.25;
                uRminus = alphaR * (-(uR - am) * (uR - am) / (4.0 * am) - duR) + duR;
            } else {
                pRminus = pR * duR / uR;
                uRminus = duR;
            }

            // Mass Flux (eqn 29)
            // The mass flux is relative to the moving interface.
            T ru_half = uLplus * rL + uRminus * rR;

            // Pressure flux (eqn 34)
            T p_half = pLplus + pRminus;

            // Momentum flux: normal direction
            // Compute blending parameter s (eqn 37),
            // the momentum flux for AUSMV (eqn 21) and AUSMD (eqn 21)
            // and blend (eqn 36).
            T dp = pL - pR;
            const T K_SWITCH = 10.0;
            dp = K_SWITCH * Ibis::abs(dp) / Ibis::min(pL, pR);
            T s = 0.5 * Ibis::min(1.0, dp);
            T ru2_AUSMV = uLplus * rL * uL + uRminus * rR * uR;
            T ru2_AUSMD = 0.5 * (ru_half * (uL + uR) - Ibis::abs(ru_half) * (uR - uL));
            T ru2_half = (0.5 + s) * ru2_AUSMV + (0.5 - s) * ru2_AUSMD;

            // Assemble components of the flux vector.
            flux.mass(i) = ru_half;
            if (ru_half >= 0.0) {
                // Wind is blowing from the left.
                flux.momentum_x(i) = (ru2_half + p_half);
                flux.momentum_y(i) = (ru_half * vL);
                if (three_d) {
                    flux.momentum_z(i) = (ru_half * wL);
                }
                flux.energy(i) = ru_half * HL;
            } else {
                // Wind is blowing from the right.
                flux.momentum_x(i) = (ru2_half + p_half);
                flux.momentum_y(i) = (ru_half * vR);
                if (three_d) {
                    flux.momentum_z(i) = (ru_half * wR);
                }
                flux.energy(i) = ru_half * HR;
            }

            // Apply entropy fix (section 3.5 in Wada and Liou's paper)
            const T C_EFIX = 0.125;
            bool caseA = ((uL - aL) < 0.0) && ((uR - aR) > 0.0);
            bool caseB = ((uL + aL) < 0.0) && ((uR + aR) > 0.0);
            T d_ua = 0.0;
            if (caseA && !caseB) {
                d_ua = C_EFIX * ((uR - aR) - (uL - aL));
            }
            if (caseB && !caseA) {
                d_ua = C_EFIX * ((uR + aR) - (uL + aL));
            }
            if (d_ua != 0.0) {
                flux.mass(i) -= d_ua * (rR - rL);
                flux.momentum_x(i) -= d_ua * (rR * uR - rL * uL);
                flux.momentum_y(i) -= d_ua * (rR * vR - rL * vL);
                if (three_d) {
                    flux.momentum_z(i) -= d_ua * (rR * wR - rL * wL);
                }
                flux.energy(i) -= d_ua * (rR * HR - rL * HL);
            }
        });
}
template class Ausmdv<Ibis::real>;
template class Ausmdv<Ibis::dual>;
