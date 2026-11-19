"""Supplementary Code S3. Reproduces the operating-unit conversions in Sections 3.3, 4.2, 8 and 9
of the manuscript (Eqs. 1, 3-6 and 8) and the values plotted in Figs. 3 and 4.
Standard library only.  Usage:  python Supplementary_Code_S3_system_units.py"""
import math
R = 8.314462618      # J mol-1 K-1
M_H2 = 2.016e-3      # kg mol-1
LHV = 120.0          # MJ kg-1 H2 (value used in the manuscript)

def pressure_factor_from_lnP(mae_ln):            # characteristic factor exp(MAE)
    return math.exp(mae_ln)
def pressure_factor_from_dH(dH_kJ, T=298.0):     # fixed entropy, Eq. 5 (second part)
    return math.exp(dH_kJ * 1e3 / (R * T))
def T1(dH_kJ, dS_J):                             # Eq. 4
    return dH_kJ * 1e3 / dS_J
def dT1_from_dH(dH_kJ, dS_J):                    # Eq. 5, fixed entropy
    return dH_kJ * 1e3 / dS_J
def dT1_from_dS(dS_err, T1_K, dS_J):             # Eq. 5, fixed enthalpy
    return T1_K * dS_err / dS_J
def dT1_from_dlnP(dlnP, T1_K, dS_J):             # Eq. 6
    return R * T1_K / dS_J * dlnP
def heat_duty(dH_kJ):                            # Eq. 8, MJ per kg H2
    return dH_kJ / M_H2 / 1e3

print('--- Fig. 3: generalisation-gap ratio G = MAE_grouped / MAE_random (Eq. 1)')
G = {
 'MH-PCTpro isotherm holdout [17] (0.55/0.17 wt%)': 0.55/0.17,
 'MH-PCTpro composition holdout [17] (0.64/0.17 wt%)': 0.64/0.17,
 'Re-analysis dH, chemical system (7.56/6.99)': 7.563/6.985,
 'Re-analysis dH, alloy family (10.17/6.99)': 10.170/6.985,
 'Re-analysis lnP, chemical system (2.19/2.00)': 2.189/1.999,
 'Re-analysis lnP, alloy family (3.24/2.00)': 3.236/1.999,
 'Re-analysis T1, chemical system (42.1/37.9)': 42.05/37.95,
 'Re-analysis T1, alloy family (58.9/37.9)': 58.89/37.95,
}
for k, v in G.items(): print(f'  {k:52s} G = {v:.2f}')
print('  MatFold exemplar [54]: MAE +17% to +133% -> G = 1.17-2.33; single-element (H) holdout ~10')
print('  Newly added compounds [23]: one to two orders of magnitude -> G = 10-100')

print('--- Fig. 4: characteristic pressure factors')
for lab, f in [('HydPARK+ ln P MAE 1.4 [12]', pressure_factor_from_lnP(1.4)),
               ('Extended set, ln P MAE 1.07 [27]', pressure_factor_from_lnP(1.07)),
               ('Descriptor, 0.40 log10 MPa [38]', 10**0.40),
               ('GPR enthalpy miss 3.3 kJ/mol [28]', pressure_factor_from_dH(3.3)),
               ('HydPARK+ dH MAE 5.5 kJ/mol [12]', pressure_factor_from_dH(5.5)),
               ('Same, at 353 K', pressure_factor_from_dH(5.5, 353)),
               ('DFT offset 11 kJ/mol [13]', pressure_factor_from_dH(11)),
               ('DFT offset 15 kJ/mol [13]', pressure_factor_from_dH(15))]:
    print(f'  {lab:36s} x{f:.1f}')
print(f'  Share of a 30:1 window used by a factor of 4: {math.log(4)/math.log(30):.0%}')

print('--- Section 8.1: equilibrium temperature (Eqs. 4-6)')
dS = 110.0
print(f'  T1 at dH=35, dS=110: {T1(35, dS):.0f} K')
print(f'  dT1 for dH error 5.5 kJ/mol at fixed dS: {dT1_from_dH(5.5, dS):.0f} K')
t1 = T1(35, dS); k = R * t1 / dS
print(f'  RT1/dS at T1={t1:.0f} K: {k:.1f} K per unit ln P')
print(f'  ln P error at T1 equivalent to 5.5 kJ/mol: {5.5e3/(R*t1):.2f}  -> dT1 = {dT1_from_dlnP(5.5e3/(R*t1), t1, dS):.0f} K')
print(f'  ln P error allowed for +/-10 K: {10/k:.2f}  (pressure factor x{math.exp(10/k):.2f})')
print(f'  dT1 for a 10 J/mol/K entropy error at T1={t1:.0f} K: {dT1_from_dS(10, t1, dS):.0f} K')
print(f'  T1 at dH=35 with dS=98.9 and 142.4: {T1(35, 98.9):.0f} K and {T1(35, 142.4):.0f} K')
print(f'  Eq. 6 check on re-analysis: lnP ratio {3.236/1.999:.2f} vs T1 ratio {58.89/37.95:.2f}')

print('--- Section 8.3: reaction heat (Eq. 8)')
for h in (34.4, 45.4):
    q = heat_duty(h); print(f'  dH {h} kJ/mol -> {q:.1f} MJ/kg H2 ({q/LHV:.0%} of LHV)')
q = heat_duty(5.5); print(f'  5.5 kJ/mol -> {q:.1f} MJ/kg H2 ({q/LHV:.1%} of LHV)')
print('--- Section 9.3: interval half-width in T1')
print(f'  20 kJ/mol at dS=110 -> +/-{dT1_from_dH(20, dS):.0f} K')
