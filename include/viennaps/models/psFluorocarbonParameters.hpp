#pragma once

#include <limits>

// Parameters from:
// A. LaMagna and G. Garozzo "Factors affecting profile evolution in plasma
// etching of SiO2: Modeling and experimental verification" Journal of the
// Electrochemical Society 150(10) 2003 pp. 1896-1902

template <typename NumericType> struct FluorocarbonParameters {
  // fluxes in (1e15 /cm² /s)
  NumericType ionFlux = 56.;
  NumericType etchantFlux = 500.;
  NumericType polyFlux = 100.;

  NumericType delta_p = 1.;
  NumericType etchStopDepth = std::numeric_limits<NumericType>::lowest();

  NumericType temperature = 300.; // K
  NumericType k_ie = 2.;
  NumericType k_ev = 2.;

  NumericType beta_pe = 0.6;
  NumericType beta_p = 0.26;
  NumericType beta_e = 0.9;

  // Mask
  struct MaskType {
    NumericType rho = 500.; // 1e22 atoms/cm³
    NumericType beta_p = 0.01;
    NumericType beta_e = 0.1;

    NumericType A_sp = 0.0139;
    NumericType B_sp = 9.3;
    NumericType Eth_sp = 20.; // eV
  } Mask;

  // SiO2
  struct SiO2Type {
    // density
    NumericType rho = 2.2; // 1e22 atoms/cm³

    // sputtering coefficients
    NumericType Eth_sp = 18.; // eV
    NumericType Eth_ie = 4.;  // eV
    NumericType A_sp = 0.0139;
    NumericType B_sp = 9.3;
    NumericType A_ie = 0.0361;

    // chemical etching
    NumericType K = 0.002789491704544977;
    NumericType E_a = 0.168; // eV
  } SiO2;

  // Polymer
  struct PolymerType {
    NumericType rho = 2.; // 1e22 atoms/cm³

    // sputtering coefficients
    NumericType Eth_ie = 4.; // eV
    NumericType A_ie = 0.0361 * 4;
  } Polymer;

  // Si3N4
  struct Si3N4Type {
    // density
    NumericType rho = 2.3; // 1e22 atoms/cm³

    // sputtering coefficients
    NumericType Eth_sp = 18.; // eV
    NumericType Eth_ie = 4.;  // eV
    NumericType A_sp = 0.0139;
    NumericType B_sp = 9.3;
    NumericType A_ie = 0.0361;

    // chemical etching
    NumericType K = 0.002789491704544977;
    NumericType E_a = 0.168; // eV
  } Si3N4;

  // Si
  struct SiType {
    // density
    NumericType rho = 5.02; // 1e22 atoms/cm³

    // sputtering coefficients
    NumericType Eth_sp = 20.; // eV
    NumericType Eth_ie = 4.;  // eV
    NumericType A_sp = 0.0337;
    NumericType B_sp = 9.3;
    NumericType A_ie = 0.0361;

    // chemical etching
    NumericType K = 0.029997010728956663;
    NumericType E_a = 0.108; // eV
  } Si;

  struct IonType {
    NumericType meanEnergy = 100.; // eV
    NumericType sigmaEnergy = 10.; // eV
    NumericType exponent = 500.;

    NumericType inflectAngle = 1.55334303;
    NumericType n_l = 10.;
    NumericType minAngle = 1.3962634;
  } Ions;
};