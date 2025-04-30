#include <optix_device.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <raygBoundary.hpp>
#include <raygLaunchParams.hpp>
#include <raygPerRayData.hpp>
#include <raygRNG.hpp>
#include <raygReflection.hpp>
#include <raygSBTRecords.hpp>
#include <raygSource.hpp>

#include <models/psPlasmaEtchingParameters.hpp>

#include <vcContext.hpp>
#include <vcVectorType.hpp>

using namespace viennaray::gpu;

extern "C" __constant__ LaunchParams launchParams;
enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

/* --------------- ION --------------- */

extern "C" __global__ void __closesthit__FCIon() {
  const HitSBTData *sbtData = (const HitSBTData *)optixGetSbtDataPointer();
  PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

  if (sbtData->isBoundary) {
    if (params.periodicBoundary) {
      applyPeriodicBoundary(prd, sbtData);
    } else {
      reflectFromBoundary(prd);
    }
  } else {
    auto primID = optixGetPrimitiveIndex();
    int material = launchParams.materialIds[primID];
    viennaps::FluorocarbonParameters<float> *params =
        reinterpret_cast<viennaps::FluorocarbonParameters<float> *>(
            launchParams.customData);

    auto geomNormal = computeNormal(sbtData, primID);
    auto cosTheta = -viennacore::DotProduct(prd->dir, geomNormal);
    float angle = acosf(max(min(cosTheta, 1.f), 0.f));

    const float sqrtE = sqrtf(prd->energy);
    const float f_e_sp = (1 + B_sp * (1 - cosTheta * cosTheta)) * cosTheta;
    const float Y_sp = Ae_sp * max(sqrtE - sqrt_Eth_sp, 0.f) * f_e_sp;
    const float Y_ie = Ae_ie * max(sqrtE - sqrt_Eth_ie, 0.f) * cosTheta;
    const float Y_p = Ap_ie * max(sqrtE - sqrt_Eth_p, 0.f) * cosTheta;

    // sputtering yield Y_sp ionSputteringFlux
    atomicAdd(&params.resultBuffer[getIdx(0, &params)], Y_sp);

    // ion enhanced etching yield Y_ie ionEnhancedFlux
    atomicAdd(&params.resultBuffer[getIdx(1, &params)], Y_ie);

    // ion enhanced O sputtering yield Y_O ionPolymerFlux
    atomicAdd(&params.resultBuffer[getIdx(2, &params)], Y_p);

    // ---------- REFLECTION ------------ //
    // Small incident angles are reflected with the energy fraction centered at
    // 0
    float Eref_peak = 0.f;
    float A = 1. / (1. + params->Ions.n_l *
                             (M_PI_2f / params->Ions.inflectAngle - 1.));
    if (angle >= params->Ions.inflectAngle) {
      Eref_peak = 1 - (1 - A) * (M_PI_2f - angle) /
                          (M_PI_2f - params->Ions.inflectAngle);
    } else {
      Eref_peak = A * pow(angle / params->Ions.inflectAngle, params->Ions.n_l);
    }

    // Gaussian distribution around the Eref_peak scaled by the particle energy
    float newEnergy;
    do {
      newEnergy = getNormalDistRand(&prd->RNGstate) * prd->energy * 0.1f +
                  Eref_peak * prd->energy;
    } while (newEnergy > prd->energy || newEnergy <= 0.f);

    // Set the flag to stop tracing if the energy is below the threshold
    float minEnergy = min(params->Substrate.Eth_ie, params->Substrate.Eth_sp);
    if (newEnergy > minEnergy) {
      prd->energy = newEnergy;
      conedCosineReflection(prd, geomNormal,
                            M_PI_2f - min(angle, params->Ions.minAngle));
    } else {
      prd->energy = -1.f;
    }
  }
}

extern "C" __global__ void __miss__FCIon() {
  getPRD<PerRayData>()->rayWeight = 0.f;
}

extern "C" __global__ void __raygen__FCIon() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  const int linearLaunchIndex =
      idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;

  // per-ray data
  PerRayData prd;
  // each ray has its own RNG state
  initializeRNGState(&prd, linearLaunchIndex, launchParams.seed);

  // initialize ray position and direction
  initializeRayPosition(&prd, &launchParams);
  initializeRayDirection(&prd, launchParams.cosineExponent);

  viennaps::PlasmaEtchingParameters<float> *params =
      reinterpret_cast<viennaps::PlasmaEtchingParameters<float> *>(
          launchParams.customData);
  float minEnergy = min(params->Substrate.Eth_ie, params->Substrate.Eth_sp);
  do {
    prd.energy = getNormalDistRand(&prd.RNGstate) * params->Ions.sigmaEnergy +
                 params->Ions.meanEnergy;
  } while (prd.energy < minEnergy);

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer((void *)&prd, u0, u1);

  while (continueRay(launchParams, prd)) {
    optixTrace(launchParams.traversable, // traversable GAS
               make_float3(prd.pos[0], prd.pos[1], prd.pos[2]), // origin
               make_float3(prd.dir[0], prd.dir[1], prd.dir[2]), // direction
               1e-4f,                                           // tmin
               1e20f,                                           // tmax
               0.0f,                                            // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,              // SBT offset
               RAY_TYPE_COUNT,                // SBT stride
               SURFACE_RAY_TYPE,              // missSBTIndex
               u0, u1);
  }
}

/* --------------- ETCHANT --------------- */

extern "C" __global__ void __closesthit__FCEtchant() {
  const HitSBTData *sbtData = (const HitSBTData *)optixGetSbtDataPointer();
  PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

  if (sbtData->isBoundary) {
    if (launchParams.periodicBoundary) {
      applyPeriodicBoundary(prd, sbtData);
    } else {
      reflectFromBoundary(prd);
    }
  } else {
    atomicAdd(&launchParams.resultBuffer[getIdx(0, &launchParams)],
              prd->rayWeight);

    // ------------- REFLECTION --------------- //
    const unsigned int primID = optixGetPrimitiveIndex();
    float *data = (float *)sbtData->cellData;
    const float &phi_e = data[primID];
    int material = launchParams.materialIds[primID];

    /// TODO:
    // Check material ID
    float gamma_e = launchParams.materialSticking[material];
    const float Seff = gamma_e * max(1.f - phi_e, 0.f);
    prd->rayWeight -= prd->rayWeight * Seff;
    diffuseReflection(prd);
  }
}

extern "C" __global__ void __miss__FCEtchant() {
  getPRD<PerRayData>()->rayWeight = 0.f;
}

extern "C" __global__ void __raygen__FCEtchant() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  const int linearLaunchIndex =
      idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;

  // per-ray data
  PerRayData prd;
  // each ray has its own RNG state
  initializeRNGState(&prd, linearLaunchIndex, launchParams.seed);

  // initialize ray position and direction
  initializeRayPosition(&prd, &launchParams);
  initializeRayDirection(&prd, launchParams.cosineExponent);

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer((void *)&prd, u0, u1);

  while (continueRay(launchParams, prd)) {
    optixTrace(launchParams.traversable, // traversable GAS
               make_float3(prd.pos[0], prd.pos[1], prd.pos[2]), // origin
               make_float3(prd.dir[0], prd.dir[1], prd.dir[2]), // direction
               1e-4f,                                           // tmin
               1e20f,                                           // tmax
               0.0f,                                            // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,              // SBT offset
               RAY_TYPE_COUNT,                // SBT stride
               SURFACE_RAY_TYPE,              // missSBTIndex
               u0, u1);
  }
}

/* ------------- POLYMER --------------- */

extern "C" __global__ void __closesthit__FCPolymer() {
  const HitSBTData *sbtData = (const HitSBTData *)optixGetSbtDataPointer();
  PerRayData *prd = (PerRayData *)getPRD<PerRayData>();

  if (sbtData->isBoundary) {
    if (launchParams.periodicBoundary) {
      applyPeriodicBoundary(prd, sbtData);
    } else {
      reflectFromBoundary(prd);
    }
  } else {
    atomicAdd(&launchParams.resultBuffer[getIdx(0, &launchParams)],
              prd->rayWeight);

    // ------------- REFLECTION --------------- //
    const unsigned int primID = optixGetPrimitiveIndex();
    float *data = (float *)sbtData->cellData;
    const float &phi_e = data[primID];
    const float &phi_p = data[primID + launchParams.numElements];
    int material = launchParams.materialIds[primID];

    float gamma_pe = launchParams.materialSticking[material];
    const float Seff = gamma_pe * max(1.f - phi_e - phi_p, 0.f);
    prd->rayWeight -= prd->rayWeight * Seff;
    diffuseReflection(prd);
  }
}

extern "C" __global__ void __miss__FCPolymer() {
  getPRD<PerRayData>()->rayWeight = 0.f;
}

extern "C" __global__ void __raygen__FCPolymer() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  const int linearLaunchIndex =
      idx.x + idx.y * dims.x + idx.z * dims.x * dims.y;

  // per-ray data
  PerRayData prd;
  // each ray has its own RNG state
  initializeRNGState(&prd, linearLaunchIndex, launchParams.seed);

  // initialize ray position and direction
  initializeRayPosition(&prd, &launchParams);
  initializeRayDirection(&prd, launchParams.cosineExponent);

  // the values we store the PRD pointer in:
  uint32_t u0, u1;
  packPointer((void *)&prd, u0, u1);

  while (continueRay(launchParams, prd)) {
    optixTrace(launchParams.traversable, // traversable GAS
               make_float3(prd.pos[0], prd.pos[1], prd.pos[2]), // origin
               make_float3(prd.dir[0], prd.dir[1], prd.dir[2]), // direction
               1e-4f,                                           // tmin
               1e20f,                                           // tmax
               0.0f,                                            // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,              // SBT offset
               RAY_TYPE_COUNT,                // SBT stride
               SURFACE_RAY_TYPE,              // missSBTIndex
               u0, u1);
  }
}
