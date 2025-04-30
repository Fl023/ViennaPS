#pragma once

#include <psgProcessModel.hpp>

#include <models/psTEOSPECVD.hpp>

namespace viennaps::gpu {

using namespace viennacore;

template <class NumericType, int D>
class TEOSPECVD : public ProcessModel<NumericType, D> {
public:
  TEOSPECVD(const NumericType radicalSticking, const NumericType radicalRate,
            const NumericType ionRate, const NumericType ionExponent,
            const NumericType ionThetaRMin = 60.,
            const NumericType ionThetaRMax = 90.,
            const NumericType ionMinAngle = 0.,
            const NumericType radicalOrder = 1.,
            const NumericType ionOrder = 1.) {
    // particles
    viennaray::gpu::Particle<NumericType> radical;
    radical.name = "Neutral";
    radical.sticking = radicalSticking;
    radical.dataLabels.push_back("radicalFlux");
    radical.materialSticking[static_cast<int>(Material::Undefined)] =
        1.; // this will initialize all to default sticking

    viennaray::gpu::Particle<NumericType> ion;
    ion.name = "Ion";
    ion.dataLabels.push_back("ionFlux");
    ion.cosineExponent = ionExponent;

    impl::IonParams params;
    params.thetaRMin = constants::degToRad(ionThetaRMin);
    params.thetaRMax = constants::degToRad(ionThetaRMax);
    params.minAngle = ionMinAngle;
    params.B_sp = -1.f;
    params.meanEnergy = -1.f;
    this->processData.allocUploadSingle(params);

    this->insertNextParticleType(ion);
    this->insertNextParticleType(radical);

    // surface model
    auto surfModel =
        SmartPointer<::viennaps::impl::PECVDSurfaceModel<NumericType>>::New(
            radicalRate, radicalOrder, ionRate, ionOrder);

    // velocity field
    auto velField =
        SmartPointer<::viennaps::DefaultVelocityField<NumericType, D>>::New(2);

    this->setUseMaterialIds(true);
    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("TEOSPECVD");
    this->setPipelineFileName("MultiParticlePipeline");
  }
};

} // namespace viennaps::gpu