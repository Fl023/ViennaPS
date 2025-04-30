#pragma once

#include <cmath>

#include <models/psFluorocarbonEtching.hpp>
#include <psMaterials.hpp>
#include <psUnits.hpp>

#include <psgProcessModel.hpp>

#include <vcLogger.hpp>
#include <vcVectorType.hpp>

namespace viennaps::gpu {

using namespace viennacore;

template <typename NumericType, int D>
class FluorocarbonEtching : public ProcessModel<NumericType, D> {
public:
  FluorocarbonEtching() { initialize(); }
  FluorocarbonEtching(const double ionFlux, const double etchantFlux,
                      const double polyFlux, const NumericType meanEnergy,
                      const NumericType sigmaEnergy,
                      const NumericType exponent = 100.,
                      const NumericType deltaP = 0.,
                      const NumericType etchStopDepth =
                          std::numeric_limits<NumericType>::lowest()) {
    params_.ionFlux = ionFlux;
    params_.etchantFlux = etchantFlux;
    params_.polyFlux = polyFlux;
    params_.Ions.meanEnergy = meanEnergy;
    params_.Ions.sigmaEnergy = sigmaEnergy;
    params_.Ions.exponent = exponent;
    params_.delta_p = deltaP;
    params_.etchStopDepth = etchStopDepth;
    initialize();
  }
  FluorocarbonEtching(const FluorocarbonParameters<NumericType> &parameters)
      : params_(parameters) {
    initialize();
  }

  FluorocarbonParameters<NumericType> &getParameters() { return params_; }
  void setParameters(const FluorocarbonParameters<NumericType> &parameters) {
    params_ = parameters;
    initialize();
  }

private:
  FluorocarbonParameters<NumericType> params_;

  void initialize() {
    // check if units have been set
    if (units::Length::getInstance().getUnit() == units::Length::UNDEFINED ||
        units::Time::getInstance().getUnit() == units::Time::UNDEFINED) {
      Logger::getInstance().addError("Units have not been set.").print();
    }

    // particles
    auto ion = std::make_unique<impl::FluorocarbonIon<NumericType, D>>(params_);
    auto etchant =
        std::make_unique<impl::FluorocarbonEtchant<NumericType, D>>(params_);
    auto poly =
        std::make_unique<impl::FluorocarbonPolymer<NumericType, D>>(params_);

    // surface model
    auto surfModel =
        SmartPointer<impl::FluorocarbonSurfaceModel<NumericType, D>>::New(
            params_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("FluorocarbonEtching");
    this->particles.clear();
    this->insertNextParticleType(ion);
    this->insertNextParticleType(etchant);
    this->insertNextParticleType(poly);
  }
};

} // namespace viennaps::gpu
