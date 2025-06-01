#include <psGDSReader.hpp>

#include <lsCompareSparseField.hpp>
#include <lsToSurfaceMesh.hpp>
#include <lsExpand.hpp>
#include <lsReduce.hpp>
#include <lsVTKWriter.hpp>
#include <vcKDTree.hpp>



namespace ps = viennaps;
namespace ls = viennals;
namespace vc = viennacore;

using NumericType = double;


class OptimizationVelocityField : public ls::VelocityField<NumericType> {
private:
  ls::SmartPointer<ls::Mesh<NumericType>> differenceMesh;
  const vc::KDTree<NumericType, vc::Vec3Dd> &kdTree;
  const std::vector<NumericType> *signedDifferencesData = nullptr;

public:
  OptimizationVelocityField(ls::SmartPointer<ls::Mesh<NumericType>> diffMesh,
                            const vc::KDTree<NumericType, vc::Vec3Dd> &tree)
      : differenceMesh(diffMesh), kdTree(tree) {
    signedDifferencesData = differenceMesh->getPointData().getScalarData("Signed differences");
  }

  NumericType
  getScalarVelocity(const std::array<NumericType, 3> &coordinate,
                    int /*material*/,
                    const std::array<NumericType, 3> & /*normalVector*/,
                    unsigned long /*pointId*/) override {

    auto nearestPointInfo = kdTree.findNearest(coordinate);

    if (nearestPointInfo) {
      return (*signedDifferencesData)[nearestPointInfo->first];
    }
    return 0.0;
  }

  std::array<NumericType, 3>
  getVectorVelocity(const std::array<NumericType, 3> & /*coordinate*/,
                    int /*material*/,
                    const std::array<NumericType, 3> & /*normalVector*/,
                    unsigned long /*pointId*/) override {
    return {0.0, 0.0, 0.0}; // Using scalar velocities only
  }
};

int main(int argc, char **argv) {

  constexpr int D = 2;

  ps::Logger::setLogLevel(ps::LogLevel::DEBUG);

  // --- GDS Geometry Setup ---
  constexpr NumericType gridDelta = 0.005;
  constexpr NumericType exposureDelta = 0.005;
  double forwardSigma = 5.;
  NumericType backsSigma = 50.;

  ls::BoundaryConditionEnum boundaryConds[D] = {
      ls::BoundaryConditionEnum::REFLECTIVE_BOUNDARY,
      ls::BoundaryConditionEnum::REFLECTIVE_BOUNDARY};

  auto maskGeometry = ps::SmartPointer<ps::GDSGeometry<NumericType, D>>::New(gridDelta, boundaryConds);
  maskGeometry->addBlur({forwardSigma, backsSigma}, // Gaussian sigmas
                {0.8, 0.2},                         // Weights
                0.5,                                // Threshold
                exposureDelta);                     // Exposure grid delta
  std::string gdsFileName = "myTest.gds";
  if (argc > 1) {
    gdsFileName = argv[1];
  }
  ps::GDSReader<NumericType, D>(maskGeometry, gdsFileName).apply();

  // --- Level Set Initialization ---
  using lsDomainType = ls::SmartPointer<ls::Domain<NumericType, D>>;

  auto targetShapeLS = maskGeometry->layerToLevelSet(0, false); // target is the GDS shape
  ls::Reduce<NumericType, D>(targetShapeLS, 1).apply();
  
  auto maskLS = lsDomainType::New(targetShapeLS); // mask will get modified
  auto blurredLS = lsDomainType::New(targetShapeLS); // blurredLS will be used for velocity calculation
  auto pointdata = maskGeometry->applyBlur(maskLS);
  blurredLS->insertPoints(pointdata);
  blurredLS->finalize(2);
  ls::Expand<NumericType, D>(blurredLS, 50).apply();

  // --- VTK Initial Output ---
  auto vtkMesh = ls::SmartPointer<ls::Mesh<NumericType>>::New();
  ls::ToSurfaceMesh<NumericType, D>(maskLS, vtkMesh).apply();
  ls::VTKWriter<NumericType>(vtkMesh, "maskLS_initial.vtp").apply();
  ls::ToSurfaceMesh<NumericType, D>(blurredLS, vtkMesh).apply();
  ls::VTKWriter<NumericType>(vtkMesh, "blurredLS_initial.vtp").apply();
  ls::ToSurfaceMesh<NumericType, D>(targetShapeLS, vtkMesh).apply();
  ls::VTKWriter<NumericType>(vtkMesh, "targetShape.vtp").apply();

  // --- Compare Initial Shapes ---
  ls::CompareSparseField<NumericType, D> compare(blurredLS, targetShapeLS);
  compare.setFillIteratedWithDistances(true);
  auto differencesMesh = ls::SmartPointer<ls::Mesh<NumericType>>::New();
  compare.setOutputMesh(differencesMesh);
  compare.apply();

  NumericType prevRMSE = compare.getRMSE();
  NumericType currentRMSE = prevRMSE;
  std::cout << "Initial RMSE: " << prevRMSE << std::endl;

  // --- Iteration Parameters ---
  NumericType timeStepRatio = 0.4999;
  const NumericType minTimeStepRatio = 1e-9;
  const NumericType maxTimeStepRatio = 0.4999;
  const NumericType decreaseFactor = 0.5;
  const NumericType increaseFactor = 1.2;

  // --- Termination Criteria ---
  int max_iterations = 100;
  int stagnation_threshold = 10; // Number of iterations without 
                                 // improvement to trigger termination
  int stagnation_count = 0;
  NumericType RMSEIncreaseAllowance = 1.1; // Allow 10% increase in RMSE to escape local minima

  // --- Main Iteration Loop ---
  int iter = 0; 
  while (iter < max_iterations) 
  {
    std::cout << "--- Iteration: " << iter << ", TimeStepRatio: " << timeStepRatio
              << ", PrevRMSE: " << prevRMSE << " ---" << std::endl;

    // Save previous state 
    auto preAdvectMaskLS = lsDomainType::New(maskLS);
    auto preAdvectDifferencesMesh =
        ls::SmartPointer<ls::Mesh<NumericType>>::New(*differencesMesh);
    vc::KDTree<NumericType, vc::Vec3Dd> kdTree;
    kdTree.setPoints(differencesMesh->getNodes());
    kdTree.build();


    do 
    {
      maskLS = lsDomainType::New(preAdvectMaskLS);
      differencesMesh = ls::SmartPointer<ls::Mesh<NumericType>>::New(*preAdvectDifferencesMesh);
      // Advect
      {
        auto velocities = ls::SmartPointer<OptimizationVelocityField>::New(differencesMesh, kdTree);
        ls::Advect<NumericType, D> advectionKernel;
        advectionKernel.insertNextLevelSet(maskLS);
        advectionKernel.setVelocityField(velocities);
        advectionKernel.setSingleStep(true);
        advectionKernel.setTimeStepRatio(timeStepRatio);
        advectionKernel.apply();
      }

      // Update the blurred level set
      auto pointdata = maskGeometry->applyBlur(maskLS);
      blurredLS->insertPoints(pointdata);
      blurredLS->finalize(2);
      ls::Expand<NumericType, D>(blurredLS, 50).apply();

      // Compare
      {
        ls::CompareSparseField<NumericType, D> compareSparseField(
            blurredLS, targetShapeLS);
        compareSparseField.setFillIteratedWithDistances(true);
        compareSparseField.setOutputMesh(differencesMesh);
        compareSparseField.apply();
        currentRMSE = compareSparseField.getRMSE();
      }

      // Check RMSE and adjust time step ratio
      if (currentRMSE > (prevRMSE * RMSEIncreaseAllowance)) {
        std::cout << "Current RMSE is greater than previous RMSE: "
                  << currentRMSE << " > " << prevRMSE << std::endl;
        timeStepRatio =
            std::max(minTimeStepRatio, timeStepRatio * decreaseFactor);
        std::cout << "Decreasing time step ratio to: " << timeStepRatio
                  << std::endl;
      }

      if (timeStepRatio <= minTimeStepRatio) {
        std::cout << "Time step ratio has reached minimum limit: "
                  << timeStepRatio << std::endl;
        break;
      }

    } while (currentRMSE > (prevRMSE * RMSEIncreaseAllowance));

    // Check for stagnation
    NumericType improvement = prevRMSE - currentRMSE;
    if (improvement <= std::numeric_limits<NumericType>::epsilon()) {
      std::cout << "No improvement in RMSE: " << improvement
                << std::endl;
      stagnation_count++;
    } else {
      stagnation_count = 0;
    }

    if (stagnation_count >= stagnation_threshold) {
      std::cout << "Termination: Stagnation threshold (" << stagnation_threshold
                << " iterations) reached." << std::endl;
      break;
    }

    if (timeStepRatio <= minTimeStepRatio) {
      break;
    }

    // Update time step ratio and save state
    timeStepRatio = std::min(maxTimeStepRatio, timeStepRatio * increaseFactor);
    prevRMSE = currentRMSE;
    ls::ToSurfaceMesh<NumericType, D>(maskLS, vtkMesh).apply();
    ls::VTKWriter<NumericType>(vtkMesh,"maskLayer_" + std::to_string(iter) + ".vtp").apply();
    ls::ToSurfaceMesh<NumericType, D>(blurredLS, vtkMesh).apply();
    ls::VTKWriter<NumericType>(vtkMesh, "blurredLayer_" + std::to_string(iter) + ".vtp").apply();
    iter++;
  
  }

  if (iter >= max_iterations) {
          std::cout << "Termination: Maximum iterations reached." << std::endl;
  }
  std::cout << "Final RMSE: " << prevRMSE << std::endl;

  // --- VTK Final Output ---
  ls::ToSurfaceMesh<NumericType, D>(maskLS, vtkMesh).apply();
  ls::VTKWriter<NumericType>(vtkMesh, "maskLS_final.vtp").apply();
  ls::ToSurfaceMesh<NumericType, D>(blurredLS, vtkMesh).apply();
  ls::VTKWriter<NumericType>(vtkMesh, "blurredLS_final.vtp").apply();

  return 0;

}