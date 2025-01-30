from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(prog="holeEtching", description="Run a hole etching process.")
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

# switch between 2D and 3D mode
if args.dim == 2:
    print("Running 2D simulation.")
    import viennaps2d as vps
else:
    print("Running 3D simulation.")
    import viennaps3d as vps

params = vps.ReadConfigFile(args.filename)

# print error output surfaces during the process
vps.Logger.setLogLevel(vps.LogLevel.INFO)

# Map the shape string to the corresponding vps.HoleShape enum
shape_map = {
    "Full": vps.HoleShape.Full,
    "Half": vps.HoleShape.Half,
    "Quarter": vps.HoleShape.Quarter,
}

hole_shape_str = params.get("holeShape", "Full").strip()

# geometry setup, all units in um
geometry = vps.Domain()
vps.MakeHole(
    domain=geometry,
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
    holeRadius=params["holeRadius"],
    holeDepth=params["maskHeight"],
    taperingAngle=params["taperAngle"],
    holeShape=shape_map[hole_shape_str],
    periodicBoundary=False,
    makeMask=True,
    material=vps.Material.Si,
).apply()

# use pre-defined model SF6O2 etching model
model = vps.SF6O2Etching(
    ionFlux=params["ionFlux"],
    etchantFlux=params["etchantFlux"],
    oxygenFlux=params["oxygenFlux"],
    meanIonEnergy=params["meanEnergy"],
    sigmaIonEnergy=params["sigmaEnergy"],
    oxySputterYield=params["A_O"],
    etchStopDepth=params["etchStopDepth"],
)
parameters = model.getParameters()
parameters.Mask.rho = 2.6

# process setup
process = vps.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setMaxCoverageInitIterations(10)
process.setNumberOfRaysPerPoint(int(params["raysPerPoint"]))
process.setProcessDuration(params["processTime"]/10)  # seconds
process.setTimeStepRatio(0.2)
# print initial surface
geometry.saveSurfaceMesh(filename="initial.vtp", addMaterialIds=True)

for i in range (1,11):
    # run the process
    process.apply()
    # print intermediate surface
    geometry.saveSurfaceMesh(filename=f"intermediate_{i}.vtp", addMaterialIds=True)

# print final surface
geometry.saveSurfaceMesh(filename="final.vtp", addMaterialIds=True)
