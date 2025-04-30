from argparse import ArgumentParser

# parse config file name and simulation dimension
parser = ArgumentParser(
    prog="trenchDeposition",
    description="Run a deposition process on a trench geometry.",
)
parser.add_argument("-D", "-DIM", dest="dim", type=int, default=2)
parser.add_argument("filename")
args = parser.parse_args()

# switch between 2D and 3D mode
if args.dim == 2:
    print("Running 2D simulation.")
    import viennaps2d as vps

    useGPU = False  # GPU support is not available for 2D
else:
    print("Running 3D simulation.")
    import viennaps3d as vps

    # Check if GPU support is available
    useGPU = True
    try:
        import viennaps3d.viennaps3d.gpu as gpu

        context = gpu.Context()
        context.create(modulePath=vps.ptxPath)
        print("Using GPU.")

    except ImportError:
        useGPU = False

params = vps.ReadConfigFile(args.filename)

geometry = vps.Domain(
    gridDelta=params["gridDelta"],
    xExtent=params["xExtent"],
    yExtent=params["yExtent"],
)
vps.MakeTrench(
    domain=geometry,
    trenchWidth=params["trenchWidth"],
    trenchDepth=params["trenchHeight"],
    trenchTaperAngle=params["taperAngle"],
).apply()

geometry.duplicateTopLevelSet(vps.Material.SiO2)

model = (
    gpu.SingleParticleProcess(
        stickingProbability=params["stickingProbability"],
        sourceExponent=params["sourcePower"],
    )
    if useGPU
    else vps.SingleParticleProcess(
        stickingProbability=params["stickingProbability"],
        sourceExponent=params["sourcePower"],
    )
)

geometry.saveHullMesh("initial")

process = gpu.Process(context) if useGPU else vps.Process()
process.setDomain(geometry)
process.setProcessModel(model)
process.setProcessDuration(params["processTime"])
process.apply()

geometry.saveHullMesh("final")
