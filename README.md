# Intro

Source code for the paper: Terrain-Adaptive Locomotion Skills using Deep Reinforcement Learning
https://www.cs.ubc.ca/~van/papers/2016-TOG-deepRL/index.html

# Setup

This section covers some of the steps to setup and compile the code. The software depends on many libraries that need to be carefully prepared and placed for the building and linking to work properly.

## Linux 

### Dependencies

 1. **libtorch** (PyTorch C++ distribution) or **ONNX Runtime** for policy inference backend.
 2. Boost  
 3. OpenCV  
 4. BulletPhysics
 5. CUDA (optional, if building with CUDA-enabled dependencies)
 6. Json_cpp (https://github.com/open-source-parsers/jsoncpp)  
 7. Eigen (http://eigen.tuxfamily.org/index.php?title=Main_Page)  
 8. bits  
	sudo apt-get install gcc-4.9-multilib g++-4.9-multilib 

Backend notes:
- `premake4 --backend=libtorch` (default)
- `premake4 --backend=onnxruntime`
- `premake4 --backend=both`
- Optional integration flags: `--with-pybind` and `--with-ipc`

### Linux Build Instructions

 1. Download the most recent compressed external file from the newest release.
 1. Extract it and move into the DeepTerrainRL directory.
 1. Place backend SDKs under `external/ml_backends/` (Linux) and/or `../library/ml_backends/` (Windows):
    - `libtorch/` for libtorch include/lib files
    - `onnxruntime/` for ONNX Runtime include/lib files
    - optional: `pybind11/` and `ipc/`
 1. Run `premake4 --backend=libtorch gmake` (or `--backend=onnxruntime` / `--backend=both`).
 1. Build with `make config=debug64`.
 1. Everything should build fine.

### Windows

This setup has been tested on Windows 10/11 with modern Visual Studio toolchains (v140+).

  1. Download the library.zip file that contains almost all of the relevant pre compiled external libraries and source code.
  2. Unpack this library in the same directory the project is located in. For example, TerrainRL/../.
  3. You might need to install opengl/glu/GL headers. We have been using freeglut for this project. glew might already be included in library.zip.
  4. You will need to copy some dll files from dynamic_lib.zip to the directory the project is compiled to. For example, optimizer/x64/Debug/. These files are needed by the framework during runtime.
  5. Might need to create a folder in TerrainRL called "output", This is where temprary and current policies will be dumped.


## Runing The System

After the system has been build there are two executable files that server different purposes. The **TerrainRL** program is for visually simulating the a controller and **TerrainRL_Optimize** is for optimizing the parameters of some controller.

Examples:  
	To simulate a controller/character  
	./TerrainRL -arg_file= args/sim_dog_args.txt  
	To simulate a controller/character with a specific policy  
	./TerrainRL_Optimizer -arg_file= args/dog_slopes_mixed_args.txt
	To Train a controller  
	./TerrainRL_Optimizer -arg_file= args/opt_args_train_mace.txt  


## Key Bindings

Most of these are togglesg

 - c fixed camera mode
 - y draw COM path and contact locations
 - q draw "filmstrip" like rendering
 - f draw torques
 - h draw Actor value functions and feature visualization
 - shift + '>' step one frame
 - p toggle draw value function
 - ',' and '.' change render speed, decrease and increase.
 - "spacebar" to pause simulation
 - r restart the scenario
 - l reload the simulation (reparses the arg file)
 - g draw state features
 - x spawn projectile
 - z spawn big projectile
 
 - click on character and drag to apply force
