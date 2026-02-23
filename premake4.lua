--
-- premake4 file to build TerrainRL
-- Copyright (c) 2009-2015 Glen Berseth
-- See license.txt for complete license.
--

local action = _ACTION or ""
local todir = "./" .. action

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

local linuxLibraryLoc = "./external/"
local windowsLibraryLoc = "../library/"

newoption {
	trigger = "backend",
	value = "NAME",
	description = "Select ML backend: libtorch, onnxruntime, or both",
	allowed = {
		{ "libtorch", "Link against libtorch" },
		{ "onnxruntime", "Link against ONNX Runtime" },
		{ "both", "Link against libtorch and ONNX Runtime" }
	}
}

newoption {
	trigger = "with-pybind",
	description = "Enable optional pybind integration"
}

newoption {
	trigger = "with-ipc",
	description = "Enable optional IPC integration"
}

local backend = _OPTIONS["backend"] or "libtorch"
local use_libtorch = (backend == "libtorch" or backend == "both")
local use_onnxruntime = (backend == "onnxruntime" or backend == "both")
local use_pybind = _OPTIONS["with-pybind"] ~= nil
local use_ipc = _OPTIONS["with-ipc"] ~= nil


solution "TerrainRL"
	configurations { 
		"Debug",
		"Release"
	}
	
	platforms {
		"x32", 
		"x64"
	}
	-- location (todir)

	-- extra warnings, no exceptions or rtti
	flags { 
		"ExtraWarnings",
--		"FloatFast",
--		"NoExceptions",
--		"NoRTTI",
		"Symbols"
	}
	-- defines { "ENABLE_GUI", "ENABLE_GLFW" }

	-- debug configs
	configuration "Debug*"
		defines { "DEBUG" }
		flags {
			"Symbols",
			Optimize = Off
		}
 
 	-- release configs
	configuration "Release*"
		defines { "NDEBUG" }
		flags { "Optimize" }

	-- windows specific
	configuration "windows"
		defines { "WIN32", "_WINDOWS" }
		libdirs { "lib" }
		targetdir ( "./" )

	configuration { "linux" }
		linkoptions { 
			-- "-stdlib=libc++" ,
			"-Wl,-rpath," .. path.getabsolute("lib")
		}
		targetdir ( "./" )
		
	configuration { "macosx" }
        buildoptions { "-stdlib=libc++ -std=c++11 -ggdb" }
		linkoptions { 
			"-stdlib=libc++" ,
			"-Wl,-rpath," .. path.getabsolute("lib")
		}
		links {
	        "OpenGL.framework",
        }
        targetdir ( "./" )
      
	if os.get() == "macosx" then
		premake.gcc.cc = "clang"
		premake.gcc.cxx = "clang++"
		-- buildoptions("-std=c++0x -ggdb -stdlib=libc++" )
	end


project "TerrainRL"
	language "C++"
	kind "WindowedApp"

	files { 
		-- Source files for this project
		"learning/*.h",
		"learning/*.cpp",
		"render/*.h",
		"render/*.cpp",
		"scenarios/*.h",
		"scenarios/*.cpp",
		"sim/*.h",
		"sim/*.cpp",
		"util/*.h",
		"util/*.cpp",
		"anim/*.h",
		"anim/*.cpp",
		"external/LodePNG/*.cpp",
		"Main.cpp"
	}
	excludes 
	{
		"learning/DMACETrainer - Copy.cpp"
	}	
	includedirs { 
		"./",
	}

	

	defines {
		"_DEBUG",
		"_CRT_SECURE_NO_WARNINGS",
		"_SCL_SECURE_NO_WARNINGS",
		"CPU_ONLY",
		"GOOGLE_GLOG_DLL_DECL=",
		"ENABLE_TRAINING",
		"ENABLE_DEBUG_PRINT",
		"ENABLE_DEBUG_VISUALIZATION",
	}

	targetdir "./"
	buildoptions("-std=c++0x -ggdb" )	

	if use_libtorch then
		defines { "ENABLE_BACKEND_LIBTORCH" }
	end
	if use_onnxruntime then
		defines { "ENABLE_BACKEND_ONNXRUNTIME" }
	end
	if use_pybind then
		defines { "ENABLE_OPTIONAL_PYBIND" }
	end
	if use_ipc then
		defines { "ENABLE_OPTIONAL_IPC" }
	end


	-- linux library cflags and libs
	configuration { "linux", "gmake" }
		buildoptions { 
			"`pkg-config --cflags gl`",
			"`pkg-config --cflags glu`" 
		}
		linkoptions { 
			"-Wl,-rpath," .. path.getabsolute("lib") ,
			"`pkg-config --libs gl`",
			"`pkg-config --libs glu`" 
		}
		libdirs { 
			-- "lib",
			linuxLibraryLoc .. "Bullet/bin",
			linuxLibraryLoc .. "jsoncpp/build/debug/src/lib_json",
			linuxLibraryLoc .. "ml_backends/libtorch/lib",
			linuxLibraryLoc .. "ml_backends/onnxruntime/lib",
					}
		
		includedirs { 
			linuxLibraryLoc .. "Bullet/src",
			linuxLibraryLoc,
			linuxLibraryLoc .. "jsoncpp/include",
			linuxLibraryLoc .. "ml_backends/libtorch/include",
			linuxLibraryLoc .. "ml_backends/libtorch/include/torch/csrc/api/include",
			linuxLibraryLoc .. "ml_backends/onnxruntime/include",
									"C:/Program Files (x86)/boost/boost_1_58_0/",
															"/usr/local/cuda/include/",
			linuxLibraryLoc .. "OpenCV/include",
					}
		defines {
			"_LINUX_",
		}
			-- debug configs
		configuration "Debug*"
			links {
				"X11",
				"dl",
				"pthread",
				-- Just a few dependancies....
				"BulletDynamics_gmake_x64_debug",
				"BulletCollision_gmake_x64_debug",
				"LinearMath_gmake_x64_debug",
				"jsoncpp",
				"boost_system",
				"glog",
				"glut",
				"GLEW",
			}
	 
	 	-- release configs
		configuration "Release*"
			defines { "NDEBUG" }
			links {
				"X11",
				"dl",
				"pthread",
				-- Just a few dependancies....
				"BulletDynamics_gmake_x64_release",
				"BulletCollision_gmake_x64_release",
				"LinearMath_gmake_x64_release",
				"jsoncpp",
				"boost_system",
				"glog",
				"glut",
				"GLEW",
			}

	-- windows library cflags and libs
	configuration { "windows" }
		-- libdirs { "lib" }
		libdirs {
			windowsLibraryLoc .. "ml_backends/libtorch/lib",
			windowsLibraryLoc .. "ml_backends/onnxruntime/lib",
			windowsLibraryLoc .. "ml_backends/ipc/lib",
		}
		includedirs { 
			windowsLibraryLoc .. "Bullet/include",
			windowsLibraryLoc,
			windowsLibraryLoc .. "Json_cpp",
			windowsLibraryLoc .. "ml_backends/libtorch/include",
			windowsLibraryLoc .. "ml_backends/libtorch/include/torch/csrc/api/include",
			windowsLibraryLoc .. "ml_backends/onnxruntime/include",
			windowsLibraryLoc .. "ml_backends/pybind11/include",
			windowsLibraryLoc .. "ml_backends/ipc/include",
						"C:/Program Files (x86)/boost/boost_1_58_0/",
															"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v7.5/include/",
			windowsLibraryLoc .. "OpenCV/include",
					}	
		links { 
			"opengl32",
			"glu32",
			-- Just a few dependancies....
			"BulletDynamics_Debug",
			"BulletCollision_Debug",
			"LinearMath_Debug",
			"jsoncpp_Debug",
			"opencv_core300d",
			"opencv_calib3d300d",
			"opencv_flann300d",
			"opencv_highgui300d",
			"opencv_imgproc300d",
			"opencv_imgcodecs300d",
			"opencv_ml300d",
			"opencv_objdetect300d",
			"opencv_photo300d",
			"opencv_features2d300d",
			"opencv_stitching300d",
			"opencv_video300d",
			"opencv_videostab300d",
			"opencv_hal300d",
			"libjpegd",
			"libjasperd",
			"libpngd",
			"IlmImfd",
			"libtiffd",
			"libwebpd",
			"cudart",
			"cuda",
			"nppi",
			"cufft",
			"cublas",
			"curand",
			"gflagsd",
			"libglogd",
			"Shlwapi",
			"zlibd",
			"libopenblas",
			"torch",
			"onnxruntime"
		}

	-- mac includes and libs
	configuration { "macosx" }
		kind "ConsoleApp" -- xcode4 failes to run the project if using WindowedApp
		-- includedirs { "/Library/Frameworks/SDL.framework/Headers" }
		buildoptions { "-Wunused-value -Wshadow -Wreorder -Wsign-compare -Wall" }
		linkoptions { 
			"-Wl,-rpath," .. path.getabsolute("lib") ,
		}
		links { 
			"OpenGL.framework", 
			"Cocoa.framework",
			"dl",
			"pthread"
		}

include "optimizer/"
