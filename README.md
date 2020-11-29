# Guided Visibility Sampling++
This repository contains the reference implementation of our GVS++ visibility algorithm. The main parts of the algorithm are implemented in the shaders [raytrace.rgen](shaders/rt/raytrace.rgen) (random sampling), [raytrace_abs.rgen](shaders/rt/raytrace_abs.rgen) (adaptive border sampling), and [reverse_sampling_new.rgen](shaders/rt/reverse_sampling_new.rgen) (reverse sampling).

## Requirements
The application was developed using the following libraries (note that the code may also compile wither newer library versions):
* Vulkan SDK 1.2.135.0
* GLFW 3.3.2
* GLM 0.9.9.7
* tinyobjloader 1.0.7

The library paths are set in [CMakeLists.txt](CMakeLists.txt#L14).

## Usage
### Algorithm Parameters
The [settings file](settings/s0.txt) (`--- SETTINGS ---` section in the file) contains various parameters that control the behavior of the algorithm.
Various parameters that control the number of samples that are used can be set. These parameters can be used to fine-tune the algorithm for specific scenes, if necessary. We found that a standard set of parameters performs well for all tested scenes and view cell placements.
* `RANDOM_RAYS_PER_ITERATION`, the number of random rays/samples used during one iteration of random sampling, e.g. 10000000.
* `ABS_DELTA`, the distance of the enlarged polygon to the original triangle, e.g. 0.001.
* `ABS_NUM_SAMPLES_PER_EDGE`, the number of samples along each edge of the enlarged polygon, e.g. 40.
* `REVERSE_SAMPLING_NUM_SAMPLES_ALONG_EDGE`, the number of samples along each edge of the view cell for the reverse sampling, e.g. 40.
* `REVERSE_SAMPLING_HALTON_NUM_HALTON_SAMPLES`, the number of samples on the view cell for the reverse sampling, e.g. 40.

The termination of the PVS search can be controlled via two parameters: GVS++ terminates if less than `NEW_TRIANGLE_TERMINATION_THRESHOLD` new triangles have been found during the last `NEW_TRIANGLE_TERMINATION_THRESHOLD_COUNT` iterations.

Other parameters that can be set:
* `USE_3D_VIEW_CELL`, whether 3D or 2D view cells are used, e.g. true.
* `ERROR_VISUALIZATION`, whether the pixel error is calculated and triangles not part of the PVS are rendered and shaded red, e.g. false.

### Scene Specification
The scene and the file in which the resulting PVS is stored is also specified in the [settings file](settings/s0.txt) (`--- SCENE ---` section in the file).

Example:
```
--- SCENE ---
CALCPVS pvs.txt
CITY
SPECIFY_VIEW_CELL_CENTER false
```
* `CALCPVS` specifies that the PVS is calculated and stored in `pvs.txt`. Alternative are `CALCPVS_NOSTORE`, where the PVS is only calculated, and `LOADPVS`, where the PVS is loaded from the specified file.
* `CITY` is the name of the scene for which GVS++ is executed.
* `SPECIFY_VIEW_CELL_CENTER` specifies whether the position of the view cell (see below) defines the center or bottom left corner of the view cell.

Scenes and view cells are specified in the [scenes file](scenes.txt). Example:
```
CITY models/city.obj
30 3 150
40 50 150
0 0
-135 3 40
40 50 165
0 90
```
* `CITY` is the name of the scene followed by the location of the obj file.
* The view cell specification contains the position (`30 3 150`), size (`40 50 150`) and rotation around the x- and y-axis (`0 0`). Multiple view cells can be specified as in the example above. In this example, 3D view cells are used, therefore `USE_3D_VIEW_CELL = true` has to be set.

### Key bindings
* `C` switch to the next view cell.
* `F` cycle through the corners of the current view cell.
* `G` toggle rendering of a mesh that represents the view cell.
* `V` toggle shading.
* `Q` print camera position.
* `Esc` release the mouse.
* `1`/`2` decrease/increase camera speed.
* `W`, `A`, `S`, `D`, `Space`, `Shift` move camera.
