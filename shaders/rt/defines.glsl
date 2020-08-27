/*
    REVERSE_SAMPLING_METHOD 0       Reverse sampling method based on the old CPU implementation
    REVERSE_SAMPLING_METHOD 1       Reverse sampling based on the GVS paper
    REVERSE_SAMPLING_METHOD 2       New reverse sampling
*/
#define REVERSE_SAMPLING_METHOD 2

/*
    SET_TYPE 0      Set implemented as an array that's as big as there are triangles
                    in the scene (4 byte per triangle).
    SET_TYPE 1      Hash set. Needs less space than SET_TYPE 1. Recommended for scenes
                    with >= 10,000,000 triangles. If the hash set runs out of space,
                    it is resized.
*/
#define SET_TYPE 1

//#define USE_3D_VIEW_CELL
