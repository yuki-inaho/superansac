# Synthetic-only C++ tests for the generic 3D-line extension.

add_executable(test_line3d_core line3d/test_line3d_core.cpp)
target_link_libraries(test_line3d_core PRIVATE SupeRANSAC)

add_test(NAME line3d_core COMMAND test_line3d_core)
