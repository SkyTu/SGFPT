cmake_minimum_required(VERSION 3.17)
project(sytorch LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maes -march=native -Wno-ignored-attributes -Wno-deprecated-declarations")
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
add_subdirectory(ext/cryptoTools)
add_subdirectory(ext/bitpack)
add_subdirectory(ext/llama)
# Retained GPU kernels and SIGMA use LLAMA, not the independent SCI/SecFloat backend.
add_library(sytorch STATIC
    src/sytorch/random.cpp
    src/sytorch/backend/cleartext.cpp
    src/sytorch/backend/baseline_cleartext.cpp
    src/sytorch/backend/float.cpp)
target_include_directories(sytorch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sytorch PUBLIC Eigen3::Eigen Threads::Threads OpenMP::OpenMP_CXX CUDA::cudart LLAMA cryptoTools)
