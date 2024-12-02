cmake_minimum_required(VERSION 3.10)
project(UpMEMProject LANGUAGES C)

# Add the `dev` binary with DPU-specific compiler
set(CMAKE_C_COMPILER "dpu-upmem-dpurte-clang")
add_executable(dev lstm.c)

# Add the `host` binary with the default system compiler
# Temporarily override the compiler to ensure it uses the system compiler
set(CMAKE_C_COMPILER "/usr/bin/gcc") # Update this path if gcc is located elsewhere
add_executable(host host.c)

# Add the include directory for DPU (applies to host)
target_include_directories(host PRIVATE /opt/upmem/include/dpu)

# Add the library directory and link the `dpu` library (applies to host)
target_link_directories(host PRIVATE /opt/upmem/lib)
target_link_libraries(host PRIVATE dpu)

# Set C99 standard for both binaries
target_compile_features(dev PRIVATE c_std_99)
target_compile_features(host PRIVATE c_std_99)

