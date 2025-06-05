cmake_minimum_required(VERSION 3.28)
project(lbp_pedro)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Encontra o OpenCV
find_package(OpenCV REQUIRED)

# Adiciona os headers da biblioteca (não precisa compilar, pois é header-only)
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

# Garante que todos os exemplos sejam construídos
file(GLOB EXAMPLES "examples/*.cpp")

foreach(example_src ${EXAMPLES})
    get_filename_component(example_name ${example_src} NAME_WE)

    add_executable(${example_name} ${example_src})

    # Linka com OpenCV
    target_link_libraries(${example_name} ${OpenCV_LIBS})

    # Opcional: define os exemplos como testes
    # enable_testing()
    # add_test(NAME run_${example_name} COMMAND ${example_name})
endforeach()
