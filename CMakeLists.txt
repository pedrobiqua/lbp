cmake_minimum_required(VERSION 3.28)
project(lbp_pedro)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Encontra o OpenCV
find_package(OpenCV REQUIRED)
find_package(Armadillo REQUIRED)

# Adiciona os headers da biblioteca (não precisa compilar, pois é header-only)
include_directories(
    /usr/include/python3.12
    /usr/lib/python3/dist-packages/numpy/core/include
    include
    ${OpenCV_INCLUDE_DIRS}
    ${ARMADILLO_INCLUDE_DIRS} # Adicionado: Armadillo include directories
)

# Garante que todos os exemplos sejam construídos
file(GLOB EXAMPLES "examples/*.cpp")

foreach(example_src ${EXAMPLES})
    get_filename_component(example_name ${example_src} NAME_WE)

    add_executable(${example_name} ${example_src})

    # Linka com OpenCV e Armadillo
    target_link_libraries(${example_name}
        ${OpenCV_LIBS}
        ${ARMADILLO_LIBRARIES}
    )

    # Link para o matplotlibcpp
    link_libraries(python3.12)

    # Opcional: define os exemplos como testes
    # enable_testing()
    # add_test(NAME run_${example_name} COMMAND ${example_name})
endforeach()