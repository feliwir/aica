

add_library(aica STATIC activations.hpp 
                        ann.cpp ann.hpp)

target_include_directories(aica PUBLIC
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    ${AICA_ROOT}/deps/xtl/include 
                    ${AICA_ROOT}/deps/xtensor/include
                    ${AICA_ROOT}/deps/xtensor-blas/include
                    ${AICA_ROOT}/deps/xtensor-blas/include/xtensor-blas/flens)