

add_library(aica STATIC activations.hpp 
                        ann.cpp ann.hpp)

target_compile_definitions(aica PUBLIC XTENSOR_USE_XSIMD)

find_package(BLAS)

if(BLAS_FOUND)
    set(BLA_VENDOR OpenBLAS)
    set(BLA_STATIC true)
    add_definitions(-DHAVE_CBLAS=1)
    target_link_libraries(aica ${BLAS_LIBRARIES})
endif(BLAS_FOUND)

target_include_directories(aica PUBLIC
                    ${CMAKE_CURRENT_SOURCE_DIR}
                    ${AICA_ROOT}/deps/xtl/include
                    ${AICA_ROOT}/deps/xsimd/include 
                    ${AICA_ROOT}/deps/xtensor/include
                    ${AICA_ROOT}/deps/xtensor-blas/include
                    ${AICA_ROOT}/deps/xtensor-blas/include/xtensor-blas/flens)