add_executable(ocr_test ocr.cpp)

target_link_libraries(ocr_test aica)

target_include_directories(ocr_test PUBLIC 
                    ${AICA_ROOT}/deps/mnist/include)