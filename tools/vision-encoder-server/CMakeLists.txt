set(TARGET vision-encoder-server)

add_executable(${TARGET} main.cpp)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/vendor
    ${CMAKE_SOURCE_DIR}/tools/mtmd
)

target_link_libraries(${TARGET} PRIVATE
    mtmd
    common
    llama
    ggml
    cpp-httplib
    ${CMAKE_THREAD_LIBS_INIT}
)

target_compile_definitions(${TARGET} PRIVATE
    MTMD_INTERNAL_HEADER
)

target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Install
install(TARGETS ${TARGET} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
