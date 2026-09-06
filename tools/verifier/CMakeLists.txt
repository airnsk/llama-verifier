cmake_minimum_required(VERSION 3.14)

set(TARGET llama-verifier)

add_executable(${TARGET} verifier.cpp)

target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE llama cpp-httplib nlohmann ${CMAKE_THREAD_LIBS_INIT})

if (MSVC)
    target_compile_options(${TARGET} PRIVATE /wd4996)
endif()

set_target_properties(${TARGET} PROPERTIES FOLDER Tools)

install(TARGETS ${TARGET} RUNTIME)
