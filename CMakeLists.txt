cmake_minimum_required(VERSION 3.10)
project(GSplatViewer)

set(CMAKE_CXX_STANDARD 17)

# --- Dependencies ---

# 1. OpenCL (Required by the Futhark backend)
find_package(OpenCL REQUIRED)

# 2. GLFW (Windowing)
find_package(glfw3 REQUIRED)

# 3. OpenGL
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

# 4. Dear ImGui
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/submodules/imgui")
file(GLOB IMGUI_SOURCES
    "${IMGUI_DIR}/imgui.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
)
include_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends)

# --- Compilation ---

# We compile the Futhark generated C code alongside our main application
add_executable(GSplatViewer
    main.cpp
    gsplat.c
    ${IMGUI_SOURCES}
)

# Link Libraries
target_link_libraries(GSplatViewer
    PRIVATE
    OpenCL::OpenCL
    glfw
    OpenGL::GL
    ${CMAKE_DL_LIBS}
)

# Definitions for ImGui
target_compile_definitions(GSplatViewer PRIVATE IMGUI_IMPL_OPENGL_LOADER_GL3W)
