#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Graphics & UI Headers
#include <GL/gl.h>
#define GLFW_INCLUDE_NONE
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

// Futhark Backend Header
#include "gsplat.h"

// --- Math Helpers ---
struct Vec3 {
  float x, y, z;
};
struct Quat {
  float w, x, y, z;
};

// Simple Camera Structure
struct Camera {
  Vec3 pos = {0.0f, 0.0f, -5.0f};
  Vec3 target = {0.0f, 0.0f, 0.0f};
  float yaw = -90.0f;
  float pitch = 0.0f;
  float fov = 45.0f;

  Quat getRotation() const {
    float cy = cos(yaw * 0.00872665f); // deg -> rad/2
    float sy = sin(yaw * 0.00872665f);
    float cp = cos(pitch * 0.00872665f);
    float sp = sin(pitch * 0.00872665f);
    return {cy * cp, cy * sp, sy * cp, -sy * sp};
  }
};

// --- Futhark Wrapper ---
class SplatRenderer {
  struct futhark_context_config *cfg;
  struct futhark_context *ctx;

  // GPU Arrays
  struct futhark_f32_1d *xyz_x, *xyz_y, *xyz_z;
  struct futhark_f32_1d *opas;
  struct futhark_f32_1d *s_x, *s_y, *s_z;
  struct futhark_f32_1d *rot_w, *rot_x, *rot_y, *rot_z;
  struct futhark_f32_1d *c_r, *c_g, *c_b;
  struct futhark_f32_2d *sh_r, *sh_g, *sh_b;

  int num_splats = 0;

public:
  SplatRenderer() {
    cfg = futhark_context_config_new();
    futhark_context_config_select_device_interactively(cfg);
    ctx = futhark_context_new(cfg);
  }

  ~SplatRenderer() {
    if (num_splats > 0)
      freeResources();
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
  }

  void freeResources() {
    futhark_free_f32_1d(ctx, xyz_x);
    futhark_free_f32_1d(ctx, xyz_y);
    futhark_free_f32_1d(ctx, xyz_z);
    futhark_free_f32_1d(ctx, opas);
    futhark_free_f32_1d(ctx, s_x);
    futhark_free_f32_1d(ctx, s_y);
    futhark_free_f32_1d(ctx, s_z);
    futhark_free_f32_1d(ctx, rot_w);
    futhark_free_f32_1d(ctx, rot_x);
    futhark_free_f32_1d(ctx, rot_y);
    futhark_free_f32_1d(ctx, rot_z);
    futhark_free_f32_1d(ctx, c_r);
    futhark_free_f32_1d(ctx, c_g);
    futhark_free_f32_1d(ctx, c_b);
    futhark_free_f32_2d(ctx, sh_r);
    futhark_free_f32_2d(ctx, sh_g);
    futhark_free_f32_2d(ctx, sh_b);
  }

  bool loadPly(const char *filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Failed to open PLY file: " << filename << std::endl;
      return false;
    }

    // --- 1. Parse Header ---
    std::string line;
    int vertexCount = 0;
    std::map<std::string, int> props; // map property name to byte offset index
    int propCounter = 0;
    bool inVertex = false;

    while (std::getline(file, line)) {
      if (line == "end_header")
        break;
      std::stringstream ss(line);
      std::string token;
      ss >> token;
      if (token == "element") {
        ss >> token;
        if (token == "vertex") {
          ss >> vertexCount;
          inVertex = true;
        } else {
          inVertex = false;
        }
      } else if (token == "property" && inVertex) {
        std::string type, name;
        ss >> type >> name;
        props[name] = propCounter++;
      }
    }

    std::cout << "Loading " << vertexCount << " splats..." << std::endl;

    // --- 2. Precompute Indices (The Fix) ---
    // We look up indices ONCE here, single-threaded.
    auto getIdx = [&](const std::string &name) -> int {
      auto it = props.find(name);
      return (it != props.end()) ? it->second : -1;
    };

    int idx_x = getIdx("x");
    int idx_y = getIdx("y");
    int idx_z = getIdx("z");
    int idx_op = getIdx("opacity");
    int idx_sx = getIdx("scale_0");
    int idx_sy = getIdx("scale_1");
    int idx_sz = getIdx("scale_2");
    int idx_rw = getIdx("rot_0");
    int idx_rx = getIdx("rot_1");
    int idx_ry = getIdx("rot_2");
    int idx_rz = getIdx("rot_3");
    int idx_cr = getIdx("f_dc_0");
    int idx_cg = getIdx("f_dc_1");
    int idx_cb = getIdx("f_dc_2");

    // Store SH indices in a flat array for fast access
    // f_rest_0 to f_rest_44
    std::vector<int> idx_rest(45);
    for (int i = 0; i < 45; ++i) {
      idx_rest[i] = getIdx("f_rest_" + std::to_string(i));
    }

    // Prepare Host Buffers
    std::vector<float> h_x(vertexCount), h_y(vertexCount), h_z(vertexCount);
    std::vector<float> h_op(vertexCount);
    std::vector<float> h_sx(vertexCount), h_sy(vertexCount), h_sz(vertexCount);
    std::vector<float> h_rw(vertexCount), h_rx(vertexCount), h_ry(vertexCount), h_rz(vertexCount);
    std::vector<float> h_cr(vertexCount), h_cg(vertexCount), h_cb(vertexCount);
    std::vector<float> h_shr(vertexCount * 15, 0.f), h_shg(vertexCount * 15, 0.f), h_shb(vertexCount * 15, 0.f);

    // --- 3. Read Binary Data ---
    size_t stride = propCounter * 4;
    std::vector<char> buffer(vertexCount * stride);
    file.read(buffer.data(), buffer.size());

// --- 4. Parallel Loop with Raw Access ---
#pragma omp parallel for
    for (int i = 0; i < vertexCount; ++i) {
      size_t base_offset = i * stride;

      // Helper to read float from base + property index
      auto readF = [&](int p_idx) -> float {
        if (p_idx == -1)
          return 0.0f;
        float val;
        // Direct memory copy, no map lookups, no strings
        std::memcpy(&val, &buffer[base_offset + p_idx * 4], 4);
        return val;
      };

      h_x[i] = readF(idx_x);
      h_y[i] = readF(idx_y);
      h_z[i] = readF(idx_z);
      h_op[i] = readF(idx_op);

      h_sx[i] = readF(idx_sx);
      h_sy[i] = readF(idx_sy);
      h_sz[i] = readF(idx_sz);

      h_rw[i] = readF(idx_rw);
      h_rx[i] = readF(idx_rx);
      h_ry[i] = readF(idx_ry);
      h_rz[i] = readF(idx_rz);

      h_cr[i] = readF(idx_cr);
      h_cg[i] = readF(idx_cg);
      h_cb[i] = readF(idx_cb);

      for (int j = 0; j < 15; ++j) {
        // Standard 3DGS PLY is interleaved: f_rest_0=R0, f_rest_1=G0, f_rest_2=B0, ...
        h_shr[i * 15 + j] = readF(idx_rest[j * 3 + 0]);
        h_shg[i * 15 + j] = readF(idx_rest[j * 3 + 1]);
        h_shb[i * 15 + j] = readF(idx_rest[j * 3 + 2]);
      }
    }

    // --- 5. Upload to GPU ---
    if (num_splats > 0)
      freeResources();
    num_splats = vertexCount;

    xyz_x = futhark_new_f32_1d(ctx, h_x.data(), vertexCount);
    xyz_y = futhark_new_f32_1d(ctx, h_y.data(), vertexCount);
    xyz_z = futhark_new_f32_1d(ctx, h_z.data(), vertexCount);
    opas = futhark_new_f32_1d(ctx, h_op.data(), vertexCount);
    s_x = futhark_new_f32_1d(ctx, h_sx.data(), vertexCount);
    s_y = futhark_new_f32_1d(ctx, h_sy.data(), vertexCount);
    s_z = futhark_new_f32_1d(ctx, h_sz.data(), vertexCount);
    rot_w = futhark_new_f32_1d(ctx, h_rw.data(), vertexCount);
    rot_x = futhark_new_f32_1d(ctx, h_rx.data(), vertexCount);
    rot_y = futhark_new_f32_1d(ctx, h_ry.data(), vertexCount);
    rot_z = futhark_new_f32_1d(ctx, h_rz.data(), vertexCount);
    c_r = futhark_new_f32_1d(ctx, h_cr.data(), vertexCount);
    c_g = futhark_new_f32_1d(ctx, h_cg.data(), vertexCount);
    c_b = futhark_new_f32_1d(ctx, h_cb.data(), vertexCount);

    sh_r = futhark_new_f32_2d(ctx, h_shr.data(), vertexCount, 15);
    sh_g = futhark_new_f32_2d(ctx, h_shg.data(), vertexCount, 15);
    sh_b = futhark_new_f32_2d(ctx, h_shb.data(), vertexCount, 15);

    futhark_context_sync(ctx);
    std::cout << "Scene uploaded to GPU." << std::endl;
    return true;
  }

  void render(int width, int height, const Camera &cam, std::vector<uint8_t> &pixel_buffer) {
    if (num_splats == 0)
      return;

    struct futhark_f32_2d *out_r, *out_g, *out_b;

    float aspect = (float)width / height;
    float fovRad = cam.fov * 3.14159f / 180.0f;
    float tanHalfFov = tan(fovRad / 2.0f);

    float fx = width / (2.0f * tanHalfFov);
    float fy = height / (2.0f * tanHalfFov);
    float cx = width / 2.0f;
    float cy = height / 2.0f;

    Quat rot = cam.getRotation();

    // Convert Camera-to-World (Orientation) to World-to-Camera (View)
    // q_w2c = conjugate(q_c2w)
    Quat q_w2c = {rot.w, -rot.x, -rot.y, -rot.z};

    // t_w2c = -R_w2c * pos_world
    // Rotate -pos by q_w2c
    float px = -cam.pos.x;
    float py = -cam.pos.y;
    float pz = -cam.pos.z;

    // Rotate vector p by quaternion q: v' = q * p * q^-1
    // Optimised formula for rotation: t = 2 * cross(q.xyz, v)
    // v' = v + q.w * t + cross(q.xyz, t)
    float tx = 2.0f * (q_w2c.y * pz - q_w2c.z * py);
    float ty = 2.0f * (q_w2c.z * px - q_w2c.x * pz);
    float tz = 2.0f * (q_w2c.x * py - q_w2c.y * px);

    float rx = px + q_w2c.w * tx + (q_w2c.y * tz - q_w2c.z * ty);
    float ry = py + q_w2c.w * ty + (q_w2c.z * tx - q_w2c.x * tz);
    float rz = pz + q_w2c.w * tz + (q_w2c.x * ty - q_w2c.y * tx);

    int res = futhark_entry_render(ctx, &out_r, &out_g, &out_b, (int64_t)width, (int64_t)height, fx, fy, cx, cy,
                                   q_w2c.w, q_w2c.x, q_w2c.y, q_w2c.z, rx, ry, rz, xyz_x, xyz_y, xyz_z, opas, s_x, s_y,
                                   s_z, rot_w, rot_x, rot_y, rot_z, c_r, c_g, c_b, sh_r, sh_g, sh_b);

    if (res != 0) {
      std::cerr << "Futhark Error: " << futhark_context_get_error(ctx) << std::endl;
      return;
    }

    std::vector<float> r_cpu(width * height);
    std::vector<float> g_cpu(width * height);
    std::vector<float> b_cpu(width * height);

    futhark_values_f32_2d(ctx, out_r, r_cpu.data());
    futhark_values_f32_2d(ctx, out_g, g_cpu.data());
    futhark_values_f32_2d(ctx, out_b, b_cpu.data());
    futhark_context_sync(ctx);

    pixel_buffer.resize(width * height * 3);
#pragma omp parallel for
    for (int i = 0; i < width * height; ++i) {
      pixel_buffer[i * 3 + 0] = (uint8_t)(std::min(1.0f, std::max(0.0f, r_cpu[i])) * 255);
      pixel_buffer[i * 3 + 1] = (uint8_t)(std::min(1.0f, std::max(0.0f, g_cpu[i])) * 255);
      pixel_buffer[i * 3 + 2] = (uint8_t)(std::min(1.0f, std::max(0.0f, b_cpu[i])) * 255);
    }

    futhark_free_f32_2d(ctx, out_r);
    futhark_free_f32_2d(ctx, out_g);
    futhark_free_f32_2d(ctx, out_b);
  }
};

// --- Main Application ---

GLuint textureID;

void updateTexture(int w, int h, const std::vector<uint8_t> &data) {
  glBindTexture(GL_TEXTURE_2D, textureID);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data.data());
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_point_cloud.ply>" << std::endl;
    return 1;
  }

  if (!glfwInit())
    return 1;
  const char *glsl_version = "#version 130";
  GLFWwindow *window = glfwCreateWindow(1280, 720, "GSplat Viewer (Ply)", NULL, NULL);
  if (!window)
    return 1;
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  // Initialize Futhark and Load PLY
  SplatRenderer renderer;
  if (!renderer.loadPly(argv[1])) {
    return 1;
  }

  glGenTextures(1, &textureID);
  std::vector<uint8_t> pixel_buffer;

  Camera cam;
  int render_w = 640;
  int render_h = 480;
  bool is_dragging = false;
  double last_x, last_y;

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // Input Handling
    if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
      double x, y;
      glfwGetCursorPos(window, &x, &y);
      if (!is_dragging) {
        is_dragging = true;
        last_x = x;
        last_y = y;
      }

      float dx = (float)(x - last_x);
      float dy = (float)(y - last_y);

      cam.yaw += dx * 0.2f;
      cam.pitch = std::max(-89.0f, std::min(89.0f, cam.pitch + dy * 0.2f));
      last_x = x;
      last_y = y;
    } else {
      is_dragging = false;
    }

    // Movement
    float speed = 0.05f;
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
      speed *= 4.0f;

    float rad = cam.yaw * 0.01745f;
    Vec3 forward = {-sinf(rad), 0, cosf(rad)};
    Vec3 right = {cosf(rad), 0, sinf(rad)};

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
      cam.pos.x += forward.x * speed;
      cam.pos.z += forward.z * speed;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
      cam.pos.x -= forward.x * speed;
      cam.pos.z -= forward.z * speed;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
      cam.pos.x -= right.x * speed;
      cam.pos.z -= right.z * speed;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
      cam.pos.x += right.x * speed;
      cam.pos.z += right.z * speed;
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
      cam.pos.y -= speed;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
      cam.pos.y += speed;

    // Render UI
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Settings");
    ImGui::Text("Splats loaded: %s", argv[1]);
    ImGui::SliderFloat("FOV", &cam.fov, 10.0f, 120.0f);
    ImGui::DragFloat3("Pos", &cam.pos.x, 0.1f);
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::End();

    // Render Scene
    renderer.render(render_w, render_h, cam, pixel_buffer);
    updateTexture(render_w, render_h, pixel_buffer);

    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    ImGui::Begin("Viewport");
    ImVec2 size = ImGui::GetContentRegionAvail();
    render_w = (int)size.x;
    render_h = (int)size.y;
    if (render_w < 1)
      render_w = 1;
    if (render_h < 1)
      render_h = 1;
    ImGui::Image((void *)(intptr_t)textureID, size, ImVec2(0, 0), ImVec2(1, 1));
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
