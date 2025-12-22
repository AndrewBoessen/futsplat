import "lib/github.com/diku-dk/linalg/linalg"

import "types"

module la = mk_linalg f32

-- Calculate FOV from camera parameters
def fov ((H, W): (i64, i64)) ({fx,fy,cx,cy}: pinhole) : (f32, f32) =
  let fovx = 2 * f32.atan (f32.i64 W / (2 * fx))
  let fovy = 2 * f32.atan (f32.i64 H / (2 * fy))
  in (fovx, fovy)

-- Create view matrix from world to camera corrdinates
def view_matrix (R: rot) (t: trans) : view =
  let Rt = transpose R
  in [
    [Rt[0,0], Rt[0,1], Rt[0,2], t[0]],
    [Rt[1,0], Rt[1,1], Rt[1,2], t[1]],
    [Rt[2,0], Rt[2,1], Rt[2,2], t[2]],
  ]

-- Create projection matrix from camera to NDC pixel corrdinates
def proj_matrix ((znear, zfar): (f32, f32)) ((fovx, fovy): (f32, f32)) : proj =
  let pixel_fx = f32.tan (fovx / 2)
  let pixel_fy = f32.tan (fovy / 2)

  let top = pixel_fy * znear
  let bot = f32.neg top
  let left = pixel_fx * znear
  let right = f32.neg left

  in [
    [2.0 * znear / (right - left), 0, 0, 0],
    [0, 2.0 * znear / (top - bot), 0, 0],
    [0, 0, zfar / (zfar - znear), f32.neg (zfar * znear) / (zfar - znear)],
    [0, 0, 1.0, 0]
  ]

-- Project mean to camera space
def cam_view (v: view) ({x,y,z}: mean3) : mean3 =
  let mean_arr = [x,y,z,1]
  let cam_arr = la.matvecmul_row v mean_arr
  in {x = cam_arr[0], y = cam_arr[1], z = cam_arr[2]}

-- Project mean to 2D pixel coordinates
def screen_proj ((H,W): (i64,i64)) (p: proj) ({x,y,z}: mean3) : mean2 =
  let cam_view = [x,y,z,1]
  let screen = la.matvecmul_row p cam_view
  let {u = ndc_x, v = ndc_y} = {u = screen[0] / screen[3], v = screen[0] / screen[3]}
  in {u = (ndc_x * 0.5 + 0.5) * f32.i64 W, v = (ndc_y * 0.5 + 0.5) * f32.i64 H}
