import "lib/github.com/diku-dk/linalg/linalg"

import "types"
import "utils"

module la = mk_linalg f32

-- Calculate FOV from camera parameters
def fov ((W,H): (i64,i64)) ({fx,fy,cx = _,cy = _}: pinhole) : (f32, f32) =
  let fovx = 2 * f32.atan (f32.i64 W / (2 * fx))
  let fovy = 2 * f32.atan (f32.i64 H / (2 * fy))
  in (fovx, fovy)

-- Create view matrix from world to camera corrdinates
def view_matrix (R: rot) (t: trans) : view =
  [
    [R[0,0], R[0,1], R[0,2], t[0]],
    [R[1,0], R[1,1], R[1,2], t[1]],
    [R[2,0], R[2,1], R[2,2], t[2]],
  ]

-- Create projection matrix from camera to NDC pixel corrdinates
def proj_matrix ((znear, zfar): (f32, f32)) ((fovx, fovy): (f32, f32)) : proj =
  let pixel_fx = f32.tan (fovx / 2)
  let pixel_fy = f32.tan (fovy / 2)

  let top = pixel_fy * znear
  let bot = f32.neg top
  let right = pixel_fx * znear
  let left = f32.neg right

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
def screen_proj ((W,H): (i64,i64)) (p: proj) ({x,y,z}: mean3) : mean2 =
  let cam_space = [x,y,z,1]
  let screen = la.matvecmul_row p cam_space
  let ndc = {u = screen[0] / screen[3], v = screen[1] / screen[3]}
  in ndc_to_pix ndc (W,H)

-- Project from world to screen coordinates
def world_to_screen [n]
                    (image_dim: (i64,i64))
                    (z_thresh: (f32,f32))
                    (cam_params: pinhole)
                    (cam_quat: quat)
                    (cam_trans: trans)
                    (world: [n]mean3)
                    : ([n]mean3, [n]mean2) =
  let v = (norm_quat >-> quat_to_rot >-> view_matrix) cam_quat cam_trans
  let p = ((fov image_dim) >-> (proj_matrix z_thresh)) cam_params
  let cam_space = world |> map (cam_view v)
  let screen_space = cam_space |> map (screen_proj image_dim p)
  in (cam_space, screen_space)
