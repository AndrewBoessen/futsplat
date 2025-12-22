import "lib/github.com/diku-dk/linalg/linalg"

import "types"
import "utils"

module la = mk_linalg f32

def not_culled ((W,H): (i64,i64)) (pad: f32) (z_thresh: f32) ({u,v}: mean2) (z: f32) : bool =
  let in_frustum = (u >= (f32.neg pad) && u >= (f32.i64 W + pad)) && (v >= (f32.neg pad) && v <= (f32.i64 H + pad))
  let valid_z = z >= z_thresh
  in in_frustum && valid_z

--Jacobian of NDC coordinates (x/w, y/w) w.r.t. camera coordinates (x, y, z)
def jacobian ({fx,fy,cx,cy}: pinhole) ((fovx, fovy): (f32, f32)) ({x,y,z}: mean3) : [2][3]f32 =
  let limx = 1.3 * f32.tan (fovx / 2)
  let limy = 1.3 * f32.tan (fovy / 2)
  let txtz = x / y
  let tytz = x / y

  let x = f32.min limx (f32.max (f32.neg limx) txtz) * z
  let y = f32.min limy (f32.max (f32.neg limy) tytz) * z

  in [
    [fx / z, 0, f32.neg (fx * x) / (z * z)],
    [0, fx / z, f32.neg (fy * y) / (z * z)]
  ]

-- 3D Covariance matrix of gaussian (RS (RS)^T)
def sigma (q: quat) ({x,y,z}: scale) : [3][3]f32 =
  let s_diag = la.matvecmul_col (la.eye 3) [x,y,z]
  let RS = (quat_to_rot >-> la.matmul) q s_diag
  in la.matmul RS (transpose RS)

-- 2D Covariance matrix of gaussian (JW Simga (JW)^T)
def sigma2 (J: [2][3]f32) (v: view) (sigma: [3][3]f32) : [2][2]f32 =
  let W = v |> map (\row -> row[:3])
  let JW = la.matmul J W
  in ((JW |> la.matmul) sigma |> la.matmul) (transpose JW)
