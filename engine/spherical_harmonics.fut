import "types"
import "utils"

-- Scale color vector
def scale_rgb (s: f32) ({r,g,b}: rgb) : rgb =
  {r = r * s, g = g * s, b = b * s}

-- Accumulate color vector
def add_rgb ({r = r1,g = g1,b = b1}: rgb) ({r = r2, g = g2, b = b2}: rgb) : rgb =
  {r = r1 + r2, g = g1 + g2, b = b1 + b2}

-- Get normalized view direction relative to camera
def cam_dir ((cx, cy, cz): (f32, f32, f32)) ({x,y,z}: mean3) : (f32, f32, f32) =
  let (dx, dy, dz) = (x - cx, y - cy, z - cz)
  let len = f32.sqrt (dx**2 + dy**2 + dz**2)
  in (dx / len, dy / len, dz / len)

-- Compute color for single point
def sh_to_color [L] ((x,y,z): (f32, f32, f32)) (dc_color: rgb) (coeffs: sh [L]) : rgb =
  let deg = i32.i64 L

  -- Precompute common geometric terms
  let xx = x * x
  let yy = y * y
  let zz = z * z
  let xy = x * y
  let yz = y * z
  let xz = x * z

  -- Helper: Sum a list of RGB contributions
  let sum_terms (base: rgb) (terms: []rgb) : rgb =
    reduce add_rgb base terms

  -- Band 0: DC Component (Always present)
  let base = scale_rgb SH_L0 dc_color

  -- Band 1 (Degree >= 1)
  let res_b1 =
    if deg >= 1 then
      let terms = [
        scale_rgb (0.0 - SH_L1 * y) coeffs[0],
        scale_rgb (      SH_L1 * z) coeffs[1],
        scale_rgb (0.0 - SH_L1 * x) coeffs[2]
      ]
      in sum_terms base terms
    else base

  -- Band 2 (Degree >= 2)
  let res_b2 =
    if deg >= 2 then
      let terms = [
        scale_rgb (SH_L2[0] * xy) coeffs[3],
        scale_rgb (SH_L2[1] * yz) coeffs[4],
        scale_rgb (SH_L2[2] * (2.0 * zz - xx - yy)) coeffs[5],
        scale_rgb (SH_L2[3] * xz) coeffs[6],
        scale_rgb (SH_L2[4] * (xx - yy)) coeffs[7]
      ]
      in sum_terms res_b1 terms
    else res_b1

  -- Band 3 (Degree >= 3)
  let res_b3 =
    if deg >= 3 then
      let terms = [
        scale_rgb (SH_L3[0] * y * (3.0 * xx - yy)) coeffs[8],
        scale_rgb (SH_L3[1] * xy * z) coeffs[9],
        scale_rgb (SH_L3[2] * y * (4.0 * zz - xx - yy)) coeffs[10],
        scale_rgb (SH_L3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)) coeffs[11],
        scale_rgb (SH_L3[4] * x * (4.0 * zz - xx - yy)) coeffs[12],
        scale_rgb (SH_L3[5] * z * (xx - yy)) coeffs[13],
        scale_rgb (SH_L3[6] * x * (xx - 3.0 * yy)) coeffs[14]
      ]
      in sum_terms res_b2 terms
    else res_b2

  let bias_rgb = {r=0.5, g=0.5, b=0.5}
  in add_rgb res_b3 bias_rgb

-- Procompute colors for n Gaussians
def precompute_color [n] [L]
                     (cam_quat: quat)
                     (cam_trans: trans)
                     (means3: [n]mean3)
                     (rgbs: [n]rgb)
                     (shs: [n]sh [L])
                     : [n]rgb =
  let c_pos = (quat_to_rot >-> campos) cam_quat cam_trans
  in map3 (\m c s -> (cam_dir c_pos >-> sh_to_color) m c s) means3 rgbs shs
