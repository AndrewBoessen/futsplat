import "types"

-- Scale color vector
def scale_rgb (s: f32) ({r,g,b}: rgb) : rgb =
  {r = r * s, g = g * s, b = b * s}

-- Accumulate color vector
def add_rgb ({r = r1,g = g1,b = b1}: rgb) ({r = r2, g = g2, b = b2}: rgb) : rgb =
  {r = r1 + r2, g = g1 + g2, b = b1 + b2}

-- Get normalized view direction relative to camera
def cam_view ((cx, cy, cz): (f32, f32, f32)) ({x,y,z}: mean3) : (f32, f32, f32) =
  let (dx, dy, dz) = (x - cx, y - cy, z - cz)
  let len = f32.sqrt (dx**2 + dy**2 + dz**2)
  in (dx / len, dy / len, dz / len)

def sh_to_color [L] ((x,y,z): (f32, f32, f32)) (color: rgb) (coeffs: sh [L]) : rgb =
  let deg = i32.i64 L
  -- Band 0 (RGB params)
  let result = scale_rgb SH_L0 color

  -- Band 1
  let result =
    if deg > 0 then
      let t1 = scale_rgb (0.0 - SH_L1 * y) coeffs[1]
      let t2 = scale_rgb (      SH_L1 * z) coeffs[2]
      let t3 = scale_rgb (0.0 - SH_L1 * x) coeffs[3]
      in result `add_rgb` t1 `add_rgb` t2 `add_rgb` t3
    else result

  -- Band 2
  let result =
    if deg > 0 then
      let xx = x*x
      let yy = y*y
      let zz = z*z
      let xy = x*y
      let yz = y*z
      let xz = x*z

      let t0 = scale_rgb (SH_L2[0] * xy) coeffs[4]
      let t1 = scale_rgb (SH_L2[1] * yz) coeffs[5]
      let t2 = scale_rgb (SH_L2[2] * (2.0 * zz - xx - yy)) coeffs[6]
      let t3 = scale_rgb (SH_L2[3] * xz) coeffs[7]
      let t4 = scale_rgb (SH_L2[4] * (xx - yy)) coeffs[8]

      in result `add_rgb` t0 `add_rgb` t1 `add_rgb` t2 `add_rgb` t3 `add_rgb` t4
    else result

  -- Band 3
  let result =
    if deg > 0 then
      let xx = x*x
      let yy = y*y
      let zz = z*z
      let xy = x*y

      let t0 = scale_rgb (SH_L3[0] * y * (3.0 * xx - yy)) coeffs[9]
      let t1 = scale_rgb (SH_L3[1] * xy * z) coeffs[10]
      let t2 = scale_rgb (SH_L3[2] * y * (4.0 * zz - xx - yy)) coeffs[11]
      let t3 = scale_rgb (SH_L3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)) coeffs[12]
      let t4 = scale_rgb (SH_L3[4] * x * (4.0 * zz - xx - yy)) coeffs[13]
      let t5 = scale_rgb (SH_L3[5] * z * (xx - yy)) coeffs[14]
      let t6 = scale_rgb (SH_L3[6] * x * (xx - 3.0 * yy)) coeffs[15]

      in result `add_rgb` t0 `add_rgb` t1 `add_rgb` t2 `add_rgb` t3 
                `add_rgb` t4 `add_rgb` t5 `add_rgb` t6
    else result

  let bias_rgb = {r=0.5, g=0.5, b=0.5}
  in add_rgb result bias_rgb
