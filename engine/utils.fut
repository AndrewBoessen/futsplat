import "types"

-- Convert NDC coordinates to pixel coordinates
def ndc_to_pix ((x, y): (f32, f32)) ((H, W): (i64, i64)) : mean2 =
  let u = ((x + 1.0) * f32.i64 W - 1.0) * 0.5
  let v = ((y + 1.0) * f32.i64 H - 1.0) * 0.5
  in {u,v}

-- Normalize quaternion
def norm_quat ({w,x,y,z}: quat) : quat =
  let len_sq = w**2 + x**2 + y**2 + z**2
  let len = f32.sqrt len_sq
  in {w = w / len, x = x / len, y = y / len, z = z / len}

-- Get outer bounding box lower and upper coordinates
def rect ({u,v}: mean2) (radius: f32) (grid_x: i64) (grid_y: i64) : ((i64,i64),(i64,i64)) =
  let rect_min =
    let min_x = i64.min grid_x (i64.max 0 (i64.f32(u - radius) // BLOCK_SIZE))
    let min_y = i64.min grid_y (i64.max 0 (i64.f32(v - radius) // BLOCK_SIZE))
    in (min_x, min_y)
  let rect_max =
    let max_x = i64.min grid_x (i64.max 0 ((i64.f32(u + radius) + BLOCK_SIZE - 1) // BLOCK_SIZE))
    let max_y = i64.min grid_x (i64.max 0 ((i64.f32(u + radius) + BLOCK_SIZE - 1) // BLOCK_SIZE))
    in (max_x, max_y)
  in (rect_min, rect_max)

-- Sigmoid activation function
def sigmoid (x: f32) : f32 =
  1.0 / (1.0 + f32.exp (f32.neg x))
