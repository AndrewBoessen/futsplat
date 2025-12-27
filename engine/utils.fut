import "lib/github.com/diku-dk/linalg/linalg"

import "types"

module la = mk_linalg f32

-- Convert NDC coordinates to pixel coordinates
def ndc_to_pix ({u,v}: mean2) ((W,H): (i64, i64)) : mean2 =
  let u = ((u + 1.0) * f32.i64 W - 1.0) * 0.5
  let v = ((v + 1.0) * f32.i64 H - 1.0) * 0.5
  in {u,v}

-- Normalize quaternion
def norm_quat ({w,x,y,z}: quat) : quat =
  let len_sq = w**2 + x**2 + y**2 + z**2
  let len = f32.sqrt len_sq
  in {w = w / len, x = x / len, y = y / len, z = z / len}

def campos (R: rot) (t: trans) : (f32, f32 ,f32) =
  let Rt = transpose R
  let neg_t = map f32.neg t
  let c_arr = la.matvecmul_row Rt neg_t
  in (c_arr[0], c_arr[1], c_arr[2])

def quat_to_rot ({w,x,y,z}: quat) : rot =
  [
	  [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
	  [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
	  [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)]
  ]

-- Get outer bounding box lower and upper coordinates
def rect ((grid_x, grid_y): (i64, i64)) ({u,v}: mean2) (radius: f32) : ((i64,i64),(i64,i64)) =
  let rect_min =
    let min_x = i64.min grid_x (i64.max 0 (i64.f32(u - radius) // BLOCK_SIZE))
    let min_y = i64.min grid_y (i64.max 0 (i64.f32(v - radius) // BLOCK_SIZE))
    in (min_x, min_y)
  let rect_max =
    let max_x = i64.min grid_x (i64.max 0 ((i64.f32(u + radius) + BLOCK_SIZE - 1) // BLOCK_SIZE))
    let max_y = i64.min grid_y (i64.max 0 ((i64.f32(v + radius) + BLOCK_SIZE - 1) // BLOCK_SIZE))
    in (max_x, max_y)
  in (rect_min, rect_max)

-- Get tile range from bounding box
def tile_range ((grid_x, _): (i64, i64)) (((min_x,min_y),(max_x,max_y)): ((i64,i64),(i64,i64))) : []i64 =
  let w = max_x - min_x
  let h = max_y - min_y

  let count = if w > 0 && h > 0 then w * h else 0
  in map (\i ->
    let dy = i / w
    let dx = i % w
    let y = min_y + dy
    let x = min_x + dx
    in y * grid_x + x
  ) (iota count)

-- Sigmoid activation function
def sigmoid (x: f32) : f32 =
  1.0 / (1.0 + f32.exp (f32.neg x))
