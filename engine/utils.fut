import "lib/github.com/diku-dk/linalg/linalg"

import "types"

module la = mk_linalg f32

-- Sigmoid activation function
def sigmoid (x: f32) : f32 =
  1.0 / (1.0 + f32.exp (f32.neg x))

-- Convert NDC coordinates to pixel coordinates
def ndc_to_pix ({u,v}: mean2) ((W,H): (i64, i64)) : mean2 =
  let u = (u + 1.0) * f32.i64 W * 0.5
  let v = (v + 1.0) * f32.i64 H * 0.5
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
    let min_x = i64.min grid_x (i64.max 0 (i64.f32(u - radius) // TILE_SIZE))
    let min_y = i64.min grid_y (i64.max 0 (i64.f32(v - radius) // TILE_SIZE))
    in (min_x, min_y)
  let rect_max =
    let max_x = i64.min grid_x (i64.max 0 ((i64.f32(u + radius) + TILE_SIZE - 1) // TILE_SIZE))
    let max_y = i64.min grid_y (i64.max 0 ((i64.f32(v + radius) + TILE_SIZE - 1) // TILE_SIZE))
    in (max_x, max_y)
  in (rect_min, rect_max)

-- Create 64 bit key from tile id and depth
def sort_key (tid: i64) (z: f32) : u64 =
  let tile_part = (u64.i64 tid) << 32
  let depth_part = u64.u32 (f32.to_bits z)
  in tile_part | depth_part

-- Find the next highest bit of the MSB
def get_higher_msb (n: u32) : i32 =
  if n == 0 then 0 else 32 - (u32.clz n)

-- Expands input array 'arr' based on a size function 'sz' and a generator 'get'
def expand 'a 'b [n] (sz: a -> i64) (get: a -> i64 -> b) (arr: [n]a) : []b =
  let counts = map sz arr
  let offsets = scan (+) 0 counts
  let total_len = if n > 0 then offsets[n-1] else 0

  in map (\i ->
    let (g_idx, _) = 
      loop (l, r) = (0, n - 1) while l < r do
        let mid = r - (r - l) / 2
        in if offsets[mid] <= i then (mid, r) else (l, mid - 1)
    let start_idx = if g_idx == 0 then 0 else offsets[g_idx - 1]
    let k = i - start_idx
    in get arr[g_idx] k
  ) (iota total_len)
