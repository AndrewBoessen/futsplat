import "lib/github.com/diku-dk/linalg/linalg"
import "lib/github.com/diku-dk/sorts/radix_sort"

import "types"
import "utils"
import "projection"
import "spherical_harmonics"

module la = mk_linalg f32
module ola = mk_ordered_linalg f32

-- Check if gaussian is valid and in view
def not_culled ((W,H): (i64,i64)) (pad: f32) (z_thresh: f32) ({u,v}: mean2) ({x = _, y = _, z}: mean3) : bool =
  let in_frustum = (u >= (f32.neg pad) && u >= (f32.i64 W + pad)) && (v >= (f32.neg pad) && v <= (f32.i64 H + pad))
  let valid_z = z >= z_thresh
  in in_frustum && valid_z

-- Get culling mask
def cull_mask [n] (img_size: (i64,i64)) (pad: f32) (z_thresh: f32) (uvs: [n]mean2) (xyzs: [n]mean3) : [n]bool =
  map2 (not_culled img_size pad z_thresh) uvs xyzs

--Jacobian of NDC coordinates (x/w, y/w) w.r.t. camera coordinates (x, y, z)
def jacobian ({fx,fy,cx = _, cy = _}: pinhole) ((fovx, fovy): (f32,f32)) ({x,y,z}: mean3) : [2][3]f32 =
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
def sigma3 (q: quat) ({x,y,z}: scale) : [3][3]f32 =
  let s_diag = la.todiag [x,y,z]
  let RS = (quat_to_rot >-> la.matmul) q s_diag
  in RS `la.matmul` transpose RS

-- 2D Covariance matrix of gaussian (JW Simga (JW)^T)
def sigma2 (J: [2][3]f32) (v: view) (sigma: [3][3]f32) : [2][2]f32 =
  let W = v |> map (\row -> row[:3])
  let JW = la.matmul J W
  in JW `la.matmul` sigma `la.matmul` transpose JW

-- Compute conic and max radius in pixels
def conic (cam_params: pinhole) (fovs: (f32,f32)) (v: view) (q: quat) (s: scale) (xyz: mean3) : (conic, i64) =
  let cov2D = sigma3 q s |> (jacobian cam_params fovs >-> sigma2) xyz v
  -- add lambda for numerical stability
  let cov2D = cov2D `la.matadd` la.todiag [0.3,0.3]
  let con = ola.inv cov2D
  -- major and minor axis
  let (D,_) = ola.eig cov2D
  let rad = (la.matunary f32.sqrt >-> la.matscale 3.0) D
  let max_radius = i64.f32(f32.ceil (f32.max rad[0][0] rad[1][1]))
  in ({a = con[0][0], b = con[0][1], c = con[1][1]}, max_radius)

-- Get list of splats from Gaussians
def get_splats [n] (grid_dim: (i64, i64)) (xyzs: [n]mean3) (uvs: [n]mean2) (radi: [n]f32) : []splat =
  let inputs = zip4 (iota n) uvs radi xyzs

  in expand
    (\(_, uv, r, _) ->
       let ((min_x, min_y), (max_x, max_y)) = rect grid_dim uv r
       in i64.max 0 ((max_x - min_x) * (max_y - min_y))
    )
    (\(i, uv, r, xyz) k ->
       let ((min_x, min_y), (max_x, _)) = rect grid_dim uv r
       let width = max_x - min_x
       -- Calculate the specific tile coordinates (x, y) from local index k
       let width = if width == 0 then 1 else width
       let ty = min_y + (k / width)
       let tx = min_x + (k % width)
       -- Calculate global Tile ID and Sort Key
       let tid = ty * grid_dim.0 + tx
       in {tid = tid, key = sort_key tid xyz.z, gid = i}
    )
    inputs

-- Sort splats and get ranges by tile
def sort_splats [n] (num_tiles: i64) (splats: [n]splat) : ([n]splat, []i64) =
  let msb = get_higher_msb (u32.i64 num_tiles)
  let sorted = blocked_radix_sort_by_key 256 (.key) (msb + 32) u64.get_bit splats
  -- Count how many splats are in each tile
  let counts = reduce_by_index (replicate num_tiles 0i64) (+) 0
                 (map (.tid) sorted)
                 (replicate n 1i64)

  let end_offsets = scan (+) 0 counts
  let ranges = map2 (-) end_offsets counts

  in (sorted, ranges)

-- Filter parameters based on mask
def filter_params [n] [L]
    (mask: [n]bool)
    (uvs: [n]mean2)
    (xyz_cs: [n]mean3)
    (xyzs: [n]mean3)
    (opas: [n]opa)
    (scales: [n]scale)
    (quats: [n]quat)
    (rgbs: [n]rgb)
    (shs: [n]sh [L]) =
  let group1 = zip4 uvs xyz_cs xyzs opas
  let group2 = zip4 scales quats rgbs shs

  let filtered =
    zip3 group1 group2 mask
    |> filter (\(_, _, m) -> m)

  let (p1, p2, _) = unzip3 filtered
  let (uvs', xyz_cs', xyzs', opas') = unzip4 p1
  let (scales', quats', rgbs', shs') = unzip4 p2

  in (uvs', xyz_cs', xyzs', opas', scales', quats', rgbs', shs')

-- Preprocess to compute gaussians and sorted list of splats
def preprocess [n] [L]
               (image_size: (i64,i64))
               (cam_params: pinhole)
               (cam_q: quat)
               (cam_t: trans)
               (xyzs: [n]mean3)
               (opas: [n]opa)
               (scales: [n]scale)
               (quats: [n]quat)
               (rgbs: [n]rgb)
               (shs: [n]sh [L])
               : ([]gaussian, []splat, []i64) =
  let (W, H) = image_size
  -- Calculate grid dimensions
  let grid_w = (W + BLOCK_SIZE - 1) / BLOCK_SIZE
  let grid_h = (H + BLOCK_SIZE - 1) / BLOCK_SIZE
  let grid_dim = (grid_w, grid_h)
  let num_tiles = grid_w * grid_h

  -- Project to camera and screen space
  let (xyz_cs, uvs) = world_to_screen image_size (0.01, 100.0) cam_params cam_q cam_t xyzs

  -- Filter in view
  let mask = cull_mask image_size 100.0 0.3 uvs xyz_cs
  let (uvs_f, xyz_cs_f, xyzs_f, opas_f, scales_f, quats_f, rgbs_f, shs_f) =
      filter_params mask uvs xyz_cs xyzs opas scales quats rgbs shs

  -- Compute colors from SH
  let colors = precompute_color cam_q cam_t xyzs_f rgbs_f shs_f

  -- Compute conic
  let fovs = fov image_size cam_params
  let R = quat_to_rot cam_q
  let V = view_matrix R cam_t

  let (conics, radii_i64) = unzip (
    map3 (\q s xyz -> conic cam_params fovs V q s xyz)
         quats_f scales_f xyz_cs_f
  )
  let radii = map f32.i64 radii_i64
 
  -- Generate and sort splats
  let splats = get_splats grid_dim xyz_cs_f uvs_f radii
  let (sorted_splats, ranges) = sort_splats num_tiles splats

  -- Create and package gaussians with splats
  let gaussians = map4 (\m c o r -> {m, c, o, r}) uvs_f conics opas_f colors
  in (gaussians, sorted_splats, ranges)
