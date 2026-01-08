module utils = import "utils"
module spherical_harmonics = import "spherical_harmonics"
module projection = import "projection"
module render = import "render"
import "types"

-- Utils Tests

-- ==
-- entry: test_sigmoid
-- input { 0.0f32 } output { 0.5f32 }
-- input { 100.0f32 } output { 1.0f32 }
-- input { -100.0f32 } output { 0.0f32 }
entry test_sigmoid (x: f32) = utils.sigmoid x

-- ==
-- entry: test_ndc_to_pix
-- input { 0.0f32 0.0f32 100i64 100i64 } output { 50.0f32 50.0f32 }
-- input { -1.0f32 -1.0f32 100i64 100i64 } output { 0.0f32 0.0f32 }
-- input { 1.0f32 1.0f32 100i64 100i64 } output { 100.0f32 100.0f32 }
entry test_ndc_to_pix (u: f32) (v: f32) (W: i64) (H: i64) =
  let res = utils.ndc_to_pix {u, v} (W, H)
  in (res.u, res.v)

-- ==
-- entry: test_norm_quat
-- input { 1.0f32 0.0f32 0.0f32 0.0f32 } output { 1.0f32 0.0f32 0.0f32 0.0f32 }
-- input { 2.0f32 0.0f32 0.0f32 0.0f32 } output { 1.0f32 0.0f32 0.0f32 0.0f32 }
-- input { 0.0f32 3.0f32 4.0f32 0.0f32 } output { 0.0f32 0.6f32 0.8f32 0.0f32 }
entry test_norm_quat (w: f32) (x: f32) (y: f32) (z: f32) =
  let q = utils.norm_quat {w, x, y, z}
  in (q.w, q.x, q.y, q.z)

-- ==
-- entry: test_campos
-- input {
--   1.0f32 0.0f32 0.0f32
--   0.0f32 1.0f32 0.0f32
--   0.0f32 0.0f32 1.0f32
--   1.0f32 2.0f32 3.0f32
-- }
-- output { -1.0f32 -2.0f32 -3.0f32 }
entry test_campos (r00: f32) (r01: f32) (r02: f32) (r10: f32) (r11: f32) (r12: f32) (r20: f32) (r21: f32) (r22: f32) (tx: f32) (ty: f32) (tz: f32) =
  let R = [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
  let t = [tx, ty, tz]
  in utils.campos R t

-- ==
-- entry: test_quat_to_rot
-- input { 1.0f32 0.0f32 0.0f32 0.0f32 }
-- output {
--   1.0f32 0.0f32 0.0f32
--   0.0f32 1.0f32 0.0f32
--   0.0f32 0.0f32 1.0f32
-- }
-- input { 0.0f32 1.0f32 0.0f32 0.0f32 }
-- output {
--   1.0f32 0.0f32 0.0f32
--   0.0f32 -1.0f32 0.0f32
--   0.0f32 0.0f32 -1.0f32
-- }
entry test_quat_to_rot (w: f32) (x: f32) (y: f32) (z: f32) =
  let R = utils.quat_to_rot {w, x, y, z}
  in (R[0,0], R[0,1], R[0,2],
      R[1,0], R[1,1], R[1,2],
      R[2,0], R[2,1], R[2,2])

-- ==
-- entry: test_get_higher_msb
-- input { 0u32 } output { 0i32 }
-- input { 1u32 } output { 1i32 }
-- input { 2u32 } output { 2i32 }
-- input { 3u32 } output { 2i32 }
-- input { 4u32 } output { 3i32 }
entry test_get_higher_msb (n: u32) = utils.get_higher_msb n

-- ==
-- entry: test_rect
-- input { 10i64 10i64 50.0f32 50.0f32 10.0f32 } output { 2i64 2i64 4i64 4i64 }
-- input { 10i64 10i64 0.0f32 0.0f32 5.0f32 } output { 0i64 0i64 1i64 1i64 }
entry test_rect (gx: i64) (gy: i64) (u: f32) (v: f32) (radius: f32) = 
  let ((min_x, min_y), (max_x, max_y)) = utils.rect (gx, gy) {u, v} radius
  in (min_x, min_y, max_x, max_y)

-- ==
-- entry: test_sort_key
-- input { 1i64 1.0f32 } output { 5360320512u64 }
entry test_sort_key (tid: i64) (z: f32) = utils.sort_key tid z

-- Spherical Harmonics Tests

-- ==
-- entry: test_scale_rgb
-- input { 2.0f32 1.0f32 2.0f32 3.0f32 } output { 2.0f32 4.0f32 6.0f32 }
entry test_scale_rgb (s: f32) (r: f32) (g: f32) (b: f32) =
  let res = spherical_harmonics.scale_rgb s {r, g, b}
  in (res.r, res.g, res.b)

-- ==
-- entry: test_add_rgb
-- input { 1.0f32 2.0f32 3.0f32 4.0f32 5.0f32 6.0f32 } output { 5.0f32 7.0f32 9.0f32 }
entry test_add_rgb (r1: f32) (g1: f32) (b1: f32) (r2: f32) (g2: f32) (b2: f32) =
  let res = spherical_harmonics.add_rgb {r=r1, g=g1, b=b1} {r=r2, g=g2, b=b2}
  in (res.r, res.g, res.b)

-- ==
-- entry: test_cam_dir_sh
-- input { 0.0f32 0.0f32 0.0f32 1.0f32 0.0f32 0.0f32 } output { 1.0f32 0.0f32 0.0f32 }
entry test_cam_dir_sh (cx: f32) (cy: f32) (cz: f32) (x: f32) (y: f32) (z: f32) =
  let res = spherical_harmonics.cam_dir (cx, cy, cz) {x, y, z}
  in (res.0, res.1, res.2)

-- ==
-- entry: test_sh_to_color
-- input { 1.0f32 0.0f32 0.0f32 0.5f32 0.5f32 0.5f32 [1.0f32, 0.0f32, 0.0f32] [1.0f32, 0.0f32, 0.0f32] [1.0f32, 0.0f32, 0.0f32] }
-- output { 0.640986f32 0.640986f32 0.640986f32 }
entry test_sh_to_color (x: f32) (y: f32) (z: f32) (dr: f32) (dg: f32) (db: f32) (c1: []f32) (c2: []f32) (c3: []f32) =
  let sh_coeffs = map3 (\r g b -> {r,g,b}) c1 c2 c3
  let band = 1
  let res = spherical_harmonics.sh_to_color (x,y,z) {r=dr, g=dg, b=db} (sh_coeffs :> [(band + 1) * (band + 1) - 1]rgb)
  in (res.r, res.g, res.b)

-- Projection Tests

-- ==
-- entry: test_fov
-- input { 100i64 100i64 50.0f32 50.0f32 } output { 1.5707964f32 1.5707964f32 }
entry test_fov (W: i64) (H: i64) (fx: f32) (fy: f32) =
  let (fovx, fovy) = projection.fov (W, H) {fx, fy, cx=0.0, cy=0.0}
  in (fovx, fovy)

-- ==
-- entry: test_view_matrix
-- input { 1.0f32 0.0f32 0.0f32 0.0f32 1.0f32 0.0f32 0.0f32 0.0f32 1.0f32 1.0f32 2.0f32 3.0f32 } output { 1.0f32 0.0f32 0.0f32 1.0f32 0.0f32 1.0f32 0.0f32 2.0f32 0.0f32 0.0f32 1.0f32 3.0f32 }
entry test_view_matrix (r00: f32) (r01: f32) (r02: f32) (r10: f32) (r11: f32) (r12: f32) (r20: f32) (r21: f32) (r22: f32) (tx: f32) (ty: f32) (tz: f32) =
  let R = [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
  let t = [tx, ty, tz]
  let V = projection.view_matrix R t
  in (V[0,0], V[0,1], V[0,2], V[0,3],
      V[1,0], V[1,1], V[1,2], V[1,3],
      V[2,0], V[2,1], V[2,2], V[2,3])

-- ==
-- entry: test_proj_matrix
-- input { 1.0f32 100.0f32 1.5707964f32 1.5707964f32 }
-- output {
--   -1.0000001f32 0.0f32 0.0f32 0.0f32
--   0.0f32 1.0000001f32 0.0f32 0.0f32
--   0.0f32 0.0f32 1.010101f32 -1.0101011f32
--   0.0f32 0.0f32 1.0f32 0.0f32
-- }
entry test_proj_matrix (znear: f32) (zfar: f32) (fovx: f32) (fovy: f32) =
  let P = projection.proj_matrix (znear, zfar) (fovx, fovy)
  in (P[0,0], P[0,1], P[0,2], P[0,3],
      P[1,0], P[1,1], P[1,2], P[1,3],
      P[2,0], P[2,1], P[2,2], P[2,3],
      P[3,0], P[3,1], P[3,2], P[3,3])

-- ==
-- entry: test_cam_view_proj
-- input { 
--   1.0f32 0.0f32 0.0f32 10.0f32
--   0.0f32 1.0f32 0.0f32 20.0f32
--   0.0f32 0.0f32 1.0f32 30.0f32
--   0.0f32 0.0f32 0.0f32
-- }
-- output { 10.0f32 20.0f32 30.0f32 }
entry test_cam_view_proj (v00: f32) (v01: f32) (v02: f32) (v03: f32)
                         (v10: f32) (v11: f32) (v12: f32) (v13: f32)
                         (v20: f32) (v21: f32) (v22: f32) (v23: f32)
                         (x: f32) (y: f32) (z: f32) =
  let V = [[v00, v01, v02, v03], [v10, v11, v12, v13], [v20, v21, v22, v23]]
  let res = projection.cam_view V {x, y, z}
  in (res.x, res.y, res.z)

-- Render Tests

-- ==
-- entry: test_not_culled
-- input { 100i64 100i64 10.0f32 0.1f32 50.0f32 50.0f32 1.0f32 } output { true }
-- input { 100i64 100i64 10.0f32 0.1f32 -20.0f32 50.0f32 1.0f32 } output { false }
-- input { 100i64 100i64 10.0f32 0.1f32 50.0f32 50.0f32 0.0f32 } output { false }
entry test_not_culled (W: i64) (H: i64) (pad: f32) (thresh: f32) (u: f32) (v: f32) (z: f32) =
  render.not_culled (W, H) pad thresh {u, v} {x=0.0, y=0.0, z}

-- ==
-- entry: test_jacobian
-- input { 100.0f32 100.0f32 1.5707964f32 1.5707964f32 1.0f32 1.0f32 10.0f32 }
-- output {
--    10.0f32 0.0f32 -1.0f32
--    0.0f32 10.0f32 -1.0f32
-- }
entry test_jacobian (fx: f32) (fy: f32) (fovx: f32) (fovy: f32) (x: f32) (y: f32) (z: f32) =
  let J = render.jacobian {fx, fy, cx=0.0, cy=0.0} (fovx, fovy) {x, y, z}
  in (J[0][0], J[0][1], J[0][2],
      J[1][0], J[1][1], J[1][2])

-- ==
-- entry: test_raster_tile
-- input {
--   16i64 16i64
--   0.0f32 0.0f32
--   1.0f32 0.5f32 0.5f32
--   1.0f32
--   1.0f32 0.0f32 0.0f32
-- }
-- output { 1.0f32 0.0f32 0.0f32 }
entry test_raster_tile (W: i64) (H: i64)
                       (u: f32) (v: f32)
                       (ca: f32) (cb: f32) (cc: f32)
                       (opa: f32)
                       (r: f32) (g: f32) (b: f32) =
  let gaussian = {m = {u, v}, c = {a=ca, b=cb, c=cc}, o = opa, r = {r, g, b}}
  -- Create a dummy splat pointing to the gaussian at index 0
  let splat = {tid = 0i64, key = 0u64, gid = 0i64}
  -- We pass range 0 to 1 because we have 1 splat
  let tile_pixels = render.raster_tile (W, H) (0, 0) [gaussian] [splat] 0 1
  -- Check pixel at 0,0
  let p = tile_pixels[0][0]
  in (p.r, p.g, p.b)

-- ==
-- entry: test_conic
-- input {
--   100.0f32 100.0f32 1.5707964f32 1.5707964f32
--   1.0f32 0.0f32 0.0f32 0.0f32 0.0f32 1.0f32 0.0f32 0.0f32 0.0f32 0.0f32 1.0f32 0.0f32
--   0.70710678f32 0.0f32 0.0f32 0.70710678f32
--   2.0f32 1.0f32 1.0f32
--   1.0f32 1.0f32 5.0f32
-- }
-- output { 0.002403f32 -0.000024f32 0.000619f32 121.0f32 }
entry test_conic (fx: f32) (fy: f32) (fovx: f32) (fovy: f32)
                 (v00: f32) (v01: f32) (v02: f32) (v03: f32)
                 (v10: f32) (v11: f32) (v12: f32) (v13: f32)
                 (v20: f32) (v21: f32) (v22: f32) (v23: f32)
                 (qw: f32) (qx: f32) (qy: f32) (qz: f32)
                 (sx: f32) (sy: f32) (sz: f32)
                 (x: f32) (y: f32) (z: f32) =
  let cam_params = {fx, fy, cx=0.0, cy=0.0}
  let fovs = (fovx, fovy)
  let V = [[v00, v01, v02, v03], [v10, v11, v12, v13], [v20, v21, v22, v23]]
  let q = {w=qw, x=qx, y=qy, z=qz}
  let s = {x=sx, y=sy, z=sz}
  let xyz = {x, y, z}
 
  let (c, rad) = render.conic cam_params fovs V q s xyz
  in (c.a, c.b, c.c, rad)

-- ==
-- entry: test_precompute_color
-- input {
--   1.0f32 0.0f32 0.0f32 0.0f32 0.0f32 1.0f32 5.0f32
--   [0.0f32] [0.0f32] [0.0f32]
--   [0.5f32] [0.5f32] [0.5f32]
--   [[0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32]]
--   [[0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32]]
--   [[0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32]]
-- }
-- output { [0.640986f32] [0.640986f32] [0.640986f32] }
entry test_precompute_color (qw: f32) (qx: f32) (qy: f32) (qz: f32) (tx: f32) (ty: f32) (tz: f32)
                            (mx: []f32) (my: []f32) (mz: []f32)
                            (cr: []f32) (cg: []f32) (cb: []f32)
                            (shr: [][]f32) (shg: [][]f32) (shb: [][]f32) =
  let cam_q = {w=qw, x=qx, y=qy, z=qz}
  let cam_t = [tx, ty, tz]
  let means = map3 (\x y z -> {x, y, z}) mx my mz
  let rgbs = map3 (\r g b -> {r, g, b}) cr cg cb
  let shs_raw = map3 (\sr sg sb -> map3 (\r g b -> {r,g,b}) sr sg sb) shr shg shb
  let shs = shs_raw :> [1]sh [3]

  let res = spherical_harmonics.precompute_color cam_q cam_t means rgbs shs
  in (map (.r) res, map (.g) res, map (.b) res)

-- ==
-- entry: test_sort_splats
-- input { 16i64 2i64 10.0f32 }
-- output { [0i64, 2i64] [0i64, 1i64, 2i64] }
entry test_sort_splats (num_tiles: i64) (gid: i64) (z: f32) =
  -- Create two splats: one in tile 1 (closer), one in tile 0 (farther)
  -- splat 0: tile 1, depth 1.0 (closer)
  -- splat 1: tile 0, depth 10.0 (farther, defined by input z)
  let k0 = utils.sort_key 1 1.0
  let k1 = utils.sort_key 0 z
  let splats = [{tid=1, key=k0, gid=gid}, {tid=0, key=k1, gid=0}]
 
  let (sorted, ranges) = render.sort_splats num_tiles splats
  -- Expected sort: tile 0 comes first (tid=0), then tile 1 (tid=1)
  -- ranges should reflect counts.
  -- If num_tiles=16, ranges[0] should be end of tile 0 (1), ranges[1] end of tile 1 (2).
  -- returning gids to show order, and first few ranges
  in (map (.gid) sorted, ranges[:3])

-- ==
-- entry: test_raster_image
-- input { 16i64 16i64 }
-- output { 1.0f32 0.0f32 0.0f32 }
entry test_raster_image (W: i64) (H: i64) =
  -- Setup a simple scene with 1 gaussian covering the top-left
  let image_size = (W, H)
  -- 2x2 grid since BLOCK is 16
  let ranges = replicate (2*2) 0i64
  let splats = [{tid=0, key=0u64, gid=0}]
  let gaussians = [{m={u=0.0, v=0.0}, c={a=1.0, b=0.0, c=1.0}, o=1.0, r={r=1.0, g=0.0, b=0.0}}]

  let image = render.raster_image image_size gaussians splats ranges
  let p = image[0][0]
  in (p.r, p.g, p.b)
