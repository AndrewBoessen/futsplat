import "engine/types"
import "engine/render"

-- Helper to reconstruct RGB from 3 floats
def mk_rgb r g b : rgb = {r, g, b}

entry render [n]
           (W: i64) (H: i64)
           -- Camera Intrinsics (pinhole)
           (cp_fx: f32) (cp_fy: f32) (cp_cx: f32) (cp_cy: f32)
           -- Camera Rotation (quat)
           (cq_w: f32) (cq_x: f32) (cq_y: f32) (cq_z: f32)
           -- Camera Translation (trans is [3]f32, which is fine, but safer to be explicit)
           (ct_x: f32) (ct_y: f32) (ct_z: f32)
           -- Means (mean3) -> Structure of Arrays
           (xyz_x: [n]f32) (xyz_y: [n]f32) (xyz_z: [n]f32)
           -- Opacities (opa is f32, so this is already fine)
           (opas: [n]f32)
           -- Scales (scale)
           (s_x: [n]f32) (s_y: [n]f32) (s_z: [n]f32)
           -- Quaternions (quat)
           (rot_w: [n]f32) (rot_x: [n]f32) (rot_y: [n]f32) (rot_z: [n]f32)
           -- RGBs (rgb)
           (c_r: [n]f32) (c_g: [n]f32) (c_b: [n]f32)
           -- Spherical Harmonics
           -- sh is [15]rgb (for degree 3). We split this into 3 arrays of shape [n][15]
           (sh_r: [n][15]f32) (sh_g: [n][15]f32) (sh_b: [n][15]f32)
           : ([][]f32, [][]f32, [][]f32) =
  -- Reconstruct Records
  let image_size = (W, H)
  let cam_params : pinhole = {fx=cp_fx, fy=cp_fy, cx=cp_cx, cy=cp_cy}
  let cam_q : quat = {w=cq_w, x=cq_x, y=cq_y, z=cq_z}
  let cam_t : trans = [ct_x, ct_y, ct_z]

  -- Reconstruct Arrays of Records
  let xyzs = map3 (\x y z -> {x, y, z}) xyz_x xyz_y xyz_z
  let scales = map3 (\x y z -> {x, y, z}) s_x s_y s_z
  let quats = map4 (\w x y z -> {w, x, y, z}) rot_w rot_x rot_y rot_z
  let rgbs = map3 mk_rgb c_r c_g c_b

  -- Reconstruct SHs
  -- We zip the 3 arrays of floats back into 1 array of RGB structs
  -- Input: [n][15]f32 -> Output: [n][15]rgb
  let shs_raw = map3 (\sr sg sb -> map3 mk_rgb sr sg sb) sh_r sh_g sh_b
  -- This makes the type match the expected argument for preprocess
  let shs = shs_raw :> [n]sh [3]

  -- Run the Pipeline
  let image = preprocess image_size cam_params cam_q cam_t xyzs opas scales quats rgbs shs
              |> (\(g, s, r) -> raster_image image_size g s r)

  -- Flatten Output
  let rs = map (map (.r)) image
  let gs = map (map (.g)) image
  let bs = map (map (.b)) image
  in (rs, gs, bs)
