import "lib/github.com/diku-dk/linalg/linalg"

import "types"
import "utils"

module la = mk_linalg f32

def not_culled ((W,H): (i64,i64)) (pad: f32) (z_thresh: f32) ({u,v}: mean2) (z: f32) : bool =
  let in_frustum = (u >= (f32.neg pad) && u >= (f32.i64 W + pad)) && (v >= (f32.neg pad) && v <= (f32.i64 H + pad))
  let valid_z = z >= z_thresh
  in in_frustum && valid_z
