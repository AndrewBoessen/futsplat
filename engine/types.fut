-- Gaussian Parameters
type mean3 = {x: f32, y: f32, z:f32}
type opa = f32
type scale = {x: f32, y: f32, z:f32}
type quat = {w: f32, x: f32, y: f32, z:f32}
type rgb = {r: f32, g: f32, b:f32}
type sh [L] = [(L + 1) * (L + 1) - 1]rgb

-- Intermediate Values
type mean2 = {u: f32, v: f32}
type conic = {a: f32, b: f32, c: f32}
type gaussian = {m: mean2, c: conic, o: opa, r: rgb}

-- Projection Matrices
type view = [3][4]f32
type proj = [4][4]f32
type rot = [3][3]f32
type trans = [3]f32

-- Camera Parameters
type pinhole = {fx: f32, fy: f32, cx: f32, cy: f32}

-- Image
let TILE_SIZE: i64 = 16
type image [m][n] = [m][n]f32
type tile = [TILE_SIZE][TILE_SIZE]f32
type splat = {tid: i64, key: u64, gid: i64}

-- Spherical Harmonics Constants
let SH_L0: f32 = 0.28209479177387814
let SH_L1: f32 = 0.4886025119029199
let SH_L2: [5]f32 = [
	1.0925484305920792,
	-1.0925484305920792,
	0.31539156525252005,
	-1.0925484305920792,
	0.5462742152960396
]
let SH_L3: [7]f32 = [
	-0.5900435899266435,
	2.890611442640554,
	-0.4570457994644658,
	0.3731763325901154,
	-0.4570457994644658,
	1.445305721320277,
	-0.5900435899266435
]
