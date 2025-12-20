-- Gaussian Parameters
type mean3 = {x: f32, y: f32, z:f32}
type opa = f32
type scale = {x: f32, y: f32, z:f32}
type quat = {w: f32, x: f32, y: f32, z:f32}
type rgb = {r: f32, g: f32, b:f32}
type sh [n] = [(n + 1) * (n + 1) - 1][3]f32

-- Intermediate Values
type mean2 = {u: f32, v: f32}
type conic = {a: f32, b: f32, c: f32}

-- Projection Matrices
type view = [4][4]f32
type proj = [4][4]f32
type rot = quat
type trans = {x: f32, y: f32, z:f32}

-- Camera Parameters
type pinhole = {fx: f32, fy: f32, cx: f32, cy: f32}
type simple_pinhole = {f: f32, cx: f32, cy: f32}

-- Image
type image [m][n] = [m][n]f32
type tile = [16][16]f32
type splat = {tid: i64, z: f32, gid: i64}
