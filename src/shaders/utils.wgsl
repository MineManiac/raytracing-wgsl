
fn linear_to_gamma_channel(channel: f32) -> f32
{
  return pow(channel, 0.4545);
}

fn linear_to_gamma(color: vec3f) -> vec3f
{
  return vec3f(linear_to_gamma_channel(color.x), linear_to_gamma_channel(color.y), linear_to_gamma_channel(color.z));
}

fn degrees_to_radians(degrees: f32) -> f32
{
  return degrees * PI / 180.0;
}

fn mapfb(p: vec2u, rez: f32) -> i32
{
  return i32(p.x) + i32(p.y) * i32(rez);
}

fn int_to_rgb(c: i32) -> vec3f
{
  var r = f32((c >> 16) & 0xff) / 255.0;
  var g = f32((c >> 8) & 0xff) / 255.0;
  var b = f32(c & 0xff) / 255.0;

  return vec3f(r, g, b);
}

fn qmul(q1: vec4f, q2: vec4f) -> vec4f
{
  return vec4f(
    q2.xyz * q1.w + q1.xyz * q2.w + cross(q1.xyz, q2.xyz),
    q1.w * q2.w - dot(q1.xyz, q2.xyz)
  );
}

fn q_conjugate(q: vec4f) -> vec4f
{
  return vec4f(-q.xyz, q.w);
}

fn rotate_vector(v: vec3f, r: vec4f) -> vec3f
{
  var r_c = r * vec4f(-1.0, -1.0, -1.0, 1.0);
  return qmul(r, qmul(vec4f(v, 0.0), r_c)).xyz;
}