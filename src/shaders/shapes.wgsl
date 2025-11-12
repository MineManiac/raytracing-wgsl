fn hit_sphere(center: vec3f, radius: f32, r: ray, rec: ptr<function, hit_record>, t_max: f32) {
  // Solve (o + t*d - c)^2 = R^2
  let oc = r.origin - center;
  let a  = dot(r.direction, r.direction);
  let half_b = dot(oc, r.direction);
  let c  = dot(oc, oc) - radius * radius;
  let disc = half_b * half_b - a * c;

  if (disc < 0.0) {
    (*rec).hit_anything = false;
    return;
  }

  let sqrtD = sqrt(disc);

  // menor t válido
  var t = (-half_b - sqrtD) / a;
  if (t < RAY_TMIN || t > t_max) {
    t = (-half_b + sqrtD) / a;
    if (t < RAY_TMIN || t > t_max) {
      (*rec).hit_anything = false;
      return;
    }
  }

  let p = r.origin + t * r.direction;
  let outward = normalize((p - center) / radius);

  let frontface = dot(r.direction, outward) < 0.0;
  // guarde o sinal e oriente a normal para fora (faceforward)
  (*rec).frontface = frontface;
  (*rec).normal    = select(-outward, outward, frontface);
  (*rec).p         = p;
  (*rec).t         = t;
  (*rec).hit_anything = true;
}

fn hit_quad(r: ray, Q: vec4f, u: vec4f, v: vec4f, record: ptr<function, hit_record>, max: f32)
{
  var n = cross(u.xyz, v.xyz);
  var normal = normalize(n);
  var D = dot(normal, Q.xyz);
  var w = n / dot(n.xyz, n.xyz);

  var denom = dot(normal, r.direction);
  if (abs(denom) < 0.0001)
  {
    record.hit_anything = false;
    return;
  }

  var t = (D - dot(normal, r.origin)) / denom;
  if (t < RAY_TMIN || t > max)
  {
    record.hit_anything = false;
    return;
  }

  var intersection = ray_at(r, t);
  var planar_hitpt_vector = intersection - Q.xyz;
  var alpha = dot(w, cross(planar_hitpt_vector, v.xyz));
  var beta = dot(w, cross(u.xyz, planar_hitpt_vector));

  if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0)
  {
    record.hit_anything = false;
    return;
  }

  if (dot(normal, r.direction) > 0.0)
  {
    record.hit_anything = false;
    return;
  }

  record.t = t;
  record.p = intersection;
  record.normal = normal;
  record.hit_anything = true;
}

fn hit_triangle(r: ray, v0: vec3f, v1: vec3f, v2: vec3f, record: ptr<function, hit_record>, max: f32)
{
  var v1v0 = v1 - v0;
  var v2v0 = v2 - v0;
  var rov0 = r.origin - v0;

  var n = cross(v1v0, v2v0);
  var q = cross(rov0, r.direction);

  var d = 1.0 / dot(r.direction, n);

  var u = d * dot(-q, v2v0);
  var v = d * dot(q, v1v0);
  var t = d * dot(-n, rov0);

  if (u < 0.0 || u > 1.0 || v < 0.0 || (u + v) > 1.0)
  {
    record.hit_anything = false;
    return;
  }

  if (t < RAY_TMIN || t > max)
  {
    record.hit_anything = false;
    return;
  }

  record.t = t;
  record.p = ray_at(r, t);
  record.normal = normalize(n);
  record.hit_anything = true;
}

fn hit_box(r: ray, center: vec3f, rad: vec3f, record: ptr<function, hit_record>, t_max: f32)
{
  // ignora box "vazio" (slots não usados no buffer)
  if (rad.x <= 0.0 || rad.y <= 0.0 || rad.z <= 0.0) {
    (*record).hit_anything = false;
    return;
  }

  // Interseção AABB "slabs"
  let invD = 1.0 / r.direction;
  let n    = invD * (r.origin - center);
  let k    = abs(invD) * rad;

  let t1 = -n - k;   // entrada
  let t2 = -n + k;   // saída

  let tN = max(max(t1.x, t1.y), t1.z);
  let tF = min(min(t2.x, t2.y), t2.z);

  if (tN > tF || tF < 0.0) {
    (*record).hit_anything = false;
    return;
  }

  var t = tN;
  if (t < RAY_TMIN || t > t_max) {
    (*record).hit_anything = false;
    return;
  }

  // ponto e normal da face atingida
  let p = r.origin + t * r.direction;

  let dx = abs(t - t1.x);
  let dy = abs(t - t1.y);
  let dz = abs(t - t1.z);

  var outward = vec3f(0.0);
  if (dx <= dy && dx <= dz) {
    outward = vec3f(-sign(r.direction.x), 0.0, 0.0);
  } else if (dy <= dx && dy <= dz) {
    outward = vec3f(0.0, -sign(r.direction.y), 0.0);
  } else {
    outward = vec3f(0.0, 0.0, -sign(r.direction.z));
  }

  let ff   = dot(r.direction, outward) < 0.0;
  let nfix = select(-outward, outward, ff);

  (*record).t            = t;
  (*record).p            = p;
  (*record).frontface    = ff;
  (*record).normal       = normalize(nfix);
  (*record).hit_anything = true;
}