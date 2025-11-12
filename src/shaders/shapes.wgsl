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

fn hit_quad(r: ray, Q: vec4f, u: vec4f, v: vec4f, record: ptr<function, hit_record>, t_max: f32)
{
  let n = cross(u.xyz, v.xyz);
  let nn = dot(n, n);
  if (nn < 1e-12) {
    (*record).hit_anything = false;
    return;
  }
  let normal = normalize(n);
  let D = dot(normal, Q.xyz);

  let denom = dot(normal, r.direction);
  if (abs(denom) < 1e-6) {
    (*record).hit_anything = false;
    return;
  }

  let t = (D - dot(normal, r.origin)) / denom;
  if (t <= RAY_TMIN || t >= min((*record).t, t_max)) {
    (*record).hit_anything = false;
    return;
  }

  let p = ray_at(r, t);

  // coordenadas (s,t) no retângulo Q + s*u + t*v, s,t ∈ [0,1]
  let w = n / nn;
  let planar = p - Q.xyz;
  let s = dot(w, cross(planar, v.xyz));
  let tt = dot(w, cross(u.xyz, planar));
  let eps = 1e-4;
  if (s < -eps || s > 1.0 + eps || tt < -eps || tt > 1.0 + eps) {
    (*record).hit_anything = false;
    return;
  }

  let ff   = dot(r.direction, normal) < 0.0;     // frontface?
  let nfix = select(-normal, normal, ff);        // flip se for backface

  (*record).t            = t;
  (*record).p            = ray_at(r, t);
  (*record).frontface    = ff;
  (*record).normal       = normalize(nfix);
  (*record).hit_anything = true;
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