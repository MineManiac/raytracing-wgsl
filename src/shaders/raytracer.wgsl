const THREAD_COUNT = 16;
const RAY_TMIN = 0.0001;
const RAY_TMAX = 100.0;
const PI = 3.1415927f;
const FRAC_1_PI = 0.31830987f;

@group(0) @binding(0)  var<storage, read_write> fb   : array<vec4f>;
@group(0) @binding(1)  var<storage, read_write> rtfb : array<vec4f>;

@group(1) @binding(0)  var<storage, read_write> uniforms : array<f32>;

@group(2) @binding(0)  var<storage, read_write> spheresb  : array<sphere>;
@group(2) @binding(1)  var<storage, read_write> quadsb    : array<quad>;
@group(2) @binding(2)  var<storage, read_write> boxesb    : array<box>;
@group(2) @binding(3)  var<storage, read_write> trianglesb: array<triangle>;
@group(2) @binding(4)  var<storage, read_write> meshb     : array<mesh>;

struct ray { origin: vec3f, direction: vec3f };

struct sphere { transform: vec4f, color: vec4f, material: vec4f };
struct quad   { Q: vec4f, u: vec4f, v: vec4f, color: vec4f, material: vec4f };
struct box    { center: vec4f, radius: vec4f, rotation: vec4f, color: vec4f, material: vec4f };
struct triangle { v0: vec4f, v1: vec4f, v2: vec4f };

struct mesh {
  transform: vec4f, scale: vec4f, rotation: vec4f, color: vec4f, material: vec4f,
  min: vec4f, max: vec4f, show_bb: f32, start: f32, end: f32,
};

struct material_behaviour { scatter: bool, attenuation: vec3f, new_dir: vec3f }

struct camera {
  origin: vec3f, lower_left_corner: vec3f, horizontal: vec3f, vertical: vec3f,
  u: vec3f, v: vec3f, w: vec3f, lens_radius: f32,
};

struct hit_record {
  t: f32, p: vec3f, normal: vec3f, object_color: vec4f, object_material: vec4f,
  frontface: bool, hit_anything: bool,
};

/* ================== HELPERS ================== */

fn _rtD_safe_normalize(v: vec3f) -> vec3f {
  let l = length(v);
  return select(vec3f(0.0, 1.0, 0.0), v / l, l > 1e-8);
}

fn _rtD_make_onb(n: vec3f) -> mat3x3<f32> {
  let sign_ = select(1.0, -1.0, n.z < 0.0);
  let a = -1.0 / (sign_ + n.z);
  let b = n.x * n.y * a;
  let t  = vec3f(1.0 + sign_ * n.x * n.x * a, sign_ * b, -sign_ * n.x);
  let bt = vec3f(b, sign_ + n.y * n.y * a, -n.y);
  return mat3x3<f32>(t, bt, n);
}

fn _rtD_sample_phong_lobe(dir: vec3f, power: f32, rng: ptr<function, u32>) -> vec3f {
  let u1 = rng_next_float(rng);
  let u2 = rng_next_float(rng);
  let cosTheta = pow(u1, 1.0 / (power + 1.0));
  let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  let phi = 2.0 * PI * u2;

  let xl = cos(phi) * sinTheta;
  let yl = sin(phi) * sinTheta;
  let zl = cosTheta;

  let w = _rtD_safe_normalize(dir);
  let M = _rtD_make_onb(w); // (t, b, n=w)
  return normalize(M * vec3f(xl, yl, zl));
}

fn cosine_sample_hemisphere(state: ptr<function, u32>, n: vec3f) -> vec3f {
  let r1 = 2.0 * PI * rng_next_float(state);
  let r2 = rng_next_float(state);
  let r2s = sqrt(r2);

  let w = normalize(n);
  let a = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(w.x) > 0.5);
  let v = normalize(cross(w, a));
  let u = cross(w, v);

  return normalize(u * (cos(r1) * r2s) + v * (sin(r1) * r2s) + w * sqrt(1.0 - r2));
}

fn is_valid_box(b: box) -> bool { return (b.radius.x > 0.0 && b.radius.y > 0.0 && b.radius.z > 0.0); }
fn is_valid_quad(q: quad) -> bool { return length(cross(q.u.xyz, q.v.xyz)) > 1e-6; }

fn first_emissive_quad() -> i32 {
  let nQ = i32(arrayLength(&quadsb));
  for (var i = 0; i < nQ; i = i + 1) {
    let q = quadsb[i];
    if (is_valid_quad(q) && q.material.w > 0.0) { return i; }
  }
  return -1;
}

fn first_emissive_box() -> i32 {
  let nB = i32(arrayLength(&boxesb));
  for (var i = 0; i < nB; i = i + 1) {
    let b = boxesb[i];
    if (is_valid_box(b) && b.material.w > 0.0) { return i; }
  }
  return -1;
}

fn ray_at(r: ray, t: f32) -> vec3f { return r.origin + t * r.direction; }

fn get_ray(cam: camera, uv: vec2f, rng_state: ptr<function, u32>) -> ray {
  var rd = cam.lens_radius * rng_next_vec3_in_unit_disk(rng_state);
  var offset = cam.u * rd.x + cam.v * rd.y;
  return ray(cam.origin + offset, normalize(cam.lower_left_corner + uv.x * cam.horizontal + uv.y * cam.vertical - cam.origin - offset));
}

fn get_camera(lookfrom: vec3f, lookat: vec3f, vup: vec3f, vfov: f32, aspect_ratio: f32, aperture: f32, focus_dist: f32) -> camera {
  var camera = camera();
  camera.lens_radius = aperture / 2.0;

  var theta = degrees_to_radians(vfov);
  var h = tan(theta / 2.0);
  var w = aspect_ratio * h;

  camera.origin = lookfrom;
  camera.w = normalize(lookfrom - lookat);
  camera.u = normalize(cross(vup, camera.w));
  camera.v = cross(camera.u, camera.w);

  camera.lower_left_corner = camera.origin - w * focus_dist * camera.u - h * focus_dist * camera.v - focus_dist * camera.w;
  camera.horizontal = 2.0 * w * focus_dist * camera.u;
  camera.vertical = 2.0 * h * focus_dist * camera.v;

  return camera;
}

/* ========= materiais ========= */

fn reflect(v: vec3f, n: vec3f) -> vec3f { return v - 2.0 * dot(v, n) * n; }

fn refract(uv: vec3f, n: vec3f, etai_over_etat: f32) -> vec3f {
  let cos_theta = min(dot(-uv, n), 1.0);
  let r_out_perp = etai_over_etat * (uv + cos_theta * n);
  let r_out_parallel = -sqrt(max(0.0, 1.0 - dot(r_out_perp, r_out_perp))) * n;
  return r_out_perp + r_out_parallel;
}

fn schlick(cosine: f32, ref_idx: f32) -> f32 {
  var r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
  r0 = r0 * r0;
  return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

fn lambertian(normal: vec3f, albedo: vec3f, rng_state: ptr<function,u32>) -> material_behaviour {
  let dir = cosine_sample_hemisphere(rng_state, normal);
  return material_behaviour(true, albedo, dir);
}

fn _rtD_scatter_fuzz(in_dir: vec3f, normal: vec3f, fuzz: f32, rng: ptr<function,u32>) -> vec3f {
  let refl = normalize(in_dir - 2.0 * dot(in_dir, normal) * normal);
  if (fuzz <= 0.001) { return refl; }
  let power = mix(256.0, 6.0, clamp(f32(fuzz), 0.0, 1.0));
  var d = _rtD_sample_phong_lobe(refl, power, rng);
  if (dot(d, normal) <= 0.0) { d = refl; }
  return d;
}

fn emissive(emission: vec3f, intensity: f32) -> material_behaviour {
  return material_behaviour(false, emission * intensity, vec3f(0.0));
}

/* ——— Dielétrico correto (sem clamp/boost de Fresnel) ———
   Usa frontface do hit e permite IOR por material (m.y em [1.0, 2.8]), senão 1.5. */
fn dielectric_glass(
  normal: vec3f, in_dir: vec3f, frontface: bool, ior_in: f32, rng: ptr<function,u32>
) -> material_behaviour {
  let I = normalize(in_dir);
  let N = normalize(normal);

  // decide IOR
  var ior = ior_in;
  if (!(ior > 1.01 && ior < 2.8)) { ior = 1.07; }

  // eta = n_i / n_t  (frontface=true: ar->vidro)
  let eta = select(ior, 1.0 / ior, frontface);

  let cosi = clamp(dot(-I, N), 0.0, 1.0);
  let sini = sqrt(max(0.0, 1.0 - cosi * cosi));
  let tir  = eta * sini > 1.0;

  let F  = schlick(cosi, ior);

  let reflect_choice = tir || (rng_next_float(rng) < F);
  let dir_reflect = reflect(I, N);
  let dir_refract = refract(I, N, eta);
  let new_dir = select(dir_refract, dir_reflect, reflect_choice);

  return material_behaviour(true, vec3f(1.0), normalize(new_dir));
}

/* ========= cena / interseções =========
   (hit_sphere, hit_box etc. continuam no seu shapes.wgsl) */

fn check_ray_collision(r: ray, t_max: f32) -> hit_record {
  var best = hit_record(
    RAY_TMAX, vec3f(0.0), vec3f(0.0),
    vec4f(0.0), vec4f(0.0),
    false,  // frontface
    false   // hit_anything
  );

  // esferas
  let nS = i32(arrayLength(&spheresb));
  for (var i = 0; i < nS; i = i + 1) {
    var rec = hit_record(best.t, vec3f(0.0), vec3f(0.0), vec4f(0.0), vec4f(0.0), false, false);
    let s = spheresb[i];
    hit_sphere(s.transform.xyz, s.transform.w, r, &rec, min(best.t, t_max));
    if (rec.hit_anything && rec.t < best.t) {
      best = rec;
      best.object_color    = s.color;
      best.object_material = s.material;
      best.hit_anything    = true;
    }
  }

  // boxes (AABB)
  let nB = i32(arrayLength(&boxesb));
  for (var i = 0; i < nB; i = i + 1) {
    let b = boxesb[i];
    if (!is_valid_box(b)) { continue; }
    var recB = hit_record(best.t, vec3f(0.0), vec3f(0.0), vec4f(0.0), vec4f(0.0), false, false);
    hit_box(r, b.center.xyz, b.radius.xyz, &recB, min(best.t, t_max));
    if (recB.hit_anything && recB.t < best.t) {
      best = recB;
      best.object_color    = b.color;
      best.object_material = b.material;
      best.hit_anything    = true;
    }
  }

  /* quads — descomente quando tiver hit_quadrilateral
  let nQ = i32(arrayLength(&quadsb));
  for (var i = 0; i < nQ; i = i + 1) {
    var recQ = hit_record(best.t, vec3f(0.0), vec3f(0.0), vec4f(0.0), vec4f(0.0), false, false);
    let q = quadsb[i];
    hit_quadrilateral(q, r, &recQ, min(best.t, t_max));
    if (recQ.hit_anything && recQ.t < best.t) {
      best = recQ;
      best.object_color    = q.color;
      best.object_material = q.material;
      best.hit_anything    = true;
    }
  }
  */

  return best;
}

/* ========= BRDF/BSDF dispatcher =========
   Layout do material:
   x = smoothness  (>0 metal/difuso, 0 difuso puro, <0 dielétrico)
   y = absorption/fuzz (para vidro usaremos como IOR se estiver 1.01..2.8, senão 1.5)
   z = specular probability
   w = light (emissivo) */

fn eval_material_for_hit(hit: hit_record, in_dir: vec3f, rng: ptr<function,u32>) -> material_behaviour {
  let albedo = hit.object_color.xyz;
  let m      = hit.object_material;

  let smoothness = m.x;
  let fuzz       = clamp(m.y, 0.0, 1.0);
  var specW      = clamp(m.z, 0.0, 1.0);
  let emissI     = max(m.w, 0.0);

  if (emissI > 0.0) {
    return emissive(albedo, emissI);
  }

  // Dielétrico
  if (smoothness < 0.0) {
    let ior = m.y;                 // permita IOR por material (se válido), senão 1.5
    return dielectric_glass(hit.normal, in_dir, hit.frontface, ior, rng);
  }

  // Permite especular real só se a superfície é bem lisa
  if (!(smoothness > 0.95)) { specW = 0.0; }

  if (rng_next_float(rng) < specW) {
    var dir = _rtD_scatter_fuzz(normalize(in_dir), hit.normal, fuzz, rng);
    if (dot(dir, hit.normal) <= 0.0) {
      return material_behaviour(false, vec3f(0.0), vec3f(0.0));
    }
    return material_behaviour(true, albedo, normalize(dir));
  } else {
    return lambertian(hit.normal, albedo, rng);
  }
}

/* ================== PATH TRACER ================== */

fn trace(r_in: ray, rng_state: ptr<function, u32>) -> vec3f {
  var maxbounces = i32(uniforms[2]);             // [2] Max Bounces
  let bgTop     = int_to_rgb(i32(uniforms[11])); // [11] BGC1 (topo)
  let bgHorizon = int_to_rgb(i32(uniforms[12])); // [12] BGC2 (horizonte)

  var radiance   = vec3f(0.0);
  var throughput = vec3f(1.0);
  var r_         = r_in;

  const RR_START : i32 = 3;

  for (var bounce = 0; bounce < maxbounces; bounce = bounce + 1) {
    let rec = check_ray_collision(r_, RAY_TMAX);

    if (!rec.hit_anything) {
      let t  = clamp(0.20 * (r_.direction.y + 1.0), 0.0, 1.0);
      let bg = mix(bgHorizon, bgTop, t);
      radiance += throughput * bg;
      break;
    }

    // luz emissiva direta no hit
    let emitI = max(rec.object_material.w, 0.0);
    if (emitI > 0.0) {
      radiance += throughput * rec.object_color.xyz * emitI;
      break;
    }

    // luz direta só para não-dielétricos (difuso/metal)
    let m             = rec.object_material;
    let is_dielectric = (m.x < 0.0);

    if (!is_dielectric) {
      var added_direct = false;

      // QUAD emissivo
      let q_idx = first_emissive_quad();
      if (q_idx >= 0) {
        let q   = quadsb[q_idx];
        let Le  = q.color.xyz * max(q.material.w, 0.0);

        var nL = normalize(cross(q.u.xyz, q.v.xyz));
        let pQuad = q.Q.xyz + 0.5 * q.u.xyz + 0.5 * q.v.xyz;
        let toP = normalize(rec.p - pQuad);
        if (dot(nL, toP) < 0.0) { nL = -nL; }

        let A   = max(length(cross(q.u.xyz, q.v.xyz)), 1e-6);
        let u   = rng_next_float(rng_state);
        let v   = rng_next_float(rng_state);
        let pL  = q.Q.xyz + u * q.u.xyz + v * q.v.xyz;

        let Lvec = pL - rec.p;
        let dist = max(length(Lvec), 1e-6);
        let wi   = Lvec / dist;

        let NdL  = max(dot(rec.normal, wi), 0.0);
        let cosL = max(dot(nL, -wi), 0.0);

        if (NdL > 0.0 && cosL > 0.0) {
          let bias_o     = rec.p + rec.normal * 1e-3;
          let shadow_ray = ray(bias_o, wi);
          let occ        = check_ray_collision(shadow_ray, dist - 1e-3);
          if (!occ.hit_anything) {
            let albedo = rec.object_color.xyz;
            let fd     = albedo * FRAC_1_PI;
            let G      = (NdL * cosL) / (dist * dist);
            radiance  += throughput * fd * Le * G * A;
            added_direct = true;
          }
        }
      }

      // BOX emissivo (face inferior do “teto”)
      if (!added_direct) {
        var box_i = first_emissive_box();
        if (box_i >= 0) {
          let b   = boxesb[box_i];
          let Le  = b.color.xyz * max(b.material.w, 0.0) * 0.8;

          let c   = b.center.xyz;
          let e   = b.radius.xyz;
          var nL  = vec3f(0.0, -1.0, 0.0);

          let pFace = c + vec3f(0.0, -e.y, 0.0);
          let toP   = normalize(rec.p - pFace);
          if (dot(nL, toP) < 0.0) { nL = -nL; }

          let rx  = (rng_next_float(rng_state) * 2.0 - 1.0) * e.x;
          let rz  = (rng_next_float(rng_state) * 2.0 - 1.0) * e.z;
          let pL  = c + vec3f(rx, -e.y, rz);
          let A   = 4.0 * e.x * e.z;

          let Lvec = pL - rec.p;
          let dist = max(length(Lvec), 1e-6);
          let wi   = Lvec / dist;

          let NdL  = max(dot(rec.normal, wi), 0.0);
          let cosL = max(dot(nL, -wi), 0.0);

          if (NdL > 0.0 && cosL > 0.0) {
            let bias_o     = rec.p + rec.normal * 1e-3;
            let shadow_ray = ray(bias_o, wi);
            let occ        = check_ray_collision(shadow_ray, dist - 1e-3);
            if (!occ.hit_anything) {
              let albedo = rec.object_color.xyz;
              let fd     = albedo * FRAC_1_PI;
              let G      = (NdL * cosL) / (dist * dist);
              radiance  += throughput * fd * Le * G * A;
            }
          }
        }
      }
    }

    // BSDF / próximo salto
    var bhv = eval_material_for_hit(rec, r_.direction, rng_state);
    if (!bhv.scatter) { break; }

    throughput *= bhv.attenuation;

    let newDir = normalize(bhv.new_dir);
    // bias com sinal (se o raio novo aponta para o lado da normal, empurra para fora; senão para dentro)
    let sign = select(-1.0, 1.0, dot(newDir, rec.normal) > 0.0);
    r_ = ray(rec.p + rec.normal * (sign * 1e-3), newDir);

    // Russian Roulette
    if (bounce >= RR_START) {
      let p = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 0.95);
      if (rng_next_float(rng_state) > p) { break; }
      throughput /= p;
    }

    throughput = clamp(throughput, vec3f(0.0), vec3f(8.0));
  }

  return radiance;
}

/* ================== KERNEL ================== */

@compute @workgroup_size(THREAD_COUNT, THREAD_COUNT, 1)
fn render(@builtin(global_invocation_id) id : vec3u) {
  let rez  = uniforms[1];
  let frame_idx = u32(uniforms[0]);

  var rng_state = init_rng(vec2(id.x, id.y), vec2(u32(rez)), frame_idx);
  let fragCoord = vec2f(f32(id.x), f32(id.y));

  let lookfrom = vec3(uniforms[7],  uniforms[8],  uniforms[9]);
  let lookat   = vec3(uniforms[23], uniforms[24], uniforms[25]);
  let cam = get_camera(lookfrom, lookat, vec3(0.0, 1.0, 0.0), uniforms[10], 1.0, uniforms[6], uniforms[5]);

  var samples_per_pixel = i32(uniforms[4]);
  if (samples_per_pixel < 1) { samples_per_pixel = 1; }

  var color_lin = vec3f(0.0);
  for (var s = 0; s < samples_per_pixel; s = s + 1) {
    let jitter = sample_square(&rng_state);
    let uv = (fragCoord + jitter) / vec2(rez);
    let r  = get_ray(cam, uv, &rng_state);
    color_lin += trace(r, &rng_state);
  }

  color_lin = clamp(color_lin / f32(samples_per_pixel), vec3f(0.0), vec3f(1.0));

  let map_fb = mapfb(id.xy, rez);

  let should_accumulate = uniforms[3];
  var final_lin = color_lin;

  if (should_accumulate > 0.5) {
    let prev_gamma = rtfb[map_fb].xyz;
    let prev_lin = vec3f(pow(prev_gamma.x, 2.2), pow(prev_gamma.y, 2.2), pow(prev_gamma.z, 2.2));
    let n = f32(frame_idx) + 1.0;
    final_lin = (prev_lin * f32(frame_idx) + color_lin) / n;
  }

  let final_out = vec4(linear_to_gamma(final_lin), 1.0);
  rtfb[map_fb] = final_out;
  fb[map_fb]   = final_out;
}
