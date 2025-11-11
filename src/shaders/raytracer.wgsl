const THREAD_COUNT = 16;
const RAY_TMIN = 0.0001;
const RAY_TMAX = 100.0;
const PI = 3.1415927f;
const FRAC_1_PI = 0.31830987f;
const FRAC_2_PI = 1.5707964f;

@group(0) @binding(0)  
  var<storage, read_write> fb : array<vec4f>;

@group(0) @binding(1)
  var<storage, read_write> rtfb : array<vec4f>;

@group(1) @binding(0)
  var<storage, read_write> uniforms : array<f32>;

@group(2) @binding(0)
  var<storage, read_write> spheresb : array<sphere>;

@group(2) @binding(1)
  var<storage, read_write> quadsb : array<quad>;

@group(2) @binding(2)
  var<storage, read_write> boxesb : array<box>;

@group(2) @binding(3)
  var<storage, read_write> trianglesb : array<triangle>;

@group(2) @binding(4)
  var<storage, read_write> meshb : array<mesh>;

struct ray {
  origin : vec3f,
  direction : vec3f,
};

struct sphere {
  transform : vec4f,
  color : vec4f,
  material : vec4f,
};

struct quad {
  Q : vec4f,
  u : vec4f,
  v : vec4f,
  color : vec4f,
  material : vec4f,
};

struct box {
  center : vec4f,
  radius : vec4f,
  rotation: vec4f,
  color : vec4f,
  material : vec4f,
};

struct triangle {
  v0 : vec4f,
  v1 : vec4f,
  v2 : vec4f,
};

struct mesh {
  transform : vec4f,
  scale : vec4f,
  rotation : vec4f,
  color : vec4f,
  material : vec4f,
  min : vec4f,
  max : vec4f,
  show_bb : f32,
  start : f32,
  end : f32,
};

struct material_behaviour { 
  scatter: bool,
  attenuation: vec3f,
  new_dir: vec3f,
}

struct camera {
  origin : vec3f,
  lower_left_corner : vec3f,
  horizontal : vec3f,
  vertical : vec3f,
  u : vec3f,
  v : vec3f,
  w : vec3f,
  lens_radius : f32,
};

struct hit_record {
  t : f32,
  p : vec3f,
  normal : vec3f,
  object_color : vec4f,
  object_material : vec4f,
  frontface : bool,
  hit_anything : bool,
};

// ================== HELPERS ==================
fn _rtD_safe_normalize(v: vec3f) -> vec3f {
  let l = length(v);
  return select(vec3f(0.0, 1.0, 0.0), v / l, l > 1e-8);
}

fn _rtD_make_onb(n: vec3f) -> mat3x3<f32> {
  // Base ortonormal estável (t, b, n)
  let sign_ = select(1.0, -1.0, n.z < 0.0);
  let a = -1.0 / (sign_ + n.z);
  let b = n.x * n.y * a;
  let t  = vec3f(1.0 + sign_ * n.x * n.x * a, sign_ * b, -sign_ * n.x);
  let bt = vec3f(b, sign_ + n.y * n.y * a, -n.y);
  return mat3x3<f32>(t, bt, n);
}

// Amostragem de lobo Phong ao redor de 'dir' — usado no Fuzz (menos ruído)
fn _rtD_sample_phong_lobe(dir: vec3f, power: f32, rng: ptr<function, u32>) -> vec3f {
  let u1 = rng_next_float(rng);
  let u2 = rng_next_float(rng);
  let cosTheta = pow(u1, 1.0 / (power + 1.0));
  let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  let phi = 2.0 * 3.1415926535 * u2;

  let xl = cos(phi) * sinTheta;
  let yl = sin(phi) * sinTheta;
  let zl = cosTheta;

  let w = _rtD_safe_normalize(dir);
  let M = _rtD_make_onb(w); // colunas = (t, b, n=w)
  return normalize(M * vec3f(xl, yl, zl));
}

// Iluminação direta "limpa" para Basic (1 SPP sem ruído)
fn _rtD_basic_direct(normal: vec3f, albedo: vec3f) -> vec3f {
  let sunDir = normalize(vec3(uniforms[13], uniforms[14], uniforms[15]));
  let ndotl  = max(dot(normal, sunDir), 0.0);
  let ambient = 0.15;
  return albedo * (ambient + 0.85 * ndotl);
}

// Espelho perfeito (Metal puro)
fn _rtD_scatter_metal(in_dir: vec3f, normal: vec3f) -> vec3f {
  return normalize(in_dir - 2.0 * dot(in_dir, normal) * normal);
}

// Espelho com blur (Fuzz) — usa lobo Phong em torno da reflexão perfeita
fn _rtD_scatter_fuzz(in_dir: vec3f, normal: vec3f, fuzz: f32, rng: ptr<function, u32>) -> vec3f {
  let refl = _rtD_scatter_metal(in_dir, normal);
  if (fuzz <= 0.001) { return refl; }
  // curva afinada para reduzir ruído com mesmo SPP
  let power = mix(256.0, 6.0, clamp(fuzz, 0.0, 1.0));
  var d = _rtD_sample_phong_lobe(refl, power, rng);
  if (dot(d, normal) <= 0.0) { d = refl; } // hemisfério correto
  return d;
}

// Utilitários (usam seu RNG existente)
fn reflect(v: vec3f, n: vec3f) -> vec3f {
  return v - 2.0 * dot(v, n) * n;
}

fn cosine_sample_hemisphere(state: ptr<function, u32>, n: vec3f) -> vec3f {
  // amostragem com peso cosseno (menos ruído que rejeição pura)
  let r1 = 2.0 * PI * rng_next_float(state);
  let r2 = rng_next_float(state);
  let r2s = sqrt(r2);

  // base ortonormal para n
  let w = normalize(n);
  let a = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(w.x) > 0.5);
  let v = normalize(cross(w, a));
  let u = cross(w, v);

  // direção local
  let dir = normalize(u * (cos(r1) * r2s) + v * (sin(r1) * r2s) + w * sqrt(1.0 - r2));
  return dir;
}

fn sun_diffuse(n: vec3f) -> vec3f {
  let sunDir = normalize(vec3(uniforms[13], uniforms[14], uniforms[15]));
  let sunI   = max(uniforms[16], 0.0);
  let sunCol = clamp(int_to_rgb(i32(uniforms[17])), vec3f(0.0), vec3f(1.0));
  let ndotl  = max(dot(n, sunDir), 0.0);
  return sunCol * (sunI * ndotl);
}

fn first_emissive_quad() -> i32 {
  let nQ = i32(arrayLength(&quadsb));
  for (var i = 0; i < nQ; i = i + 1) {
    let q = quadsb[i];
    // quad válido se |u×v| > 0
    let A = length(cross(q.u.xyz, q.v.xyz));
    if (A > 1e-6 && q.material.w > 0.0) { return i; }
  }
  return -1;
}

fn first_emissive_box() -> i32 {
  let nB = i32(arrayLength(&boxesb));
  for (var i = 0; i < nB; i = i + 1) {
    let b = boxesb[i];
    // box válido se radius tem dimensão positiva
    if ((b.radius.x > 0.0) && (b.radius.y > 0.0) && (b.radius.z > 0.0) && b.material.w > 0.0) {
      return i;
    }
  }
  return -1;
}

fn is_in_sunlight(p: vec3f, n: vec3f) -> bool {
  let sunDir = normalize(vec3(uniforms[13], uniforms[14], uniforms[15]));
  // empurra um pouquinho para fora da superfície
  let o = p + n * 1e-3;
  // raio em direção ao sol
  let r = ray(o, sunDir);

  // se bater em qualquer coisa, está na sombra
  let rec = check_ray_collision(r, RAY_TMAX);
  return !rec.hit_anything;
}

fn ray_at(r: ray, t: f32) -> vec3f
{
  return r.origin + t * r.direction;
}

fn get_ray(cam: camera, uv: vec2f, rng_state: ptr<function, u32>) -> ray
{
  var rd = cam.lens_radius * rng_next_vec3_in_unit_disk(rng_state);
  var offset = cam.u * rd.x + cam.v * rd.y;
  return ray(cam.origin + offset, normalize(cam.lower_left_corner + uv.x * cam.horizontal + uv.y * cam.vertical - cam.origin - offset));
}

fn get_camera(lookfrom: vec3f, lookat: vec3f, vup: vec3f, vfov: f32, aspect_ratio: f32, aperture: f32, focus_dist: f32) -> camera
{
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

fn envoriment_color(direction: vec3f, color1: vec3f, color2: vec3f) -> vec3f {
  let d = normalize(direction);
  let t = 0.5 * (d.y + 1.0);       // igual ao gabarito
  return mix(color2, color1, t);   // color1=topo (BGC1), color2=horizonte (BGC2)
}

fn check_ray_collision(r: ray, t_max: f32) -> hit_record
{
  // Inicializa "sem hit" e t mais distante
  var best = hit_record(
    RAY_TMAX,                  // t
    vec3f(0.0),                // p
    vec3f(0.0),                // normal
    vec4f(0.0),                // object_color
    vec4f(0.0),                // object_material
    false,                     // hit_anything
    false                      // frontface
  );

  // --- Esferas ---
  let nS = i32(arrayLength(&spheresb));
  for (var i = 0; i < nS; i = i + 1) {
    var rec = hit_record(best.t, vec3f(0.0), vec3f(0.0),
                         vec4f(0.0), vec4f(0.0), false, false);

    let s = spheresb[i];
    hit_sphere(s.transform.xyz, s.transform.w, r, &rec, min(best.t, t_max));

    if (rec.hit_anything && rec.t < best.t) {
      best = rec;
      best.object_color    = s.color;
      best.object_material = s.material;
      best.hit_anything    = true;
    }
  }

  // (Se na sua etapa D você ainda não usa quad/box, pare aqui.
  // Quando avançar, chame hit_quadrilateral / hit_box no mesmo estilo.)

  // --- Boxes ---
  let nB = i32(arrayLength(&boxesb));
  for (var i = 0; i < nB; i = i + 1) {
    let b = boxesb[i];

    // ignore slots não usados (radius zero/negativo)
    if (b.radius.x <= 0.0 || b.radius.y <= 0.0 || b.radius.z <= 0.0) { continue; }

    var recB = hit_record(best.t, vec3f(0.0), vec3f(0.0),
                          vec4f(0.0), vec4f(0.0), false, false);

    hit_box(r, b.center.xyz, b.radius.xyz, &recB, min(best.t, t_max));
    if (recB.hit_anything && recB.t < best.t) {
      best = recB;
      best.object_color    = b.color;
      best.object_material = b.material;
      best.hit_anything    = true;
    }
  }

  // --- Quads (se já tiver hit_quadrilateral) ---
  /*
  let nQ = i32(arrayLength(&quadsb));
  for (var i = 0; i < nQ; i = i + 1) {
    var recQ = hit_record(best.t, vec3f(0.0), vec3f(0.0),
                          vec4f(0.0), vec4f(0.0), false, false);
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

fn lambertian(normal: vec3f, albedo: vec3f, rng_state: ptr<function,u32>) -> material_behaviour {
  let dir = cosine_sample_hemisphere(rng_state, normal);
  return material_behaviour(true, albedo, dir);
}



fn metal(normal: vec3f, in_dir: vec3f, albedo: vec3f, fuzz: f32, rng_state: ptr<function,u32>) -> material_behaviour {
  let dir = _rtD_scatter_fuzz(normalize(in_dir), normal, clamp(fuzz,0.0,1.0), rng_state);
  if (dot(dir, normal) <= 0.0) {
    return material_behaviour(false, vec3f(0.0), vec3f(0.0));
  }
  return material_behaviour(true, albedo, dir);
}

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


fn dielectric(normal: vec3f, in_dir: vec3f, ior: f32, rng: ptr<function,u32>) -> material_behaviour {
  let I    = normalize(in_dir);
  let Nraw = normalize(normal);

  let entering = dot(I, Nraw) < 0.0;
  let N = select(-Nraw, Nraw, entering);

  let eta_i = select(ior, 1.0, entering);
  let eta_t = select(1.0, ior, entering);
  let eta   = eta_i / eta_t;

  let cosi = clamp(-dot(I, N), 0.0, 1.0);
  let sini = sqrt(max(0.0, 1.0 - cosi*cosi));
  let tir  = eta * sini > 1.0;

  let r0 = pow((eta_i - eta_t) / (eta_i + eta_t), 2.0);
  let F  = r0 + (1.0 - r0) * pow(1.0 - cosi, 5.0);

  var dir: vec3f;
  if (tir || rng_next_float(rng) < F) {
    dir = reflect(I, N);
  } else {
    dir = refract(I, N, eta);
  }

  return material_behaviour(true, vec3f(1.0), normalize(dir));
}


fn emissive(emission: vec3f, intensity: f32) -> material_behaviour {
  // só adiciona energia e termina o caminho
  return material_behaviour(false, emission * intensity, vec3f(0.0));
}

fn sun_specular(n: vec3f, v_dir: vec3f, shininess: f32, ks: f32) -> vec3f {
  // v_dir = direção da câmera para o ponto (aprox = -ray.direction)
  let L = normalize(vec3(uniforms[13], uniforms[14], uniforms[15]));
  let V = normalize(-v_dir);
  let H = normalize(L + V);
  let spec = pow(max(dot(n, H), 0.0), max(shininess, 1.0));
  let sunCol = clamp(int_to_rgb(i32(uniforms[17])), vec3f(0.0), vec3f(1.0));
  let sunI   = max(uniforms[16], 0.0);
  return sunCol * (sunI * ks * spec);
}

fn eval_material_for_hit(hit: hit_record, in_dir: vec3f, rng: ptr<function,u32>) -> material_behaviour {
  let albedo = hit.object_color.xyz;

  // Layout do material do projeto:
  // x = smoothness  (>0 metal/difuso, 0 difuso puro, <0 dielétrico)
  // y = absorption/fuzz (em metal vira fuzz)
  // z = specular probability (prob de reflexão especular vs lambert)
  // w = light (emissivo)
  let m          = hit.object_material;
  let smoothness = m.x;
  let fuzz       = clamp(m.y, 0.0, 1.0);
  var specW      = clamp(m.z, 0.0, 1.0);
  let emissI     = max(m.w, 0.0);

  // Emissivo: termina caminho
  if (emissI > 0.0) {
    return emissive(albedo, emissI);
  }

  // Dielectric (vidro) quando smoothness < 0.0
  if (smoothness < 0.0) {
    // vidro incolor: dielectric já retorna attenuation = vec3(1)
    return dielectric(hit.normal, in_dir, 1.5, rng);
  }

  // *** NUNCA deixar o chão reflexivo ***
  // só permite espelho quando a superfície é realmente lisa
  if (!(smoothness > 0.95)) {
    specW = 0.0; // força difuso (chão com 0.74 vira difuso)
  }

  // Metal/espelho vs difuso por probabilidade
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




fn trace(r_in: ray, rng_state: ptr<function, u32>) -> vec3f
{
  // uniforms do app:
  // [2]  Max Bounces
  // [11] BGC1 (topo)   [12] BGC2 (horizonte)
  var maxbounces = i32(uniforms[2]);

  let bgTop     = int_to_rgb(i32(uniforms[11]));
  let bgHorizon = int_to_rgb(i32(uniforms[12]));

  var radiance   = vec3f(0.0);
  var throughput = vec3f(1.0);
  var r_         = r_in;

  const RR_START : i32 = 3;

  for (var bounce = 0; bounce < maxbounces; bounce = bounce + 1)
  {
    // 1) Interseção
    let rec = check_ray_collision(r_, RAY_TMAX);

    if (!rec.hit_anything) {
      let t  = clamp(0.5 * (r_.direction.y + 1.0), 0.0, 1.0);
      let bg = mix(bgHorizon, bgTop, t);
      radiance += throughput * bg;
      break;
    }

    // 2) Emissivo direto no hit (material.w = intensidade)
    let emitI = max(rec.object_material.w, 0.0);
    if (emitI > 0.0) {
      radiance += throughput * rec.object_color.xyz * emitI;
      break;
    }

    // 3) Luz direta (reduz ruído) — QUAD preferencial, senão BOX emissivo
    var added_direct = false;

    // --- QUAD emissivo ---
    let q_idx = first_emissive_quad();
    if (q_idx >= 0) {
      let q   = quadsb[q_idx];
      let Le  = q.color.xyz * max(q.material.w, 0.0);

      // normal e área do quad
      var nL = normalize(cross(q.u.xyz, q.v.xyz));

      // garante que a normal do quad aponte para o ponto rec.p
      let pQuad = q.Q.xyz + 0.5 * q.u.xyz + 0.5 * q.v.xyz; // centro aproximado
      let toP = normalize(rec.p - pQuad);
      if (dot(nL, toP) < 0.0) { nL = -nL; }

      let A   = max(length(cross(q.u.xyz, q.v.xyz)), 1e-6);

      // amostragem u,v aleatória estável
      let u   = rng_next_float(rng_state);
      let v   = rng_next_float(rng_state);
      let pL  = q.Q.xyz + u * q.u.xyz + v * q.v.xyz;

      let Lvec = pL - rec.p;
      let dist = max(length(Lvec), 1e-6);
      let wi   = Lvec / dist;

      let NdL  = max(dot(rec.normal, wi), 0.0);
      let cosL = max(dot(nL, -wi), 0.0);

      if (NdL > 0.0 && cosL > 0.0) {
        // shadow-ray
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

    // -------- BOX emissivo (teto fino) --------
  if (!added_direct) {
    var box_i = -1;
    let nb = i32(arrayLength(&boxesb));
    for (var bi = 0; bi < nb; bi = bi + 1) {
      if (boxesb[bi].material.w > 0.0) { box_i = bi; break; }
    }

    if (box_i >= 0) {
      let b   = boxesb[box_i];
      let Le  = b.color.xyz * max(b.material.w, 0.0) * 0.8; // leve exposure down

      // amostra a FACE INFERIOR (teto), alinhado aos eixos
      let c   = b.center.xyz;
      let e   = b.radius.xyz;          // semi-extensões (x,z = largura/profundidade)
      var nL  = vec3f(0.0, -1.0, 0.0);

      // normal aponta para o ponto sendo sombreado
      let pFace = c + vec3f(0.0, -e.y, 0.0);
      let toP   = normalize(rec.p - pFace);
      if (dot(nL, toP) < 0.0) { nL = -nL; }

      // amostragem uniforme na face
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
          added_direct = true;
        }
      }
    }
  }

    // 4) Próximo salto com seu seletor de material
    var bhv = eval_material_for_hit(rec, r_.direction, rng_state);

    if (!bhv.scatter) { break; }

    throughput *= bhv.attenuation;
    let nd = dot(bhv.new_dir, rec.normal);
    let bias = select(-rec.normal, rec.normal, nd > 0.0); // empurra para fora do lado certo
    r_ = ray(rec.p + bias * 1e-3, normalize(bhv.new_dir));

    // 5) Russian Roulette (performance)
    if (bounce >= RR_START) {
      let p = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 0.95);
      if (rng_next_float(rng_state) > p) { break; }
      throughput /= p;
    }

    // clamp leve para evitar energia “explodida”
    throughput = clamp(throughput, vec3f(0.0), vec3f(8.0));
  }

  return radiance;
}





@compute @workgroup_size(THREAD_COUNT, THREAD_COUNT, 1)
fn render(@builtin(global_invocation_id) id : vec3u)
{
    let rez  = uniforms[1];
    let frame_idx = u32(uniforms[0]);                  // contador de frames (0,1,2,...)

    // RNG: posição do pixel, resolução e frame
    var rng_state = init_rng(vec2(id.x, id.y), vec2(u32(rez)), frame_idx);

    // coords do pixel
    let fragCoord = vec2f(f32(id.x), f32(id.y));

    // câmera (aspect = 1.0 como no template)
    let lookfrom = vec3(uniforms[7],  uniforms[8],  uniforms[9]);
    let lookat   = vec3(uniforms[23], uniforms[24], uniforms[25]);
    let cam = get_camera(lookfrom, lookat, vec3(0.0, 1.0, 0.0), uniforms[10], 1.0, uniforms[6], uniforms[5]);

    // SPP
    var samples_per_pixel = i32(uniforms[4]);
    if (samples_per_pixel < 1) { samples_per_pixel = 1; }

    // acumula cor em ESPAÇO LINEAR
    var color_lin = vec3f(0.0);

    // 1) loop das amostras -> 2) gera raio -> 3) trace -> 4) acumula
    for (var s = 0; s < samples_per_pixel; s = s + 1) {
        let jitter = sample_square(&rng_state);                 // jitter por amostra
        let uv = (fragCoord + jitter) / vec2(rez);              // UV com jitter
        let r  = get_ray(cam, uv, &rng_state);
        color_lin += trace(r, &rng_state);                      // trace devolve linear
    }

    // média do SPP em linear
    color_lin = color_lin / f32(samples_per_pixel);
    color_lin = clamp(color_lin, vec3f(0.0), vec3f(1.0));       // segurança

    // índice no framebuffer
    let map_fb = mapfb(id.xy, rez);

    // acumulação entre frames (uniforms[3] > 0.5 ativa)
    let should_accumulate = uniforms[3];

    var final_lin = color_lin;

    if (should_accumulate > 0.5) {
        // lê o que já está no rtfb (está em GAMMA), converte para LINEAR
        let prev_gamma = rtfb[map_fb].xyz;
        let prev_lin = vec3f(pow(prev_gamma.x, 2.2), pow(prev_gamma.y, 2.2), pow(prev_gamma.z, 2.2));

        // média cumulativa estável: new = (prev*frame + cur)/(frame+1)
        let n = f32(frame_idx) + 1.0;
        final_lin = (prev_lin * f32(frame_idx) + color_lin) / n;
    }

    // converte UMA vez para gamma e escreve
    let final_out = vec4(linear_to_gamma(final_lin), 1.0);
    rtfb[map_fb] = final_out;
    fb[map_fb]   = final_out;
}