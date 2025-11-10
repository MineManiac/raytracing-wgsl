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

// ================== MATERIAL HELPERS (D) ==================
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

fn envoriment_color(direction: vec3f, color1: vec3f, color2: vec3f) -> vec3f
{
  var unit_direction = normalize(direction);
  var t = 0.5 * (unit_direction.y + 1.0);
  var col = (1.0 - t) * color1 + t * color2;

  var sun_direction = normalize(vec3(uniforms[13], uniforms[14], uniforms[15]));
  var sun_color = int_to_rgb(i32(uniforms[17]));
  var sun_intensity = uniforms[16];
  var sun_size = uniforms[18];

  var sun = clamp(dot(sun_direction, unit_direction), 0.0, 1.0);
  col += sun_color * max(0, (pow(sun, sun_size) * sun_intensity));

  return col;
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

  return best;
}

fn lambertian(normal: vec3f, albedo: vec3f, rng_state: ptr<function,u32>) -> material_behaviour {
  let dir = cosine_sample_hemisphere(rng_state, normal);
  return material_behaviour(true, albedo, dir);
}



fn metal(normal: vec3f, in_dir: vec3f, albedo: vec3f, fuzz: f32, rng_state: ptr<function,u32>) -> material_behaviour {
  // direção refletida especular
  var refl = reflect(normalize(in_dir), normal);
  // fuzz -> microfacet simples
  let r = rng_next_vec3_in_unit_sphere(rng_state);
  refl = normalize(refl + clamp(fuzz, 0.0, 1.0) * r);
  // se refletiu "para dentro" do material, absorve (evita manchas escuras)
  if (dot(refl, normal) <= 0.0) {
    return material_behaviour(false, vec3f(0.0), vec3f(0.0));
  }
  return material_behaviour(true, albedo, refl);
}


fn dielectric(normal: vec3f, in_dir: vec3f, ior: f32, rng_state: ptr<function,u32>) -> material_behaviour {
  // ainda não usado no D; deixa absorver
  return material_behaviour(false, vec3f(0.0), vec3f(0.0));
}

// ---------- EMISSIVE (para C) ----------
fn emissive(emission: vec3f, intensity: f32) -> material_behaviour {
  return material_behaviour(false, emission * intensity, vec3f(0.0));
}

// Decodifica material do objeto atingido e devolve o comportamento.
// 'is_floor_fix': se a normal é quase +Y/-Y, não permitir metal (corrige reflexo do chão na cena Metal)
fn eval_material_for_hit(hit: hit_record,
                         in_dir: vec3f,
                         rng_state: ptr<function,u32>) -> material_behaviour
{
  // albedo e parâmetros vindos do seu record
  let albedo = hit.object_color.xyz;
  let m      = hit.object_material;      // x=tipo, y=fuzz, z=ior, w=emissive
  let mtype  = i32(round(m.x));
  let fuzz   = m.y;
  let ior    = max(1.0, m.z);
  let emissI = m.w;

  // "chão não espelha" (aproxima seu gabarito Metal)
  let is_floor_like = abs(hit.normal.y) > 0.9;

  // emissivo (para a próxima etapa; aqui só curto-circuito)
  if (mtype == 3) {
    return emissive(albedo, emissI);
  }

  // metal apenas se não for o chão
  if (mtype == 1 && !is_floor_like) {
    return metal(hit.normal, in_dir, albedo, fuzz, rng_state);
  }

  // dielectric (mantido como placeholder)
  if (mtype == 2) {
    return dielectric(hit.normal, in_dir, ior, rng_state);
  }

  // padrão → lambert
  return lambertian(hit.normal, albedo, rng_state);
}


fn trace(r: ray, rng_state: ptr<function, u32>) -> vec3f
{
  var maxbounces = i32(uniforms[2]);
  var light = vec3f(0.0);
  var color = vec3f(1.0);
  var r_ = r;

  // Céu (linear)
  let backgroundcolor1 = int_to_rgb(i32(uniforms[11])); // topo
  let backgroundcolor2 = int_to_rgb(i32(uniforms[12])); // horizonte

  // Sol (dos seus uniforms)
  let sunDir = normalize(vec3(uniforms[13], uniforms[14], uniforms[15]));
  let sunInt = max(uniforms[16], 0.0);
  let sunCol = clamp(int_to_rgb(i32(uniforms[17])), vec3f(0.0), vec3f(1.0));

  var behaviour = material_behaviour(true, vec3f(1.0), r_.direction);

  for (var j = 0; j < maxbounces; j = j + 1)
  {
    // 1) Interseção
    let rec = check_ray_collision(r_, RAY_TMAX);
    if (!rec.hit_anything) {
      // fundo gradiente (somar só em miss)
      let t = 0.5 * (normalize(r_.direction).y + 1.0);
      let bg = mix(backgroundcolor2, backgroundcolor1, t);  // (topo=BG1, baixo=BG2)
      light += color * bg;
      break;
    }

    // 2) Avalia material do objeto atingido (passa direção do raio atual)
    var bhv = eval_material_for_hit(rec, r_.direction, rng_state);

    // 3) Emissivo: acumula e termina
    if (!bhv.scatter && any(bhv.attenuation > vec3f(0.0))) {
      light += color * bhv.attenuation;
      break;
    }

    // 4) Absorção
    if (!bhv.scatter) { break; }

    // 5) Throughput e novo raio (com pequeno bias)
    color *= bhv.attenuation;
    r_ = ray(rec.p + rec.normal * 1e-3, normalize(bhv.new_dir));
  }

  return light; // gamma só no render()
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