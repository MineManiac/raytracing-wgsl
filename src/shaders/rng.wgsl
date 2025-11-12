fn rng_next_int(state: ptr<function, u32>)
{
    // PCG random number generator
    // Based on https://www.shadertoy.com/view/XlGcRh

    let oldState = *state + 747796405u + 2891336453u;
    let word = ((oldState >> ((oldState >> 28u) + 4u)) ^ oldState) * 277803737u;
    *state = (word >> 22u) ^ word;
}

fn rng_next_float(state: ptr<function, u32>) -> f32
{
    rng_next_int(state);
    return f32(*state) / f32(0xffffffffu);
}

fn init_rng(pixel: vec2<u32>, resolution: vec2<u32>, frame: u32) -> u32
{
   let pixelindex = pixel.x + pixel.y * resolution.x;
   let seed = pixelindex + frame * 719393;
   return seed;
}

fn rng_next_vec3_in_unit_disk(state: ptr<function, u32>) -> vec3<f32>
{
    // r^2 ~ U(0, 1)
    let r = sqrt(rng_next_float(state));
    let alpha = 2f * PI * rng_next_float(state);

    let x = r * cos(alpha);
    let y = r * sin(alpha);

    return vec3(x, y, 0f);
}

fn sample_square(rngState: ptr<function, u32>) -> vec2f
{
  return vec2f(rng_next_float(rngState), rng_next_float(rngState));
}