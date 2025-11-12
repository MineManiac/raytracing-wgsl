import { Sphere, Quad, Box, Mesh } from './objects.js';
import { loadMesh, getObjBoundingBox, getSpheresRandom } from './util.js';

const groundDefault = new Sphere([0, -1001, 0], [0.7, 0.7, 0.7], 1000, [0.9, 0.0, 0.6, 0.0]);



async function getAvailableScene(index, scenesList)
{
    let sceneName = Object.keys(scenesList)[index];
    let sceneFunction = eval(sceneName);
    return await sceneFunction();
}

async function Spheres(numSpheres = 4)
{
    let offset = [0, -1, -5];
    let spheres = [groundDefault];
    spheres = spheres.concat(getSpheresRandom(numSpheres, offset));

    spheres.push(new Sphere([offset[0], 1.3 + offset[1], offset[2]], [1.0, 1.0, 1.0], 1.3, [-1.0, 0.001, 0.9, 0.0]));
    spheres.push(new Sphere([-3.3 + offset[0], 1.3 + offset[1], offset[2]], [1.0, 0.1, 0.1], 1.3, [0.0, 0.0, 0.0, 0.0]));
    spheres.push(new Sphere([3.3 + offset[0], 1.3 + offset[1], offset[2]], [0.7, 0.7, 0.7], 1.3, [1.0, 0.0, 1.0, 0.0]));

    return {
        spheres, 
        quads: [], 
        boxes: [], 
        triangles: [], 
        meshes: [], 
        backgroundColor1: [0.0, 0.5, 1.0], 
        backgroundColor2: [1.0, 1.0, 1.0], 
        focusDistance: 5.0, 
        focusAngle: 0.2,
        sunIntensity: 1.0,
        samplesPerPixel: 1.0,
        maxBounces: 10.0
    };
}

async function Night()
{
    let { spheres, quads, boxes, triangles, meshes, backgroundColor1, backgroundColor2 } = await Spheres();

    return {
        spheres,
        quads,
        boxes,
        triangles,
        meshes,
        backgroundColor1: [0.0, 0.0, 0.0],
        backgroundColor2: [0.0, 0.0, 0.0],
        focusDistance: 5.0,
        focusAngle: 0.2,
        sunIntensity: 0.1,
        samplesPerPixel: 1.0,
        maxBounces: 10.0
    }
}

async function Basic()
{
    let spheres = [
        new Sphere([0, -1001, 0], [0.5, 0.5, 0.5], 1000, [0.0, 0.0, 0.0, 0.0]),
        new Sphere([-0.0, -0.5, -2.0], [1.0, 0.0, 0.0], 0.5, [0.0, 0.0, 0.0, 0.0]),
    ];

    return {
        spheres, 
        quads: [], 
        boxes: [], 
        triangles: [], 
        meshes: [], 
        backgroundColor1: [0.0, 0.5, 1.0], 
        backgroundColor2: [1.0, 1.0, 1.0],
        focusDistance: 5.0, 
        focusAngle: 0.0,
        sunIntensity: 1.0,
        samplesPerPixel: 1.0,
        maxBounces: 10.0
    };
}

async function Metal() 
{
	let spheres = [
		new Sphere([0, -1001, 0], [0.10980392156862745, 0.12156862745098039, 0.2901960784313726], 1000, [1, 0, 0.01, 0]), 
		new Sphere([0, -0.26, -2], [0.3, 0.3, 1], 0.5, [1, 0, 1, 0]), 
		new Sphere([-1, -0.5, -2], [1, 0.3, 0.3], 0.5, [1, 0, 1, 0]), 
		new Sphere([1, -0.5, -2], [0.3, 1, 0.3], 0.5, [1, 0, 1, 0]), 
		new Sphere([0.01, 0.51, -1.9], [1, 1, 1], 0.23, [1, 0, 1, 0]), 
	];

	let quads = [
	];

	let boxes = [
	];

	return {
		spheres : spheres,
		quads : quads,
		boxes : boxes,
		triangles: [],
		meshes: [],
		backgroundColor1 : [0, 0.5, 1],
		backgroundColor2 : [1, 1, 1],
		focusDistance: 5,
		focusAngle: 0,
		sunIntensity: 1,
		samplesPerPixel: 1,
		maxBounces: 30
	};
}

async function Fuzz() 
{
	let spheres = [
		new Sphere([0, -1001, 0], [0.21568627450980393, 0.21568627450980393, 0.40784313725490196], 1000, [0, 0, 0, 0]), 
		new Sphere([0, -0.2, -2], [1, 1, 1], 0.5, [1, 0.1, 1, 0]), 
		new Sphere([-1, -0.5, -2], [1, 1, 1], 0.5, [1, 0.5, 1, 0]), 
		new Sphere([1, -0.5, -2], [1, 1, 1], 0.5, [1, 0, 1, 0]), 
	];

	let quads = [
	];

	let boxes = [
	];

	return {
		spheres : spheres,
		quads : quads,
		boxes : boxes,
		triangles: [],
		meshes: [],
		backgroundColor1 : [0, 0.3803921568627451, 0.7607843137254902],
		backgroundColor2 : [1, 1, 1],
		focusDistance: 5,
		focusAngle: 0,
		sunIntensity: 1,
		samplesPerPixel: 1,
		maxBounces: 30
	};
}

async function Specular() 
{
	let spheres = [
		new Sphere([0, -1001, 0], [194/255, 163/255, 1], 1000, [0.74, 0, 1, 0]), 
		new Sphere([0, -0.5, -2], [1, 1, 1], 0.5, [1, 0, 0.1, 0]), 
		new Sphere([-1, -0.5, -2], [1, 1, 1], 0.5, [1, 0, 0.02, 0]), 
		new Sphere([1, -0.5, -2], [1, 1, 1], 0.5, [1, 0, 1, 0]), 
	];

	let quads = [
	];

	let boxes = [
		new Box([0, 1.54, -2, 0], [1, 1, 1], [0, 0, 0, 0], [1.3, 0.02, 0.36, 0], [0, 0, 0, 3]), 
	];

	return {
		spheres : spheres,
		quads : quads,
		boxes : boxes,
		triangles: [],
		meshes: [],
		backgroundColor1 : [0, 0, 0],
		backgroundColor2 : [0.34509803921568627, 0.6039215686274509, 0.9921568627450981],
		focusDistance: 5,
		focusAngle: 0,
		sunIntensity: 1,
		samplesPerPixel: 10,
		maxBounces: 30
	};
}

async function Emissive()
{
    let spheres = [
        groundDefault,
        new Sphere([.5, 0., -2.5], [1.0, 0.0, 0.0], 1.0, [0.0, 0.0, 0.0, 0.0]),
        new Sphere([-1.0, -0.5, -2.0], [1.0, 1.0, 1.0], 0.5, [0.0, 0.0, 0.0, 3.0]),
    ];

    return {
        spheres, 
        quads: [], 
        boxes: [], 
        triangles: [], 
        meshes: [], 
        backgroundColor1: [0.0, 0.0, 0.0], 
        backgroundColor2: [0.0, 0.0, 0.0],
        focusDistance: 5.0, 
        focusAngle: 0.0,
        sunIntensity: 0.0,
        samplesPerPixel: 1.0,
        maxBounces: 10.0
    };
}

async function Dielectric()
{
    let spheres = [
        groundDefault,
        new Sphere([0.0, -0.5, -4.0], [1.0, 0.0, 0.0], 0.5, [0.0, 0.0, 0.0, 0.0]),
        new Sphere([-0.2, 0.7, -4.0], [0.0, 0.0, 1.0], 0.5, [0.0, 0.0, 0.0, 0.0]),
        new Sphere([0.0, 0.0, -2.1], [1.0, 1.0, 1.0], 1, [-1.0, 1.5, 0.0, 0.0]),
    ];

    return {
        spheres, 
        quads: [], 
        boxes: [], 
        triangles: [], 
        meshes: [], 
        backgroundColor1: [0.0, 0.5, 1.0], 
        backgroundColor2: [1.0, 1.0, 1.0],
        focusDistance: 5.0, 
        focusAngle: 0.0,
        sunIntensity: 1.0,
        samplesPerPixel: 1.0,
        maxBounces: 10.0
    };
}

async function Cubes() 
{
  let spheres = [
    new Sphere([0, -1001, 0], [0.5, 0.5, 0.5], 1000, [0.9, 0, 0.6, 0]),
    new Sphere([0, 0.49, -1.5], [0.4, 0.9, 0.8], 0.25, [1, 0, 1, 0]), // espelhada
  ];

  let quads = [];

  let boxes = [
    new Box([-1, 0, -2.23, 0], [1, 0, 0], [0, 0, 0, 0], [0.5, 0.5, 0.5, 0], [0, 0, 0, 0]), // vermelho
    new Box([1, 0, -2.21, 0], [1, 1, 1], [0, 0, 0, 0], [0.5, 0.5, 0.5, 0], [0, 0, 0, 0]),   // branco
  ];

  return {
    spheres, quads, boxes, triangles: [], meshes: [],
    backgroundColor1: [0, 0.3803921568627451, 0.7607843137254902],
    backgroundColor2: [1, 1, 1],
    focusDistance: 5, focusAngle: 0,
    sunIntensity: 0.5,
    samplesPerPixel: 1,
    maxBounces: 30
  };
}

async function Cornell() {
  let spheres = [
    new Sphere([-0.5, -0.4, -1.4], [1, 1, 1], 0.5, [0, 0, 0, 0]),
    // espelho liso (azul claro só tinge o reflexo; se quiser neutro use [1,1,1])
    new Sphere([0.07, 0.42, -1.5], [0.9, 0.95, 1.0], 0.39, [1.0, 0.0, 1.0, 0.0]),
  ];

  const quads = [
    // esquerda (vermelha)
    new Quad([-1,-1, 0,0], [0,0,-2,0], [0, 2, 0,0], [0.80,0.18,0.18,1], [0,0,0,0]),
    // fundo (cinza claro)
    new Quad([-1,-1,-2,0], [2,0, 0,0], [0, 2, 0,0], [0.72,0.72,0.72,1], [0,0,0,0]),
	// frente (azul claro)
	new Quad([-1, -1,  0, 0], [ 2, 0, 0, 0], [0, 2, 0, 0], [0, 0.2, 1, 1], [0, 0, 0, 0]),
    // direita (verde)
    new Quad([ 1,-1,-2,0], [0,0, 2,0], [0, 2, 0,0], [0.18,0.80,0.18,1], [0,0,0,0]),
    // chão (cinza ~0.78)
    new Quad([ 1,-1,-2,0], [-2,0,0,0], [0,0, 2,0], [0.78,0.78,0.78,1], [0,0,0,0]),
    // teto (cinza ~0.78)
    new Quad([-1, 1,-2,0], [ 2,0,0,0], [0,0, 2,0], [0.78,0.78,0.78,1], [0,0,0,0]),
    // luz (mais intensa)
    new Quad([-0.5,0.98,-1.5,0], [1,0,0,0], [0,0,1,0], [1,1,1,1], [0,0,0,1.0]),
  ];

  let boxes = [
    new Box([0.5, -0.5, -1.36, 0], [0.07, 0.27, 0.86, 0], [0,0,0,0], [0.24, 0.5, 0.24, 0], [0.0, 0.0, 0.0, 0]),
  ];

  return {
    spheres, quads, boxes, triangles: [], meshes: [],
    backgroundColor1: [0,0,0], backgroundColor2: [0,0,0],
    focusDistance: 5, focusAngle: 0,
    sunIntensity: 0,
    samplesPerPixel: 8,
    maxBounces: 14,
  };
}

async function Mirror() 
{
  let spheres = [
    new Sphere([0.3, 0, -1.4], [0.023529411764705882, 0.9764705882352941, 0.7843137254901961], 0.5, [0, 0, 0, 0]),
    new Sphere([-0.5, -0.4, -1.4], [0.34901960784313724, 0, 1], 0.3, [1, 0, 0.5, 0]), // espelho
  ];

  let quads = [
	// fundo difuso cinza neutro
	new Quad([-1,-1,-2,0],[2,0,0,0],[0,2,0,0],[0.8,0.8,0.8,1],[0,0,0,0]),
	// esquerda espelho
	new Quad([-1,-1, 0,0],[0,0,-2,0],[0,2,0,0],[1,1,1,1],[1.0, 0.0, 1.0, 0.0]),
	// direita espelho
	new Quad([ 1,-1,-2,0],[0,0, 2,0],[0,2,0,0],[1,1,1,1],[1.0, 0.0, 1.0, 0.0]),
	// chão difuso
	new Quad([ 1,-1,-2,0],[-2,0,0,0],[0,0, 2,0],[0.8,0.8,0.8,1],[0,0,0,0]),
	// teto difuso
	new Quad([-1, 1,-2,0],[ 2,0,0,0],[0,0, 2,0],[0.8,0.8,0.8,1],[0,0,0,0]),
	// luz
	new Quad([-0.5,0.99,-1.5,0],[1,0,0,0],[0,0,1,0],[1,1,1,1],[0,0,0,0.7])
	];

  let boxes = [
    new Box([0.5, -0.5, -1.36, 0], [0.03137254901960784, 0.23529411764705882, 0.8627450980392157, 0],
            [0, 0, 0, 0], [0.24, 0.5, 0.24, 0], [0.5, 0.5, 0.5, 0]),
  ];

  return {
    spheres, quads, boxes, triangles: [], meshes: [],
    backgroundColor1: [0, 0, 0], backgroundColor2: [0, 0, 0],
    focusDistance: 5, focusAngle: 0,
    sunIntensity: 0,
    samplesPerPixel: 3,
    maxBounces: 10
  };
}

async function Infinite() {
  // Materiais “constantes”
  const M = [1.0, 0.0, 1.0, 0.0];      // espelho perfeito
  const C = [0.98, 0.98, 0.98, 10.0];   // quase branco (evita energia extra)

  let spheres = [
    // esfera difusa só para compor
    new Sphere([0.30, 0.00, -1.00], [0.40, 0.90, 0.80], 0.30, [0.0, 0.0, 0.0, 0.0]),
    // “lâmpada” emissiva — raio negativo!  (w = intensidade)
    new Sphere([-0.50,-0.40,-1.00], [1.0, 1.0, 1.0], -0.25, [0.0, 0.0, 0.0, 0.30]),
    // esfera espelho ideal
    new Sphere([-0.32, 0.50, -1.00], [0.98, 0.98, 0.98], 0.20, [1.0, 0.0, 1.0, 0.0]),
  ];

  // 6 faces do “quarto” todas espelhadas
  let quads = [
    new Quad([-1,-1, 0,0], [0,0,-2,0], [0, 2,0,0], C, M), // esquerda
    new Quad([-1,-1,-2,0], [2,0, 0,0], [0, 2,0,0], C, M), // fundo
    new Quad([-1, 1,-2,0], [2,0, 0,0], [0, 0,2,0], C, M), // teto
    new Quad([ 1,-1,-2,0], [0,0, 2,0], [0, 2,0,0], C, M), // direita
    new Quad([ 1,-1,-2,0], [-2,0,0,0], [0, 0,2,0], C, M), // chão
    new Quad([ 1,-1, 0,0], [-2,0,0,0], [0, 2,0,0], C, M), // frente
  ];

  let boxes = [];

  return {
    spheres, quads, boxes, triangles: [], meshes: [],
    backgroundColor1: [0,0,0],
    backgroundColor2: [0,0,0],
    focusDistance: 5, focusAngle: 0,
    sunIntensity: 0,          // <— desligado
    samplesPerPixel: 6,       // 5–8 OK
    maxBounces: 20            // mais “profundo”
  };
}

async function Bunny()
{
    let offset = [0, -1, -2];
    let spheres = [groundDefault];
    spheres = spheres.concat(getSpheresRandom(3, offset));

    let { verticesMesh, trianglesMesh, triangles } = await loadMesh('./media/stanford-bunny.obj');
    
    let meshes = [new Mesh([0, -1, -4], [2, 2, 2], [0, .2, 0], [0.4, 0.3, 0.6, 1.0], [1.0, 0.0, 0.5, 0.0], 0, 0, triangles.length)];
    let { min, max } = getObjBoundingBox(verticesMesh);
    meshes[0].setBoundingBox(min, max);

    return {
        spheres, 
        quads: [], 
        boxes: [], 
        triangles, 
        meshes, 
        backgroundColor1: [0.0, 0.5, 1.0], 
        backgroundColor2: [1.0, 1.0, 1.0], 
        focusDistance: 4.0, 
        focusAngle: 0.2,
        sunIntensity: 1.0,
        samplesPerPixel: 1.0,
        maxBounces: 10.0
    };
}

async function Suzzanne()
{
    let { verticesMesh, trianglesMesh, triangles } = await loadMesh('./media/suzanne.obj');
    let meshes = [new Mesh([0, 0.0, -6], [2, 2, 2], [0, 0, 0], [0.6, 0.3, 0.7, 1.0], [0.0, 0.0, 0.0, 0.0], 0, 0, triangles.length)];
    let { min, max } = getObjBoundingBox(verticesMesh);
    meshes[0].setBoundingBox(min, max);

    return {
        spheres: [], 
        quads: [], 
        boxes: [], 
        triangles, 
        meshes, 
        backgroundColor1: [0.0, 0.5, 1.0], 
        backgroundColor2: [1.0, 1.0, 1.0], 
        focusDistance: 5.0, 
        focusAngle: 0.2,
        sunIntensity: 1.0,
        samplesPerPixel: 1.0,
        maxBounces: 10.0
    };
}

async function Rotation() 
{
	let spheres = [
		new Sphere([0, -1001, 0], [0.5, 0.5, 0.5], 1000, [0.9, 0, 0.6, 0]), 
		new Sphere([0, -0.58, -2.34], [0.8, 0.1, 0.2], 0.4, [0, 0.001, 0, 0]), 
		new Sphere([0.54, -0.81, -1.8], [0.4, 0.9, 0.8], 0.19, [0, 0, 0, 3]), 
	];

	let quads = [
	];

	let boxes = [
		new Box([-1.37, 0.6, -2.68, 0], [1, 1, 1], [-0.46, 5.32, 0.23, 0], [0.89, 1.41, 0.15, 0], [1, 0, 0.9, 0]), 
		new Box([0.49, 0.07, -2.04, 0], [1, 1, 1], [-0.56, -0.42, 0.31, 0], [0.2, 0.2, 0.2, 0], [0, 0, 0, 0]), 
		new Box([1.21, -0.36, -1.74, 0], [1, 1, 1], [-1.21, -3.65, -1.76, 0], [0.3, 0.3, 0.3, 0], [1, 0.03, 1, 0]), 
		new Box([1.13, -0.71, -1.17, 0], [1, 1, 1], [1.13, -0.73, 0, 0], [0.81, 0.01, 0.94, 0], [1, 0.01, 0.8, 0]), 
	];

	let { verticesMesh, trianglesMesh, triangles } = await loadMesh('./media/stanford-bunny.obj');
    let meshes = [new Mesh([-0.69, -1, -1.77], [.4, .4, .4], [0, -.75, 0], [0.4, 0.7, 0.6, 1.0], [1.0, 0.0, 0.5, 0.0], 0, 0, triangles.length)];
    let { min, max } = getObjBoundingBox(verticesMesh);
    meshes[0].setBoundingBox(min, max);

	return {
		spheres : spheres,
		quads : quads,
		boxes : boxes,
		triangles: triangles,
		meshes: meshes,
		backgroundColor1 : [1, 1, 1],
		backgroundColor2 : [0.06274509803921569, 0.0784313725490196, 0.19607843137254902],
		focusDistance: 3,
		focusAngle: 0,
		sunIntensity: 1,
		samplesPerPixel: 1,
		maxBounces: 10
	};
}

async function Everything()
{
    let offset = [0, -1, -2];
    let spheres = [groundDefault];
    spheres = spheres.concat(getSpheresRandom(3, offset));

    let boxes = [new Box([-1.0, 0.0, -2.6], [0.3, 0.5, 0.3], [0.0, -0.6, 0.0], [0.3, 0.9, 0.3], [1.0, 0.0, 0.9, 0.0]),
                new Box([1.0, 0.00, -2.6], [0.1, 0.2, 0.5], [0.0, -0.6, 0.0], [0.3, 0.9, 0.3], [-1.0, 0.0, 0.0, 0.0]),
                new Box([0.0, 0.00, -2.6], [0.5, 0.0, 0.5], [0.6, -0.6, 0.1], [0.3, 0.3, 0.3], [0.0, 0.0, 0.0, 0.0]),
                new Box([0.0, 1.0, -2.6], [0.3, 0.3, 0.3], [-0.6, -0.2, -0.1], [0.3, 0.3, 0.3], [1.0, 0.0, 0.9, 0.0]),
    ];

    let { verticesMesh, trianglesMesh, triangles } = await loadMesh('./media/suzanne.obj');
    let meshes = [new Mesh([0, -0.6, -2], [.25, .25, .25], [0.5, 0.3, 0.2], [0.6, 0.3, 0.7, 1.0], [1.0, 0.0, 0.7, 0.0], 0, 0, triangles.length)];
    let { min, max } = getObjBoundingBox(verticesMesh);
    meshes[0].setBoundingBox(min, max);

    return {
        spheres: spheres, 
        quads: [], 
        boxes: boxes, 
        triangles: triangles, 
        meshes: meshes,
        backgroundColor1: [0.0, 0.5, 1.0], 
        backgroundColor2: [1.0, 1.0, 1.0], 
        focusDistance: 3.0, 
        focusAngle: 0.0,
        sunIntensity: 1.0,
        samplesPerPixel: 1.0,
        maxBounces: 10.0
    };
}

export { getAvailableScene };