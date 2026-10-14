import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d11);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
const CAM_HOME = new THREE.Vector3(4.2, 2.0, 5.2);
const TARGET_HOME = new THREE.Vector3(0, 0.25, 0);
camera.position.copy(CAM_HOME);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(TARGET_HOME);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 1.2;
controls.maxDistance = 30;
controls.minDistance = 1.2;
controls.maxPolarAngle = Math.PI * 0.495;

// свет
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(5, 8, 3);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -5; sun.shadow.camera.right = 5;
sun.shadow.camera.top = 5; sun.shadow.camera.bottom = -5;
scene.add(sun);
scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x202226, 0.5));

// земля + сетка
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(60, 60),
  new THREE.ShadowMaterial({ opacity: 0.35 })
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.73;
ground.receiveShadow = true;
scene.add(ground);

const grid = new THREE.GridHelper(30, 30, 0x2e3a4d, 0x1a212c);
grid.position.y = -0.729;
scene.add(grid);

function resize() {
  const w = canvas.parentElement.clientWidth;
  const h = canvas.parentElement.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas.parentElement);
resize();

// состояние
let car = null;
let bodyMaterial = null;
let allMaterials = [];

const loaderEl = document.getElementById('loader');
const loadText = document.getElementById('loadText');
const progressBar = document.getElementById('progressBar');
const statTris = document.getElementById('statTris');
const statVerts = document.getElementById('statVerts');
const statMats = document.getElementById('statMats');
const statTex = document.getElementById('statTex');
const statSize = document.getElementById('statSize');

function clearCar() {
  if (!car) return;
  scene.remove(car);
  car.traverse((o) => {
    if (o.isMesh) {
      o.geometry.dispose();
    }
  });
  car = null;
  bodyMaterial = null;
  allMaterials = [];
}

function analyse(model) {
  let tris = 0, verts = 0;
  const mats = new Set();
  const texs = new Set();
  const box = new THREE.Box3();
  model.traverse((o) => {
    if (o.isMesh) {
      o.castShadow = true;
      o.receiveShadow = true;
      const g = o.geometry;
      verts += g.attributes.position.count;
      const idx = g.index ? g.index.count : g.attributes.position.count;
      tris += Math.round(idx / 3);
      const m = Array.isArray(o.material) ? o.material : [o.material];
      m.forEach((mm) => {
        mats.add(mm);
        if (mm.map) texs.add(mm.map);
      });
      box.expandByObject(o);
    }
  });
  statTris.textContent = tris.toLocaleString('ru-RU');
  statVerts.textContent = verts.toLocaleString('ru-RU');
  statMats.textContent = mats.size;
  statTex.textContent = texs.size;
  if (!box.isEmpty()) {
    const s = new THREE.Vector3();
    box.getSize(s);
    statSize.textContent = `${s.x.toFixed(2)} × ${s.y.toFixed(2)} × ${s.z.toFixed(2)}`;
  }
  allMaterials = [...mats];
  // кузов = материал краски (clearcoat/paint), иначе крупнейший без текстуры
  let best = null, bestScore = -1;
  model.traverse((o) => {
    if (!o.isMesh) return;
    const g = o.geometry;
    const count = (g.index ? g.index.count : g.attributes.position.count);
    const m = Array.isArray(o.material) ? o.material[0] : o.material;
    if (!m) return;
    // витринный блеск: усиливаем отражения на металлике/лаке как в игре
    if (m.metalness > 0.5 || (m.clearcoat && m.clearcoat > 0)) {
      m.envMapIntensity = 1.4;
    }
    if (m.map) return;
    const isPaint = m.name.startsWith('paint_');
    const score = count + (isPaint ? 1e9 : 0);
    if (score > bestScore) { bestScore = score; best = m; }
  });
  bodyMaterial = best;
  if (bodyMaterial) {
    const c = bodyMaterial.color.clone();
    document.getElementById('paint').value =
      '#' + c.getHexString();
  }
}

const gltfLoader = new GLTFLoader();
function loadGLB(url, label) {
  loaderEl.classList.remove('hidden');
  loadText.textContent = `Загрузка ${label}…`;
  progressBar.style.width = '0%';
  clearCar();
  gltfLoader.load(
    url,
    (gltf) => {
      car = gltf.scene;
      scene.add(car);
      analyse(car);
      loaderEl.classList.add('hidden');
    },
    (xhr) => {
      if (xhr.total) {
        const p = (xhr.loaded / xhr.total) * 100;
        progressBar.style.width = p.toFixed(1) + '%';
        loadText.textContent = `Загрузка ${label}… ${(xhr.loaded / 1024 / 1024).toFixed(1)} / ${(xhr.total / 1024 / 1024).toFixed(1)} МБ`;
      } else {
        loadText.textContent = `Загрузка ${label}… ${(xhr.loaded / 1024 / 1024).toFixed(1)} МБ`;
      }
    },
    (err) => {
      loadText.textContent = 'Ошибка загрузки: ' + err.message;
      console.error(err);
    }
  );
}

loadGLB('model.glb', 'model.glb');

// UI
document.getElementById('autoRotate').addEventListener('change', (e) => {
  controls.autoRotate = e.target.checked;
});
document.getElementById('wireframe').addEventListener('change', (e) => {
  const v = e.target.checked;
  allMaterials.forEach((m) => { m.wireframe = v; });
});
document.getElementById('gridToggle').addEventListener('change', (e) => {
  grid.visible = e.target.checked;
  ground.visible = e.target.checked;
});
document.getElementById('exposure').addEventListener('input', (e) => {
  renderer.toneMappingExposure = parseFloat(e.target.value);
  document.getElementById('exposureVal').textContent = parseFloat(e.target.value).toFixed(2);
});
document.getElementById('resetBtn').addEventListener('click', () => {
  camera.position.copy(CAM_HOME);
  controls.target.copy(TARGET_HOME);
});
document.getElementById('paintApply').addEventListener('click', () => {
  if (!bodyMaterial) return;
  bodyMaterial.color.set(document.getElementById('paint').value);
});
document.getElementById('fileInput').addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  loadGLB(url, f.name);
});

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});
