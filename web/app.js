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
controls.maxDistance = 30;
controls.minDistance = 0.3;
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

// Ходовая как в GTA SA:
// - wheel_{rf,lf,rb,lb}_steer — ноды поворота (yaw вокруг Y Three =
//   вертикаль Z GTA; поворачиваются ТОЛЬКО передние rf/lf, ±35°);
// - wheel_*_spin — ноды вращения (вокруг X Three = ось ступицы X GTA);
// - steering — руль в салоне (фрейм «руль», пивот на центре; вращение
//   вокруг Z Three = колонка Y GTA, пропорционально повороту).
const FRONT_SIDES = ['rf', 'lf'];
const ALL_SIDES = ['rf', 'lf', 'rb', 'lb'];
const STEER_RATIO = 2.5; // руль:поворот, как в игре (руль отзывчивее)
let steerNodes = {};   // side -> Object3D (yaw)
let spinNodes = {};    // side -> Object3D (roll)
let steeringNode = null;
let spinAngle = 0;     // накопленный угол вращения колёс, рад
let steerDeg = 0;      // поворот передних колёс, градусы (+ = направо)
let wheelManualDeg = 0;
let wheelSpeed = 6.0;  // рад/с (+ = вперёд)
let spinOn = true;
let steerLinked = true;

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
  steerNodes = {};
  spinNodes = {};
  steeringNode = null;
  spinAngle = 0;
}

// Находит ноды ходовой по именам (иерархия из convert.py).
// Возвращает true, если есть хотя бы колёса или руль.
function findDrivetrain(model) {
  steerNodes = {};
  spinNodes = {};
  steeringNode = null;
  model.traverse((o) => {
    if (!o.isMesh && !o.isGroup && !o.isObject3D) return;
    const m = /^wheel_(rf|lf|rb|lb)_steer$/.exec(o.name || '');
    if (m) steerNodes[m[1]] = o;
    const s = /^wheel_(rf|lf|rb|lb)_spin$/.exec(o.name || '');
    if (s) spinNodes[s[1]] = o;
    if (o.name === 'steering') steeringNode = o;
  });
  const hasWheels = Object.keys(spinNodes).length > 0;
  const hasSteering = !!steeringNode;
  document.getElementById('wheelsPanel').classList.toggle('hidden', !(hasWheels || hasSteering));
  document.getElementById('wheelsNote').classList.toggle('hidden', hasWheels && hasSteering);
  applySteer();
  return hasWheels || hasSteering;
}

// Применяет углы: yaw передних нод + руль.
// Знаки: перед машины смотрит в −Z. Поворот направо (steerDeg>0):
// ноды steer — rotation.y<0; руль (вид водителя) — rotation.z<0 (по часовой).
function applySteer() {
  const steerRad = steerDeg * Math.PI / 180;
  for (const side of FRONT_SIDES) {
    if (steerNodes[side]) steerNodes[side].rotation.y = -steerRad;
  }
  if (!steeringNode) return;
  const wheelDeg = steerLinked ? steerDeg * STEER_RATIO : wheelManualDeg;
  steeringNode.rotation.z = -wheelDeg * Math.PI / 180;
}

function applySpin() {
  for (const side of ALL_SIDES) {
    if (spinNodes[side]) spinNodes[side].rotation.x = spinAngle;
  }
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
      spinAngle = 0;
      findDrivetrain(car);
      applySpin();
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

// --- Колёса и руль (как в GTA SA) ---
const steerSlider = document.getElementById('steerSlider');
const steerVal = document.getElementById('steerVal');
steerSlider.addEventListener('input', () => {
  steerDeg = parseFloat(steerSlider.value);
  steerVal.textContent = `${steerDeg.toFixed(1)}°`;
  applySteer();
});
const spinSpeed = document.getElementById('spinSpeed');
const spinVal = document.getElementById('spinVal');
spinSpeed.addEventListener('input', () => {
  wheelSpeed = parseFloat(spinSpeed.value);
  spinVal.textContent = `${wheelSpeed.toFixed(1)} рад/с`;
});
document.getElementById('spinToggle').addEventListener('change', (e) => {
  spinOn = e.target.checked;
});
document.getElementById('steerLink').addEventListener('change', (e) => {
  steerLinked = e.target.checked;
  applySteer();
});
const wheelManual = document.getElementById('wheelManual');
const wheelManualVal = document.getElementById('wheelManualVal');
wheelManual.addEventListener('input', () => {
  wheelManualDeg = parseFloat(wheelManual.value);
  wheelManualVal.textContent = `${wheelManualDeg.toFixed(0)}°`;
  applySteer();
});

// Пресеты камеры: кузов / колесо RF / салон (руль)
function flyTo(pos, tgt) {
  camera.position.set(pos[0], pos[1], pos[2]);
  controls.target.set(tgt[0], tgt[1], tgt[2]);
}
document.getElementById('camBody').addEventListener('click', () => {
  flyTo([CAM_HOME.x, CAM_HOME.y, CAM_HOME.z],
        [TARGET_HOME.x, TARGET_HOME.y, TARGET_HOME.z]);
});
document.getElementById('camWheel').addEventListener('click', () => {
  // переднее правое колесо (Three-координаты dummy wheel_rf_dummy)
  const w = (steerNodes.rf || spinNodes.rf)
    ? new THREE.Vector3().setFromMatrixPosition(
        (steerNodes.rf || spinNodes.rf).matrixWorld)
    : new THREE.Vector3(0.78, -0.41, -1.46);
  flyTo([w.x + 1.1, w.y + 0.5, w.z - 1.1], [w.x, w.y, w.z]);
});
document.getElementById('camSteer').addEventListener('click', () => {
  // руль: вид с места водителя
  const s = steeringNode
    ? new THREE.Vector3().setFromMatrixPosition(steeringNode.matrixWorld)
    : new THREE.Vector3(-0.34, 0.18, -0.6);
  flyTo([s.x + 0.55, s.y + 0.35, s.z + 0.75], [s.x, s.y, s.z]);
});

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = Math.min(clock.getDelta(), 0.05);
  // Вращение колёс: вперёд (машина смотрит в −Z) — rotation.x уменьшается.
  if (spinOn && wheelSpeed !== 0 && Object.keys(spinNodes).length) {
    spinAngle -= wheelSpeed * dt;
    applySpin();
  }
  controls.update();
  renderer.render(scene, camera);
});
