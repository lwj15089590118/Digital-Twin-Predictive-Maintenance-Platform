/**
 * ================================================================================
 *  数字孪生 3D 可视化场景 (Three.js)
 * ================================================================================
 *  职责:
 *    1. 构建包含 主电机(MOTOR-001) / 离心风机(FAN-001) / 齿轮传动箱(GEARBOX-001)
 *       的 3D 数字化车间, 含地面、安全黄线、支柱与灯光环境;
 *    2. 设备外观颜色随健康评分动态渐变: 绿(健康) -> 黄(提醒) -> 橙(严重) -> 红(紧急);
 *    3. 设备运转动画(转子/叶轮/齿轮啮合)速度随健康度衰减, 状态信标灯呼吸闪烁;
 *    4. 鼠标点击设备弹出详情面板(健康评分 / RUL / 故障诊断, 数据来自看板 API);
 *       悬停高亮 + 顶部设备铭牌文字标签;
 *    5. 数据来源:
 *         - 看板模式: 轮询 /api/devices 获取实时健康评分;
 *         - 独立模式: 直接以 file:// 打开时自动切换内置演示数据。
 *
 *  依赖: three.min.js (r128) 与 OrbitControls.js, 由宿主页面(dashboard)通过 CDN 引入。
 * ================================================================================
 */

/* ---------------------------------- 全局配置 ---------------------------------- */
const CONFIG = {
  // 设备注册表: 与后端 device_id 一致, 含 3D 摆放位置与展示名称
  devices: [
    { id: 'MOTOR-001',   name: '主电机',     type: 'motor',   x: -8, z: 0 },
    { id: 'FAN-001',     name: '离心风机',   type: 'fan',     x:  0, z: 0 },
    { id: 'GEARBOX-001', name: '齿轮传动箱', type: 'gearbox', x:  8, z: 0 }
  ],
  // 健康评分 -> 颜色的锚点(线性插值渐变)
  colorStops: [
    { score: 100, color: { r: 0x2e, g: 0xcc, b: 0x71 } },   // 绿色: 健康
    { score: 75,  color: { r: 0xf1, g: 0xc4, b: 0x0f } },   // 黄色: 提醒
    { score: 55,  color: { r: 0xe6, g: 0x7e, b: 0x22 } },   // 橙色: 严重
    { score: 0,   color: { r: 0xe7, g: 0x4c, b: 0x3c } }    // 红色: 紧急
  ],
  pollInterval: 3000,       // 健康数据轮询周期(ms)
  camera: { fov: 50, near: 0.1, far: 500 }
};

// 运行时状态: deviceId -> { group, health, spinningParts[], beacon, labelSprite }
const runtimeDevices = {};
let scene, camera, renderer, raycaster, mouse, controls;
let selectedDeviceId = null;      // 当前选中的设备(点击)
let hoveredDeviceId = null;       // 当前悬停的设备
let standaloneMode = false;       // true = 无后端, 使用内置演示数据
const clock = new THREE.Clock();

/* ------------------------------ 工具函数 ------------------------------ */

/**
 * 健康评分(0~100)转颜色: 在 colorStops 锚点之间线性插值, 返回 THREE.Color
 */
function healthToColor(score) {
  const stops = CONFIG.colorStops;
  if (score >= stops[0].score) return new THREE.Color(stops[0].color.r / 255, stops[0].color.g / 255, stops[0].color.b / 255);
  for (let i = 0; i < stops.length - 1; i++) {
    const high = stops[i], low = stops[i + 1];
    if (score <= high.score && score >= low.score) {
      // t: 在 [low, high] 区间内的归一化位置
      const t = (high.score - score) / (high.score - low.score);
      const mix = (a, b) => (a + (b - a) * t) / 255;
      return new THREE.Color(mix(high.color.r, low.color.r),
                             mix(high.color.g, low.color.g),
                             mix(high.color.b, low.color.b));
    }
  }
  const last = stops[stops.length - 1].color;
  return new THREE.Color(last.r / 255, last.g / 255, last.b / 255);
}

/**
 * 健康评分转状态文本
 */
function healthToStatus(score) {
  if (score >= 80) return '健康';
  if (score >= 60) return '轻度衰退';
  if (score >= 40) return '明显退化';
  return '严重退化 / 临故障';
}

/**
 * 生成文字贴图 Sprite(设备铭牌标签)
 */
function makeTextSprite(text, subText) {
  const canvas = document.createElement('canvas');
  canvas.width = 512; canvas.height = 160;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(15, 25, 40, 0.85)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = 'rgba(80, 180, 255, 0.9)';
  ctx.lineWidth = 6;
  ctx.strokeRect(3, 3, canvas.width - 6, canvas.height - 6);
  ctx.font = 'bold 64px "Microsoft YaHei", sans-serif';
  ctx.fillStyle = '#ffffff';
  ctx.textAlign = 'center';
  ctx.fillText(text, canvas.width / 2, 68);
  ctx.font = '40px "Microsoft YaHei", sans-serif';
  ctx.fillStyle = '#7fd4ff';
  ctx.fillText(subText, canvas.width / 2, 128);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(4.4, 1.4, 1);
  return sprite;
}

/**
 * 生成发光信标灯(设备顶部状态灯, 颜色随健康变化并呼吸闪烁)
 */
function makeBeacon() {
  const group = new THREE.Group();
  const pole = new THREE.Mesh(
    new THREE.CylinderGeometry(0.06, 0.06, 0.8, 8),
    new THREE.MeshStandardMaterial({ color: 0x555555, roughness: 0.6 })
  );
  pole.position.y = 0.4;
  const bulb = new THREE.Mesh(
    new THREE.SphereGeometry(0.22, 16, 16),
    new THREE.MeshStandardMaterial({ color: 0x2ecc71, emissive: 0x2ecc71,
                                     emissiveIntensity: 1.2, roughness: 0.3 })
  );
  bulb.position.y = 0.9;
  // 信标点光源(紧急时红灯照亮周围)
  const light = new THREE.PointLight(0x2ecc71, 0.6, 6);
  light.position.y = 0.9;
  group.add(pole, bulb, light);
  group.userData.bulb = bulb;
  group.userData.light = light;
  return group;
}

/* ------------------------------ 场景环境搭建 ------------------------------ */

/**
 * 初始化渲染器 / 相机 / 轨道控制器 / 灯光
 */
function initRenderer(container) {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1420);
  scene.fog = new THREE.Fog(0x0d1420, 40, 120);          // 远景雾效增强纵深

  camera = new THREE.PerspectiveCamera(CONFIG.camera.fov,
    container.clientWidth / container.clientHeight,
    CONFIG.camera.near, CONFIG.camera.far);
  camera.position.set(0, 12, 24);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  // 轨道控制器: 支持鼠标拖拽旋转 / 滚轮缩放 / 右键平移
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI / 2.1;                 // 防止钻到地下
  controls.target.set(0, 2, 0);

  // 灯光: 环境光 + 主平行光(投影) + 两盏车间顶灯点光源
  scene.add(new THREE.AmbientLight(0xffffff, 0.45));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
  dirLight.position.set(15, 25, 12);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.set(2048, 2048);
  dirLight.shadow.camera.left = -30; dirLight.shadow.camera.right = 30;
  dirLight.shadow.camera.top = 30; dirLight.shadow.camera.bottom = -30;
  scene.add(dirLight);
  const lamp1 = new THREE.PointLight(0x88bbff, 0.5, 60); lamp1.position.set(-10, 14, -8);
  const lamp2 = new THREE.PointLight(0xffddaa, 0.5, 60); lamp2.position.set(10, 14, 8);
  scene.add(lamp1, lamp2);

  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();
}

/**
 * 构建车间环境: 地面 / 网格 / 安全黄线 / 四角支柱
 */
function buildEnvironment() {
  // 主地面(深色哑光, 接收阴影)
  const floor = new THREE.Mesh(
    new THREE.BoxGeometry(46, 0.4, 30),
    new THREE.MeshStandardMaterial({ color: 0x232b38, roughness: 0.9 })
  );
  floor.position.y = -0.2;
  floor.receiveShadow = true;
  scene.add(floor);

  // 辅助网格线
  const grid = new THREE.GridHelper(46, 23, 0x3a4a63, 0x2a3648);
  grid.position.y = 0.01;
  scene.add(grid);

  // 设备巡检安全黄线(围绕三台设备的矩形框)
  const lineMat = new THREE.MeshStandardMaterial({ color: 0xd9a514, roughness: 0.7 });
  const mkLine = (w, d, x, z) => {
    const line = new THREE.Mesh(new THREE.BoxGeometry(w, 0.06, d), lineMat);
    line.position.set(x, 0.04, z);
    scene.add(line);
  };
  mkLine(26, 0.3, 0, -5.5); mkLine(26, 0.3, 0, 5.5);   // 上下边
  mkLine(0.3, 11, -13, 0);   mkLine(0.3, 11, 13, 0);   // 左右边

  // 四角结构支柱
  const pillarMat = new THREE.MeshStandardMaterial({ color: 0x3b4759, roughness: 0.5, metalness: 0.4 });
  [[-20, -12], [-20, 12], [20, -12], [20, 12]].forEach(([x, z]) => {
    const pillar = new THREE.Mesh(new THREE.BoxGeometry(1.2, 12, 1.2), pillarMat);
    pillar.position.set(x, 6, z);
    pillar.castShadow = true;
    scene.add(pillar);
  });
}

/* ------------------------------ 设备建模 ------------------------------ */

/**
 * 通用基座: 设备共用的减振底座
 */
function makeBaseplate(width, depth) {
  const base = new THREE.Mesh(
    new THREE.BoxGeometry(width, 0.4, depth),
    new THREE.MeshStandardMaterial({ color: 0x4a5568, roughness: 0.7, metalness: 0.3 })
  );
  base.position.y = 0.2;
  base.castShadow = true; base.receiveShadow = true;
  return base;
}

/**
 * 建模: 主电机(卧式电机本体 + 散热筋 + 端盖 + 转轴 + 接线盒)
 * 返回 { group, spinning: [转子轴] }
 */
function buildMotor() {
  const group = new THREE.Group();
  group.add(makeBaseplate(5.6, 3.6));

  const bodyMat = new THREE.MeshStandardMaterial({ color: 0x2ecc71, roughness: 0.45, metalness: 0.5 });
  const steelMat = new THREE.MeshStandardMaterial({ color: 0x9aa5b1, roughness: 0.3, metalness: 0.8 });

  // 电机本体(水平圆柱: 旋转轴对齐 X)
  const body = new THREE.Mesh(new THREE.CylinderGeometry(1.4, 1.4, 4.4, 32), bodyMat);
  body.rotation.z = Math.PI / 2;
  body.position.y = 1.8;
  body.castShadow = true;
  group.add(body);

  // 散热筋(沿本体均匀分布的纵向浅筋条)
  for (let i = 0; i < 8; i++) {
    const fin = new THREE.Mesh(new THREE.BoxGeometry(3.8, 0.08, 0.5), steelMat);
    const angle = (i / 8) * Math.PI * 2;
    fin.position.set(0, 1.8 + Math.cos(angle) * 1.42, Math.sin(angle) * 1.42);
    fin.rotation.x = -angle;
    group.add(fin);
  }

  // 前后端盖
  [-2.3, 2.3].forEach((x) => {
    const cap = new THREE.Mesh(new THREE.CylinderGeometry(1.15, 1.15, 0.5, 24), steelMat);
    cap.rotation.z = Math.PI / 2;
    cap.position.set(x, 1.8, 0);
    group.add(cap);
  });

  // 伸出转轴(动画中绕 X 轴旋转)
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 1.8, 16), steelMat);
  shaft.rotation.z = Math.PI / 2;
  shaft.position.set(3.2, 1.8, 0);
  group.add(shaft);

  // 接线盒(顶部方盒)
  const terminal = new THREE.Mesh(
    new THREE.BoxGeometry(1.2, 0.7, 1.0),
    new THREE.MeshStandardMaterial({ color: 0x34495e, roughness: 0.6 })
  );
  terminal.position.set(-0.8, 3.3, 0);
  group.add(terminal);

  return { group, spinning: [shaft], bodyMeshes: [body] };
}

/**
 * 建模: 离心风机(蜗壳 + 进风口 + 旋转叶轮 + 排风管 + 桁架支腿)
 */
function buildFan() {
  const group = new THREE.Group();
  group.add(makeBaseplate(6.0, 4.4));

  const shellMat = new THREE.MeshStandardMaterial({ color: 0x2ecc71, roughness: 0.5, metalness: 0.4 });
  const ductMat = new THREE.MeshStandardMaterial({ color: 0xb0bec5, roughness: 0.4, metalness: 0.7 });

  // 蜗壳(大圆柱竖放, 中空感用双圆柱叠加表达)
  const shell = new THREE.Mesh(new THREE.CylinderGeometry(2.0, 2.0, 2.4, 36), shellMat);
  shell.position.y = 2.6;
  shell.castShadow = true;
  group.add(shell);

  // 中心轮毂 + 六片叶轮叶片(整体绕 Y 轴旋转)
  const impeller = new THREE.Group();
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.55, 1.2, 20),
    new THREE.MeshStandardMaterial({ color: 0x78909c, metalness: 0.8, roughness: 0.3 }));
  impeller.add(hub);
  const bladeGeo = new THREE.BoxGeometry(1.5, 0.08, 0.7);
  const bladeMat = new THREE.MeshStandardMaterial({ color: 0xcfd8dc, metalness: 0.7, roughness: 0.3 });
  for (let i = 0; i < 6; i++) {
    const blade = new THREE.Mesh(bladeGeo, bladeMat);
    const angle = (i / 6) * Math.PI * 2;
    blade.position.set(Math.cos(angle) * 1.25, 0, Math.sin(angle) * 1.25);
    blade.rotation.y = -angle + 0.5;                      // 叶片安装角
    impeller.add(blade);
  }
  impeller.position.y = 2.6;
  group.add(impeller);

  // 进风口(喇叭口)与顶部排风管
  const inlet = new THREE.Mesh(new THREE.CylinderGeometry(1.0, 0.7, 0.9, 24, 1, true),
    new THREE.MeshStandardMaterial({ color: 0x90a4ae, side: THREE.DoubleSide, metalness: 0.6, roughness: 0.4 }));
  inlet.position.set(0, 2.6, 2.3);
  inlet.rotation.x = Math.PI / 2;
  group.add(inlet);
  const duct = new THREE.Mesh(new THREE.CylinderGeometry(0.8, 0.8, 2.2, 24), ductMat);
  duct.position.set(0, 4.9, 0);
  group.add(duct);

  // 桁架支腿
  const legMat = new THREE.MeshStandardMaterial({ color: 0x546e7a, metalness: 0.6, roughness: 0.4 });
  [[-2.2, -1.6], [2.2, -1.6], [-2.2, 1.6], [2.2, 1.6]].forEach(([x, z]) => {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.3, 1.6, 0.3), legMat);
    leg.position.set(x, 0.8, z);
    group.add(leg);
  });

  return { group, spinning: [impeller], bodyMeshes: [shell, duct] };
}

/**
 * 建模: 齿轮传动箱(两级箱体 + 大小啮合齿轮 + 联轴器 + 油位视镜)
 * 齿轮用带轮齿感的圆柱(周向小方齿)表达, 动画中两齿轮反向旋转。
 */
function buildGearbox() {
  const group = new THREE.Group();
  group.add(makeBaseplate(5.8, 4.0));

  const housingMat = new THREE.MeshStandardMaterial({ color: 0x2ecc71, roughness: 0.55, metalness: 0.45 });
  const gearMat = new THREE.MeshStandardMaterial({ color: 0xf0c419, metalness: 0.85, roughness: 0.25 });

  // 箱体(主箱 + 上盖)
  const housing = new THREE.Mesh(new THREE.BoxGeometry(4.6, 2.8, 3.2), housingMat);
  housing.position.y = 1.8;
  housing.castShadow = true;
  group.add(housing);
  const cover = new THREE.Mesh(new THREE.BoxGeometry(3.6, 0.7, 2.4),
    new THREE.MeshStandardMaterial({ color: 0x37474f, roughness: 0.5, metalness: 0.5 }));
  cover.position.y = 3.5;
  group.add(cover);

  // 齿轮构建函数: 圆柱基体 + 周向轮齿
  const makeGear = (radius, teeth) => {
    const gear = new THREE.Group();
    const disk = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 0.4, 32), gearMat);
    gear.add(disk);
    const toothGeo = new THREE.BoxGeometry(0.22, 0.42, 0.22);
    for (let i = 0; i < teeth; i++) {
      const tooth = new THREE.Mesh(toothGeo, gearMat);
      const angle = (i / teeth) * Math.PI * 2;
      tooth.position.set(Math.cos(angle) * (radius + 0.1), 0, Math.sin(angle) * (radius + 0.1));
      tooth.rotation.y = -angle;
      gear.add(tooth);
    }
    // 中心轴孔装饰
    const axle = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 1.6, 12),
      new THREE.MeshStandardMaterial({ color: 0xb0bec5, metalness: 0.8, roughness: 0.3 }));
    gear.add(axle);
    return gear;
  };

  // 大小两个齿轮(啮合布置: 大齿轮驱动小齿轮, 转速不同步体现传动比)
  const bigGear = makeGear(1.25, 18);
  bigGear.position.set(-0.7, 1.9, 0);
  const smallGear = makeGear(0.75, 12);
  smallGear.position.set(1.35, 1.9, 0);
  group.add(bigGear, smallGear);

  // 两侧联轴器
  const couplingMat = new THREE.MeshStandardMaterial({ color: 0x8d6e63, roughness: 0.6, metalness: 0.3 });
  [-2.7, 2.7].forEach((x) => {
    const coupling = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 0.8, 20), couplingMat);
    coupling.rotation.z = Math.PI / 2;
    coupling.position.set(x, 1.9, 0);
    group.add(coupling);
  });

  // 油位视镜(小圆片, 紧急状态时会随整体变红)
  const gauge = new THREE.Mesh(new THREE.CircleGeometry(0.3, 20),
    new THREE.MeshStandardMaterial({ color: 0x4fc3f7, emissive: 0x1a3a4a, roughness: 0.2 }));
  gauge.position.set(0, 1.3, 1.62);
  group.add(gauge);

  return { group, spinning: [bigGear, smallGear], bodyMeshes: [housing, cover] };
}

/**
 * 按类型分发建模, 并挂载铭牌/信标/阴影
 */
function createDevice(cfg) {
  const built = cfg.type === 'motor' ? buildMotor()
             : cfg.type === 'fan'   ? buildFan()
             : buildGearbox();
  built.group.position.set(cfg.x, 0, cfg.z);

  // 顶部铭牌标签(设备名 + 编号)
  const label = makeTextSprite(cfg.name, cfg.id);
  label.position.set(cfg.x, 6.8, cfg.z);
  scene.add(label);

  // 状态信标(置于设备上方)
  const beacon = makeBeacon();
  beacon.position.set(cfg.x + 2.2, 3.8, cfg.z + 1.4);
  scene.add(beacon);

  built.group.traverse((obj) => { if (obj.isMesh) { obj.userData.deviceId = cfg.id; } });
  scene.add(built.group);

  runtimeDevices[cfg.id] = {
    id: cfg.id, name: cfg.name, type: cfg.type,
    group: built.group, spinning: built.spinning,
    bodyMeshes: built.bodyMeshes, beacon,
    health: 100, spinPhase: Math.random() * Math.PI * 2,
    baseSpinSpeed: cfg.type === 'fan' ? 4.0 : (cfg.type === 'motor' ? 9.0 : 2.2)
  };
}

/* ------------------------------ 健康状态应用 ------------------------------ */

/**
 * 将新健康评分应用到设备: 更新本体颜色 / 信标 / 面板刷新
 */
function applyHealth(deviceId, health) {
  const dev = runtimeDevices[deviceId];
  if (!dev || health === undefined || health === null) return;
  dev.health = Math.max(0, Math.min(100, Number(health)));
  const color = healthToColor(dev.health);

  // 本体着色(保留原始金属质感, 仅改 diffuse)
  dev.bodyMeshes.forEach((mesh) => { mesh.material.color.copy(color); mesh.material.emissive.setScalar(0.05); });

  // 信标灯与点光源颜色
  dev.beacon.userData.bulb.material.color.copy(color);
  dev.beacon.userData.bulb.material.emissive.copy(color);
  dev.beacon.userData.light.color.copy(color);

  // 悬停/选中状态的高亮强度(在动画循环里根据该标记调制)
  dev.highlight = dev.id === selectedDeviceId || dev.id === hoveredDeviceId;
  if (dev.id === selectedDeviceId) refreshDetailPanel(dev);
}

/* ------------------------------ 交互: 拾取与面板 ------------------------------ */

/**
 * 初始化鼠标事件: 点击选中设备并弹出详情面板, 悬停高亮
 */
function initInteraction(container) {
  const onPointerMove = (event) => {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const meshes = [];
    Object.values(runtimeDevices).forEach((d) => d.group.traverse((o) => { if (o.isMesh) meshes.push(o); }));
    const hits = raycaster.intersectObjects(meshes);
    const newHover = hits.length ? hits[0].object.userData.deviceId : null;
    if (newHover !== hoveredDeviceId) {
      // 悬停设备变化时刷新一次高亮标记
      hoveredDeviceId = newHover;
      Object.values(runtimeDevices).forEach((d) => { d.highlight = d.id === selectedDeviceId || d.id === hoveredDeviceId; });
      container.style.cursor = newHover ? 'pointer' : 'default';
    }
  };

  const onClick = (event) => {
    onPointerMove(event);
    if (hoveredDeviceId) {
      selectedDeviceId = hoveredDeviceId;
      openDetailPanel(runtimeDevices[selectedDeviceId]);
    } else {
      selectedDeviceId = null;
      closeDetailPanel();
    }
  };

  renderer.domElement.addEventListener('mousemove', onPointerMove);
  renderer.domElement.addEventListener('click', onClick);
}

/**
 * 打开/关闭/刷新 详情面板(DOM 覆盖层)
 */
function openDetailPanel(dev) { ensurePanel().style.display = 'block'; refreshDetailPanel(dev); }
function closeDetailPanel() { const p = document.getElementById('twin-detail-panel'); if (p) p.style.display = 'none'; }

function ensurePanel() {
  let panel = document.getElementById('twin-detail-panel');
  if (panel) return panel;
  panel = document.createElement('div');
  panel.id = 'twin-detail-panel';
  panel.style.cssText = [
    'position:absolute', 'right:16px', 'top:16px', 'width:320px', 'z-index:20',
    'background:rgba(13,20,32,0.92)', 'border:1px solid #2f4a6b', 'border-radius:10px',
    'padding:14px 16px', 'color:#e8f1fa', 'font-size:13px', 'line-height:1.7',
    'box-shadow:0 6px 24px rgba(0,0,0,0.5)', 'display:none'
  ].join(';');
  panel.innerHTML = '<div id="twin-detail-body">加载中 ...</div>' +
    '<div style="text-align:right;margin-top:8px">' +
    '<button id="twin-detail-close" style="background:#1d3350;border:1px solid #3d5a80;color:#cfe3f7;' +
    'border-radius:6px;padding:3px 12px;cursor:pointer">关闭</button></div>';
  document.getElementById('twin-3d-container').appendChild(panel);
  document.getElementById('twin-detail-close').onclick = () => { selectedDeviceId = null; closeDetailPanel(); };
  return panel;
}

/**
 * 刷新详情面板内容: 优先拉取看板 /api/prediction/<id>(含 RUL 与故障诊断)
 */
function refreshDetailPanel(dev) {
  const body = document.getElementById('twin-detail-body');
  if (!body) return;
  const score = dev.health.toFixed(1);
  const colorCss = '#' + healthToColor(dev.health).getHexString();
  body.innerHTML =
    '<div style="font-size:15px;font-weight:bold;margin-bottom:6px">' + dev.name +
    ' <span style="color:#7fa8c9">(' + dev.id + ')</span></div>' +
    '<div>健康评分: <b style="color:' + colorCss + ';font-size:18px">' + score + '</b> / 100' +
    ' <span class="twin-status">[' + healthToStatus(dev.health) + ']</span></div>' +
    '<div id="twin-detail-extra">预测数据加载中 ...</div>';

  // 看板模式: 请求预测接口补充 RUL / 故障诊断 / 建议
  if (!standaloneMode) {
    fetch('/api/prediction/' + encodeURIComponent(dev.id))
      .then((r) => r.json())
      .then((data) => {
        const extra = document.getElementById('twin-detail-extra');
        if (!extra) return;
        // /api/prediction/<id> 返回 {prediction, fault, twin}, 预测指标在 prediction 子对象里
        const pred = data.prediction || {};
        const rul = pred.rul || {};
        const rulText = rul.rul_hours !== null && rul.rul_hours !== undefined
          ? rul.rul_hours.toFixed(1) + ' 小时 (80%CI: ' + rul.rul_ci_low.toFixed(1) + ' ~ ' + rul.rul_ci_high.toFixed(1) + ')'
          : '暂不可估(' + (rul.trend || '数据不足') + ')';
        const fault = data.fault && !data.fault.ruled_out
          ? data.fault.matched[0].name + '(置信度 ' + Math.round(data.fault.matched[0].confidence * 100) + '%)'
          : '暂无显著故障特征';
        extra.innerHTML =
          '<div>剩余寿命 RUL: <b>' + rulText + '</b></div>' +
          '<div>异常评分: ' + (pred.anomaly_score !== undefined ? pred.anomaly_score : '-') + ' / 100</div>' +
          '<div>疑似故障: ' + fault + '</div>' +
          (data.fault && !data.fault.ruled_out
            ? '<div style="margin-top:4px">处置建议: ' + data.fault.matched[0].actions[0] + '</div>'
            : '');
      })
      .catch(() => {
        const extra = document.getElementById('twin-detail-extra');
        if (extra) extra.innerHTML = '<div style="color:#e74c3c">预测服务不可用</div>';
      });
  } else {
    document.getElementById('twin-detail-extra').innerHTML =
      '<div style="color:#8899aa">(独立演示模式: 启动 dashboard 后可查看 RUL 与故障诊断)</div>';
  }
}

/* ------------------------------ 数据轮询 ------------------------------ */

/**
 * 拉取全部设备健康数据并应用到 3D 场景;
 * file:// 直开或接口异常时进入内置演示模式(健康度缓慢正弦波动)。
 */
function pollHealth() {
  if (standaloneMode) { demoTick(); return; }
  fetch('/api/devices')
    .then((r) => r.json())
    .then((data) => {
      data.devices.forEach((d) => applyHealth(d.device_id, d.health_score));
    })
    .catch(() => { standaloneMode = true; demoTick(); });
}

/** 独立演示模式: 用正弦+噪声合成每台设备的健康度波动 */
function demoTick() {
  const t = clock.getElapsedTime() / 20;
  CONFIG.devices.forEach((cfg, i) => {
    const score = 55 + 45 * Math.sin(t + i * 2.1);
    applyHealth(cfg.id, score);
  });
}

/* ------------------------------ 动画主循环 ------------------------------ */

/**
 * 渲染循环: 设备运转动画(速度随健康衰减) + 信标呼吸 + 相机阻尼
 */
function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();
  const time = clock.getElapsedTime();

  Object.values(runtimeDevices).forEach((dev) => {
    // 转速 = 基准转速 * (0.3 + 0.7 * 健康度/100): 退化越重转得越慢
    const speed = dev.baseSpinSpeed * (0.3 + 0.7 * dev.health / 100);
    dev.spinPhase += speed * delta;
    if (dev.type === 'motor') {
      dev.spinning[0].rotation.x = dev.spinPhase;                 // 电机轴绕 X
    } else if (dev.type === 'fan') {
      dev.spinning[0].rotation.y = dev.spinPhase;                 // 叶轮绕 Y
    } else {
      dev.spinning[0].rotation.y = dev.spinPhase;                 // 大齿轮
      dev.spinning[1].rotation.y = -dev.spinPhase * 1.5;          // 小齿轮反向+传动比
    }

    // 信标呼吸闪烁: 健康时慢闪, 紧急时快闪
    const critical = dev.health < 55;
    const blink = 0.5 + 0.5 * Math.sin(time * (critical ? 6 : 2));
    dev.beacon.userData.bulb.material.emissiveIntensity = 0.6 + blink * (critical ? 1.6 : 0.5);
    dev.beacon.userData.light.intensity = 0.3 + blink * (critical ? 1.2 : 0.3);

    // 高亮调制: 悬停/选中的设备本体自发光增强
    const glow = dev.highlight ? 0.35 : 0.05;
    dev.bodyMeshes.forEach((m) => { m.material.emissiveIntensity = glow; });
  });

  controls.update();
  renderer.render(scene, camera);
}

/**
 * 窗口尺寸自适应
 */
function onResize() {
  const container = document.getElementById('twin-3d-container');
  if (!container) return;
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

/* ------------------------------ 启动入口 ------------------------------ */

/**
 * 初始化整个 3D 场景(由宿主页面在 DOMContentLoaded 后调用)
 */
function initTwinScene() {
  const container = document.getElementById('twin-3d-container');
  if (!container || typeof THREE === 'undefined') {
    console.error('[3D场景] 未找到容器 #twin-3d-container 或未加载 three.js');
    return;
  }
  // file:// 直接打开时无后端, 自动进入演示模式
  standaloneMode = window.location.protocol === 'file:';

  initRenderer(container);
  buildEnvironment();
  CONFIG.devices.forEach((cfg) => createDevice(cfg));
  initInteraction(container);
  window.addEventListener('resize', onResize);

  pollHealth();
  setInterval(pollHealth, CONFIG.pollInterval);
  animate();
  console.log('[3D场景] 数字孪生车间初始化完成, 设备数:', CONFIG.devices.length,
    standaloneMode ? '(独立演示模式)' : '(看板模式)');
}

// 兼容两种加载方式: DOM 就绪后自动启动
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initTwinScene);
} else {
  initTwinScene();
}
