# Renault Symbol 2013 — просмотр 3D-модели из GTA

В папке `reno/` лежат `sentinel.dff` + `sentinel.txd` (RenderWare, GTA SA).
Файл называется sentinel, но внутри — **Renault Symbol 2013** (59 фреймов, 37 геометрий).

Веб-приложение: `web/` — Three.js viewer для сконвертированной модели.

## Быстрый старт

```bash
# 1. Конвертация DFF+TXD -> GLB (уже выполнена, model.glb на месте)
python3 tools/convert.py --dff reno/sentinel.dff --txd reno/sentinel.txd --out web/model.glb

# 2. Запуск просмотрщика
python3 -m http.server 8000 --directory web
# открыть http://localhost:8000
```

## Что сделано

- `tools/convert.py` — парсер RenderWare DFF (фреймы **с именами** — распознан
  фрейм `руль`, `wheel_sr6`, `wheel_*_dummy`, 59 шт.; геометрии, материалы/MatFX,
  specular 0x253F2F6, reflection 0x253F2FC) + TXD (DXT1-декодер на numpy) →
  иерархический `model.glb` (27 МБ: `chassis` — статика с запечёнными
  трансляциями; `steering` — руль с пивотом в центре; `wheel_{rf,lf,rb,lb}_steer`
  → `wheel_*_spin` — колёса, левая сторона отзеркалена; всего ~950k
  треугольников, 40 материалов, 17 текстур).
  Координаты GTA (Z-up) переведены в Three.js (Y-up). Материалы как в игре:
  краска/хром (env MatFX → металлик + clearcoat-лак), стёкла (альфа 102–190 →
  BLEND + стекло), фары/фонари (vehiclelights128 → emissive), specular
  vehiclespecdot64 → roughness.
- `web/index.html` / `app.js` / `styles.css` — viewer на Three.js (CDN):
  OrbitControls, сетка, тени, RoomEnvironment-отражения,
  смена цвета кузова, загрузка своего .glb, статистика (треугольники/вершины/габариты).
  Габариты: ~2.08 × 1.68 × 4.72 м.
- Ходовая **как в GTA SA** с управлением **W/S/A/D**: `W/S` — вращение всех
  4 колёс вокруг ступицы (вперёд/назад),   `A/D` — поворот только передних
  `wheel_rf/lf_dummy` (yaw ±35°, с возвратом в центр), руль в салоне
  (фрейм `руль`) всегда повторяет поворот: полтора оборота (540°)
  при полном вывороте вокруг измеренной оси колонки (PCA-нормаль
  к плоскости руля, наклон ~25° — кручение вокруг «прямой» оси давало
  конусное биение до 13 см); пресеты камеры «Кузов / Колесо / Салон».

## Требования

- Python 3 + `numpy` + `Pillow` (для конвертера)
- Браузер + интернет (Three.js грузится с unpkg CDN)
- `model.glb` уже лежит в `web/` — заново конвертировать не нужно
