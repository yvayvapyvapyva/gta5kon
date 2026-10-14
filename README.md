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

- `tools/convert.py` — парсер RenderWare DFF (фреймы, геометрии, материалы/MatFX,
  specular 0x253F2F6, reflection 0x253F2FC) + TXD (DXT1-декодер на numpy) →
  единый `model.glb` (26 МБ, ~940k треугольников, 40 материалов, 16 текстур).
  Координаты GTA (Z-up) переведены в Three.js (Y-up), применены трансляции
  фреймов (колёса, бамперы на местах). Материалы как в игре:
  краска/хром (env MatFX → металлик + clearcoat-лак), стёкла (альфа 102–190 →
  BLEND + стекло), фары/фонари (vehiclelights128 → emissive), specular
  vehiclespecdot64 → roughness.
- `web/index.html` / `app.js` / `styles.css` — viewer на Three.js (CDN):
  OrbitControls, автовращение, wireframe, сетка, тени, RoomEnvironment-отражения,
  смена цвета кузова, загрузка своего .glb, статистика (треугольники/вершины/габариты).
  Габариты: ~2.08 × 1.68 × 4.72 м.

## Требования

- Python 3 + `numpy` + `Pillow` (для конвертера)
- Браузер + интернет (Three.js грузится с unpkg CDN)
- `model.glb` уже лежит в `web/` — заново конвертировать не нужно
