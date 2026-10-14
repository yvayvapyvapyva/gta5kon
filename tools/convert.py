#!/usr/bin/env python3
"""Конвертер GTA SA DFF+TXD (RenderWare) -> GLB для веб-просмотрщика.
Использование:
    python3 tools/convert.py --dff path/to/car.dff --txd path/to/car.txd --out path/to/car.glb
"""
import argparse
import struct
import os
import sys
import json
import io
from collections import defaultdict

import numpy as np
from PIL import Image


# ---------- RW version ----------
def rw_version(library_id):
    if library_id & 0xFFFF0000:
        return ((library_id >> 14) & 0x3FF00) + 0x30000 | ((library_id >> 16) & 0x3F)
    return library_id << 8


# ---------- TXD / DXT1 ----------
def decode_dxt1(data: bytes, w: int, h: int):
    """Декодирует DXT1 в RGBA numpy (h, w, 4) uint8. Векторизованно."""
    bw = w // 4
    bh = h // 4
    nb = bw * bh
    expected = nb * 8
    if len(data) < expected:
        raise ValueError(f"DXT1 data too short: {len(data)} < {expected}")
    raw = np.frombuffer(data[:expected], dtype=np.uint8).reshape(nb, 8)
    c0 = (raw[:, 1].astype(np.uint16) << 8) | raw[:, 0].astype(np.uint16)
    c1 = (raw[:, 3].astype(np.uint16) << 8) | raw[:, 2].astype(np.uint16)
    idx32 = (raw[:, 7].astype(np.uint32) << 24) | (raw[:, 6].astype(np.uint32) << 16) | \
            (raw[:, 5].astype(np.uint32) << 8) | raw[:, 4].astype(np.uint32)

    def dec565(c):
        r = (c >> 11) & 31
        g = (c >> 5) & 63
        b = c & 31
        R = (r * 255 + 15) // 31
        G = (g * 255 + 31) // 63
        B = (b * 255 + 15) // 31
        return R.astype(np.uint8), G.astype(np.uint8), B.astype(np.uint8)

    r0, g0, b0 = dec565(c0)
    r1, g1, b1 = dec565(c1)

    four = c0 > c1  # (nb,) bool

    # c2, c3
    r2 = np.where(four, (2 * r0.astype(np.uint16) + r1.astype(np.uint16)) // 3,
                  (r0.astype(np.uint16) + r1.astype(np.uint16)) // 2).astype(np.uint8)
    g2 = np.where(four, (2 * g0.astype(np.uint16) + g1.astype(np.uint16)) // 3,
                  (g0.astype(np.uint16) + g1.astype(np.uint16)) // 2).astype(np.uint8)
    b2 = np.where(four, (2 * b0.astype(np.uint16) + b1.astype(np.uint16)) // 3,
                  (b0.astype(np.uint16) + b1.astype(np.uint16)) // 2).astype(np.uint8)
    r3 = np.where(four, (r0.astype(np.uint16) + 2 * r1.astype(np.uint16)) // 3,
                  np.zeros_like(r0, dtype=np.uint16)).astype(np.uint8)
    g3 = np.where(four, (g0.astype(np.uint16) + 2 * g1.astype(np.uint16)) // 3,
                  np.zeros_like(g0, dtype=np.uint16)).astype(np.uint8)
    b3 = np.where(four, (b0.astype(np.uint16) + 2 * b1.astype(np.uint16)) // 3,
                  np.zeros_like(b0, dtype=np.uint16)).astype(np.uint8)

    # палитра (nb, 4, 4)
    pal = np.empty((nb, 4, 4), dtype=np.uint8)
    pal[:, 0, 0] = r0; pal[:, 0, 1] = g0; pal[:, 0, 2] = b0; pal[:, 0, 3] = 255
    pal[:, 1, 0] = r1; pal[:, 1, 1] = g1; pal[:, 1, 2] = b1; pal[:, 1, 3] = 255
    pal[:, 2, 0] = r2; pal[:, 2, 1] = g2; pal[:, 2, 2] = b2; pal[:, 2, 3] = 255
    alpha3 = np.where(four, 255, 0).astype(np.uint8)
    pal[:, 3, 0] = r3; pal[:, 3, 1] = g3; pal[:, 3, 2] = b3; pal[:, 3, 3] = alpha3

    # индексы (nb, 16): texel k (0..15), x=k%4, y=k//4, bits = (idx32 >> (k*2)) & 3
    k = np.arange(16, dtype=np.uint32)
    shifts = (k * 2).astype(np.uint32)  # (16,)
    indices = ((idx32[:, None] >> shifts[None, :]) & 3).astype(np.uint8)  # (nb,16)

    # выбираем цвета
    # blocks: (nb,16,4)
    nb_idx = np.arange(nb)[:, None, None]
    # palIdx: (nb,16) -> gather
    cols = pal[nb_idx.squeeze(-1) if False else np.arange(nb)[:, None], indices]  # (nb,16,4)

    img = np.empty((h, w, 4), dtype=np.uint8)
    # раскладываем блоки
    # block (bx,by) -> pixels [by*4:(by+1)*4, bx*4:(bx+1)*4]
    # cols[b, y*4+x]
    for y in range(4):
        for x in range(4):
            t = y * 4 + x
            # все блоки сразу
            patch = cols[:, t, :]  # (nb,4)
            # reshape в (bh,bw,4)
            patch_img = patch.reshape(bh, bw, 4)
            img[y::4, x::4, :] = patch_img.reshape(bh, bw, 4) if False else np.repeat(
                np.repeat(patch.reshape(bh, bw, 4), 1, axis=0), 1, axis=1
            ) if False else patch.reshape(bh, bw, 4)[:, :, :]
            # Проще: img[by*4+y, bx*4+x] = patch[by*bw+bx]
            # векторно:
    # Выше цикл даёт неверный порядок из-за reshape? Перепишем явно векторно:
    # (делаем заново корректно)
    cols4 = cols.reshape(bh, bw, 4, 4, 4)  # (bh,bw,y,x,rgba)? нет, cols (nb,16,4), 16=y*4+x
    cols4 = cols.reshape(bh, bw, 4, 4, 4)
    for y in range(4):
        for x in range(4):
            img[y::4, x::4] = cols4[:, :, y, x]
    return img


def parse_txd(path):
    return parse_txd_data(open(path, 'rb').read())


def parse_txd_data(data: bytes):
    # пропускаем root + struct count
    # root header 12
    t, s, v = struct.unpack_from('<III', data, 0)
    assert t == 0x16, f"not TXD, root type {hex(t)}"
    pos = 12
    # struct count
    t2, s2, v2 = struct.unpack_from('<III', data, pos)
    pos += 12 + s2
    textures = []
    idx = 0
    while pos + 12 <= len(data):
        a, b, c = struct.unpack_from('<III', data, pos)
        if a == 0x03:  # extension в конце
            break
        if a != 0x15:
            # пропускаем неизвестное
            if b <= 0 or b > len(data):
                break
            pos += 12 + b
            continue
        # TextureNative
        a2, b2, c2 = struct.unpack_from('<III', data, pos + 12)
        d_off = pos + 24
        ver = struct.unpack_from('<I', data, d_off)[0]
        name = data[d_off + 8:d_off + 40].split(b'\x00')[0].decode('ascii', errors='ignore')
        off = 72
        base = d_off
        alpha_flags = struct.unpack_from('<I', data, base + off)[0]
        d3d = data[base + off + 4:base + off + 8]
        w, h = struct.unpack_from('<HH', data, base + off + 8)
        depth, mips, tcode, flg = struct.unpack_from('<BBBB', data, base + off + 12)
        ds = struct.unpack_from('<I', data, base + off + 16)[0]
        img_data = data[base + off + 20:base + off + 20 + ds]
        textures.append({
            'name': name, 'w': w, 'h': h, 'd3d': d3d, 'depth': depth,
            'mips': mips, 'data': img_data, 'alpha_flags': alpha_flags,
        })
        pos += 12 + b
        idx += 1
    return textures


def txd_to_images(textures, out_dir=None):
    """Декодирует текстуры в PIL Image + определяет прозрачность. Возвращает dict name -> (bytes_png_or_jpg, mime)."""
    result = {}
    for t in textures:
        name = t['name']
        w, h, d3d = t['w'], t['h'], t['d3d']
        try:
            if d3d == b'DXT1':
                img = decode_dxt1(t['data'], w, h)
            elif d3d == b'DXT3':
                raise NotImplementedError("DXT3 not in this file")
            else:
                print(f"[txd] пропуск {name}: неизвестный формат {d3d!r}, будет белый")
                continue
        except Exception as e:
            print(f"[txd] ошибка декодирования {name}: {e}")
            continue
        has_transparent = bool(np.any(img[:, :, 3] < 128))
        pil = Image.fromarray(img, 'RGBA')
        buf = io.BytesIO()
        if has_transparent:
            pil.save(buf, format='PNG', optimize=True)
            mime = 'image/png'
        else:
            pil.convert('RGB').save(buf, format='JPEG', quality=90)
            mime = 'image/jpeg'
        result[name] = (buf.getvalue(), mime, has_transparent, (w, h))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            ext = 'png' if has_transparent else 'jpg'
            with open(os.path.join(out_dir, f"{name}.{ext}"), 'wb') as f:
                f.write(buf.getvalue())
        print(f"[txd] {name}: {w}x{h} {'alpha' if has_transparent else 'opaque'} -> {len(buf.getvalue())//1024}KB")
    return result


# ---------- DFF ----------
def parse_material_full(mat_raw: bytes):
    """Полный разбор чанка Material: цвет/альфа, базовая текстура,
    MatFX env/dual, specular level, reflection intensity."""
    tex_names = []

    def walk(buf, start, end):
        p = start
        while p + 12 <= end:
            ct, cs, cv = struct.unpack_from('<III', buf, p)
            cend = p + 12 + cs
            if cend > end or cs < 0:
                break
            if ct == 0x06:  # Texture
                q = p + 12
                qend = cend
                strs = []
                while q + 12 <= qend:
                    st, ss, sv = struct.unpack_from('<III', buf, q)
                    send = q + 12 + ss
                    if send > qend:
                        break
                    if st == 0x02:  # String
                        s = buf[q + 12:send].split(b'\x00')[0].decode('ascii', errors='ignore')
                        strs.append(s)
                    q = send
                if strs and strs[0]:
                    tex_names.append(strs[0])
            elif ct in (0x03, 0x120):  # Extension или MatFX — рекурсия
                walk(buf, p + 12, cend)
            elif ct == 0x01 or ct == 0x02:
                pass
            else:
                if cs > 12 and cs < 100000:
                    walk(buf, p + 12, cend)
            p = cend

    walk(mat_raw, 0, len(mat_raw))
    base_tex = tex_names[0] if tex_names else None

    # specular / reflection лежат в Extension материала (0x03)
    spec_lvl, spec_tex, refl_i, matfx = 0.0, '', 0.0, 0
    p = 0
    # ищем чанк Extension внутри Material
    while p + 12 <= len(mat_raw):
        ct, cs, cv = struct.unpack_from('<III', mat_raw, p)
        cend = p + 12 + cs
        if cend > len(mat_raw) or cs < 0:
            break
        if ct == 0x03:
            q = p + 12
            while q + 12 <= cend:
                sc, ss, sv = struct.unpack_from('<III', mat_raw, q)
                send = q + 12 + ss
                if send > cend:
                    break
                sub = mat_raw[q + 12:send]
                if sc == 0x120 and len(sub) >= 4:  # Material Effects PLG
                    et = struct.unpack_from('<I', sub, 0)[0]
                    matfx = et
                elif sc == 0x253F2F6 and len(sub) >= 28:  # Specular
                    spec_lvl = round(struct.unpack_from('<f', sub, 0)[0], 3)
                    spec_tex = sub[4:28].split(b'\x00')[0].decode('ascii', errors='ignore')
                elif sc == 0x253F2FC and len(sub) >= 20:  # Reflection
                    vals = struct.unpack_from('<5f', sub, 0)
                    refl_i = round(float(vals[4]), 3)
                q = send
        p = cend
    return {'texture': base_tex, 'matfx': matfx,
            'spec_lvl': spec_lvl, 'spec_tex': spec_tex, 'refl_i': refl_i}


def parse_material_texture_name(mat_data: bytes, mat_end_rel: int = None):
    m = parse_material_full(mat_data)
    return m['texture']


def classify_material(m):
    """GTA SA vehicle material -> PBR параметры как в игре.
    Возвращает dict для glTF: metallic, roughness, alphaMode, opacity,
    emissive, clearcoat..."""
    r, g, b, a = m.get('color', (255, 255, 255, 255))
    tex = m.get('texture')
    matfx = m.get('matfx', 0)
    spec = min(max(m.get('spec_lvl', 0.0), 0.0), 1.0)
    refl = m.get('refl_i', 0.0)
    alpha = a / 255.0
    is_textured = tex is not None
    has_env = (matfx == 2)
    is_glass = alpha < 0.98
    is_light_tex = (tex == 'vehiclelights128')
    bright_lens = (r, g, b) in ((128, 0, 0), (255, 0, 0), (255, 60, 0),
                                (0, 255, 200), (255, 128, 64), (255, 255, 0))

    out = {'metallic': 0.05, 'roughness': 0.85, 'alphaMode': 'OPAQUE',
           'opacity': 1.0, 'emissive': [0, 0, 0], 'emissiveTex': False,
           'clearcoat': 0.0, 'kind': 'std'}
    if is_light_tex:
        # фары/фонари: светятся своей текстурой + цветом
        out.update({'metallic': 0.1, 'roughness': 0.25,
                    'emissive': [r / 255.0 or 0.6, g / 255.0 or 0.6, b / 255.0 or 0.6],
                    'emissiveTex': True, 'clearcoat': 0.6, 'kind': 'light'})
    elif is_glass:
        # стёкла и прозрачные рассеиватели
        if bright_lens:
            out.update({'metallic': 0.2, 'roughness': 0.15,
                        'emissive': [r / 255.0 * 0.7, g / 255.0 * 0.7, b / 255.0 * 0.7],
                        'clearcoat': 1.0, 'kind': 'lens'})
        else:
            out.update({'metallic': 0.9, 'roughness': 0.06,
                        'clearcoat': 1.0, 'kind': 'glass'})
        out['alphaMode'] = 'BLEND'
        out['opacity'] = max(alpha, 0.15)
    elif has_env and not is_textured:
        # краска / хром — как в игре: env-отражение + лак.
        # Краска — ДИЭЛЕКТРИК (metallic 0): глянец даёт лак (clearcoat),
        # иначе любой светлый цвет выглядит хромом-зеркалом.
        # Хром — настоящий металл (metallic 1).
        bright = (r + g + b) / 3.0 / 255.0
        chrome_like = bright > 0.55 and abs(r - g) < 12 and abs(g - b) < 12
        if chrome_like:
            out.update({'metallic': 1.0, 'roughness': 0.12,
                        'clearcoat': 1.0, 'kind': 'chrome'})
        else:
            rough = 0.42 - min(refl, 0.5) * 0.3 - spec * 0.1
            out.update({'metallic': 0.0,
                        'roughness': round(max(0.2, rough), 3),
                        'clearcoat': 1.0, 'kind': 'paint'})
    elif is_textured:
        # салон/текстуры: матовые, но specular из игры даёт блик
        out.update({'metallic': 0.0,
                    'roughness': round(max(0.25, 0.85 - spec * 0.6), 3),
                    'kind': 'tex'})
    else:
        # матовый пластик/резина без env
        out.update({'metallic': 0.25,
                    'roughness': round(max(0.3, 0.65 - spec * 0.5), 3),
                    'kind': 'plastic'})
        if bright_lens:  # красные катафоты, поворотники без альфы
            out['emissive'] = [r / 255.0 * 0.5, g / 255.0 * 0.5, b / 255.0 * 0.5]
    return out


def parse_dff(path):
    return parse_dff_data(open(path, 'rb').read())


def parse_dff_data(data: bytes):
    mv = memoryview(data)
    # root clump
    t, s, v = struct.unpack_from('<III', data, 0)
    assert t == 0x10, f"not DFF clump: {hex(t)}"
    lib_id = v
    version = rw_version(lib_id)
    print(f"[dff] clump size {s}, rw_version {hex(version)}")
    pos = 12
    end = 12 + s
    # clump struct
    t1, s1, v1 = struct.unpack_from('<III', data, pos)
    assert t1 == 0x01
    num_atomics = struct.unpack_from('<I', data, pos + 12)[0]
    print(f"[dff] num_atomics(по clump) = {num_atomics}")
    pos += 12 + s1
    # FrameList
    t2, s2, v2 = struct.unpack_from('<III', data, pos)
    assert t2 == 0x0E, f"expected FrameList, got {hex(t2)}"
    fl_end = pos + 12 + s2
    fpos = pos + 12
    # framelist struct
    ft, fs, fv = struct.unpack_from('<III', data, fpos)
    assert ft == 0x01
    nframes = struct.unpack_from('<I', data, fpos + 12)[0]
    print(f"[dff] frames: {nframes}")
    frames = []
    fp = fpos + 12 + 4
    for i in range(nframes):
        right = struct.unpack_from('<3f', data, fp)
        up = struct.unpack_from('<3f', data, fp + 12)
        at = struct.unpack_from('<3f', data, fp + 24)
        pxyz = struct.unpack_from('<3f', data, fp + 36)
        parent, flags = struct.unpack_from('<2i', data, fp + 48)
        frames.append({'right': right, 'up': up, 'at': at, 'pos': pxyz, 'parent': parent})
        fp += 56
    # мировые трансляции (сумма по родителям; вращения единичные)
    world = [None] * nframes
    # построение матриц 3x3 для корректности
    R = []
    for f in frames:
        Rm = np.array([f['right'], f['up'], f['at']], dtype=np.float64)  # строки?
        # RW хранит right/up/at как строки? Для identity неважно.
        # Транспонируем к column-major: мир = parent @ local
        R.append(Rm.T)  # пробуем как столбцы
    W_R = [None] * nframes
    W_T = [None] * nframes
    for i in range(nframes):
        par = frames[i]['parent']
        lt = np.array(frames[i]['pos'], dtype=np.float64)
        lr = R[i]
        if par < 0:
            W_R[i] = lr
            W_T[i] = lt
        else:
            W_R[i] = W_R[par] @ lr
            W_T[i] = W_R[par] @ lt + W_T[par]
    # --- имена фреймов (FrameList extensions 0x03 -> 0x0253F2FE) ---
    # Имя может быть в cp1251 (напр. b'\xf0\xf3\xeb\xfc' = "руль").
    frame_names = []
    try:
        fp_ext = fpos + 12 + fs
        for _ in range(nframes):
            if fp_ext + 12 > fl_end:
                break
            et, es, ev = struct.unpack_from('<III', data, fp_ext)
            eend = fp_ext + 12 + es
            name = ''
            if et == 0x03:
                q = fp_ext + 12
                while q + 12 <= eend:
                    st, ss, sv = struct.unpack_from('<III', data, q)
                    send = q + 12 + ss
                    if send > eend:
                        break
                    if st == 0x0253F2FE and ss > 0:
                        raw = data[q + 12:send].split(b'\x00')[0]
                        for enc in ('ascii', 'cp1251', 'utf-8'):
                            try:
                                name = raw.decode(enc)
                                break
                            except Exception:
                                continue
                        # ascii удался, но там мусорные байты >127? перепроверим cp1251
                        if any(ord(c) > 127 and enc == 'ascii' for c in name):
                            pass
                        break
                    q = send
            frame_names.append(name)
            fp_ext = eend
    except Exception as e:
        print(f"[dff] не удалось прочитать имена фреймов: {e}")
    while len(frame_names) < nframes:
        frame_names.append('')
    print(f"[dff] имена фреймов: {[n or '?' for n in frame_names[:8]]} ...")
    pos = fl_end
    # GeometryList
    t3, s3, v3 = struct.unpack_from('<III', data, pos)
    assert t3 == 0x1A, f"expected GeometryList, got {hex(t3)}"
    gl_end = pos + 12 + s3
    gpos = pos + 12
    gt, gs, gv = struct.unpack_from('<III', data, gpos)
    assert gt == 0x01
    ngeoms = struct.unpack_from('<I', data, gpos + 12)[0]
    print(f"[dff] geometries: {ngeoms}")
    gpos += 12 + gs
    geoms = []
    for gi in range(ngeoms):
        gp = gpos
        ct, cs, cv = struct.unpack_from('<III', data, gp)
        assert ct == 0x0F, f"geom {gi}: expected 0x0F, got {hex(ct)}"
        gend = gp + 12 + cs
        # struct
        st, ss, sv = struct.unpack_from('<III', data, gp + 12)
        assert st == 0x01
        sdata_off = gp + 24
        flags, ntri, nvert, nmorph = struct.unpack_from('<IIII', data, sdata_off)
        gver = rw_version(sv)
        p = sdata_off + 16
        if gver < 0x34000:
            p += 12
        native = bool(flags & 0x01000000)
        uvs = None
        tris = None
        verts = None
        normals = None
        if not native:
            if flags & 0x08:  # PRELIT
                p += nvert * 4
            tex_count = (flags & 0x00FF0000) >> 16
            if tex_count == 0:
                if flags & 0x80:
                    tex_count = 2
                elif flags & 0x04:
                    tex_count = 1
                else:
                    tex_count = 0
            if tex_count > 0:
                # читаем первый слой, остальные пропускаем
                uvs = np.frombuffer(data, dtype=np.float32, count=nvert * 2,
                                    offset=p).reshape(nvert, 2).copy()
                p += tex_count * nvert * 8
            else:
                uvs = None
            tris = np.frombuffer(data, dtype=np.uint16, count=ntri * 4,
                                 offset=p).reshape(ntri, 4).copy()
            # Triangle: b, a, material, c
            p += ntri * 8
        # morph
        # sphere 16 + hasV(4)+hasN(4)
        has_v = struct.unpack_from('<I', data, p + 16)[0]
        has_n = struct.unpack_from('<I', data, p + 20)[0]
        p += 24
        if has_v:
            verts = np.frombuffer(data, dtype=np.float32, count=nvert * 3,
                                  offset=p).reshape(nvert, 3).copy()
            p += nvert * 12
        if has_n:
            normals = np.frombuffer(data, dtype=np.float32, count=nvert * 3,
                                    offset=p).reshape(nvert, 3).copy()
            p += nvert * 12
        # MaterialList — находится после struct внутри Geometry
        # struct chunk занимает ss байт: p должен указывать на конец struct?
        # Найдём MaterialList поиском: после struct (gp+12+12+ss) идёт MatList
        ml_pos = gp + 12 + 12 + ss
        mt, ms, mv2 = struct.unpack_from('<III', data, ml_pos)
        assert mt == 0x08, f"geom {gi}: expected MatList, got {hex(mt)}"
        ml_end = ml_pos + 12 + ms
        mp = ml_pos + 12
        mst, mss, msv = struct.unpack_from('<III', data, mp)
        assert mst == 0x01
        nmats = struct.unpack_from('<I', data, mp + 12)[0]
        mp += 12 + mss
        materials = []
        for mi in range(nmats):
            mcp, mct, mcs, mcv = mp, *struct.unpack_from('<III', data, mp)
            assert mct == 0x07, f"geom {gi} mat {mi}: {hex(mct)}"
            mend = mcp + 12 + mcs
            # material struct
            mstp, msts, mstv = struct.unpack_from('<III', data, mp + 12)
            msd_off = mp + 24
            mflags = struct.unpack_from('<I', data, msd_off)[0]
            r, g, b, a = struct.unpack_from('<4B', data, msd_off + 4)
            # is_textured = struct.unpack_from('<I', data, msd_off+12)[0]
            mat_raw = data[mcp + 12:mend]
            full = parse_material_full(mat_raw)
            full['color'] = (r, g, b, a)
            materials.append(full)
            mp = mend
        geoms.append({
            'flags': flags, 'ntri': ntri, 'nvert': nvert,
            'uvs': uvs, 'tris': tris, 'verts': verts, 'normals': normals,
            'materials': materials,
        })
        gpos = gend
        if (gi + 1) % 10 == 0:
            print(f"[dff]   geom {gi+1}/{ngeoms} parsed")
    # Atomics
    pos = gl_end
    atomics = []
    ai = 0
    while pos + 12 <= end:
        ct, cs, cv = struct.unpack_from('<III', data, pos)
        if ct == 0x03:  # финальный extension
            break
        if ct != 0x14:
            print(f"[dff] пропуск чанка {hex(ct)} на {hex(pos)}")
            pos += 12 + cs
            continue
        st, ss, sv = struct.unpack_from('<III', data, pos + 12)
        assert st == 0x01
        d = struct.unpack_from('<4I', data, pos + 24)
        frame_idx, geom_idx = d[0], d[1]
        atomics.append({'frame': frame_idx, 'geom': geom_idx})
        pos += 12 + cs
        ai += 1
    print(f"[dff] atomics: {len(atomics)}")
    return {
        'frames': frames, 'W_R': W_R, 'W_T': W_T,
        'geoms': geoms, 'atomics': atomics,
        'frame_names': frame_names,
    }


def parse_paint(s):
    """'#3cff00' / '3cff00' -> (r, g, b)."""
    s = (s or '').strip()
    if s.startswith('#'):
        s = s[1:]
    if len(s) != 6 or any(c not in '0123456789abcdefABCDEF' for c in s):
        raise ValueError(f'нужен HEX вида #rrggbb, получено {s!r}')
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


# ---------- GLB ----------
def build_glb_bytes(dff, images, paint=None):
    geoms = dff['geoms']
    atomics = dff['atomics']
    W_R = dff['W_R']
    W_T = dff['W_T']

    # группировка по материалу: key -> {'pos':[], 'nor':[], 'uv':[], 'idx':[], 'map':{}}
    groups = {}
    order = []

    def gkey(m):
        c = m.get('color', (255, 255, 255, 255))
        return (m.get('texture') or '', c[0], c[1], c[2], c[3],
                m.get('matfx', 0), round(m.get('spec_lvl', 0.0), 3),
                m.get('refl_i', 0.0))

    total_tris = 0

    def world_verts(g, fi):
        """Вершины/нормали геометрии в мировых координатах GTA + UV."""
        T = W_T[fi] if fi < len(W_T) else np.zeros(3)
        Rm = W_R[fi] if fi < len(W_R) else np.eye(3)
        # вершины в мир GTA (вращения ~единичные)
        V = g['verts'].astype(np.float64)  # (n,3) x,y,z gta
        try:
            Vw = V @ np.asarray(Rm).T + np.asarray(T)
        except Exception:
            Vw = V + np.asarray(T)
        if g['normals'] is not None:
            N = g['normals'].astype(np.float64)
            try:
                Nw = N @ np.asarray(Rm).T
            except Exception:
                Nw = N
        else:
            Nw = np.zeros_like(Vw)
        if g['uvs'] is not None:
            # Без flip: и GTA(D3D), и glTF — top-origin (v=0 верх).
            UV = g['uvs'].copy()
        else:
            UV = np.zeros((len(Vw), 2), dtype=np.float32)
        return Vw, Nw, UV

    def new_store():
        return {}, []

    def emit_job(groups, order, g, Vw, Nw, UV, tag, flip):
        """Складывает треугольники в группы материалов.
        Vw/Nw — координаты GTA (мировые для кузова, локальные для
        руля/колёс — нода несёт трансляцию); flip=True зеркалит обход."""
        nonlocal total_tris
        # конвертация GTA -> Three: (x, y, z) -> (x, z, -y)
        P = np.empty_like(Vw)
        P[:, 0] = Vw[:, 0]
        P[:, 1] = Vw[:, 2]
        P[:, 2] = -Vw[:, 1]
        Nn = np.empty_like(Nw)
        Nn[:, 0] = Nw[:, 0]
        Nn[:, 1] = Nw[:, 2]
        Nn[:, 2] = -Nw[:, 1]
        nl = np.linalg.norm(Nn, axis=1, keepdims=True)
        nl[nl == 0] = 1
        Nn /= nl
        mats = g['materials']
        tris = g['tris']  # (ntri,4): b,a,mat,c
        for t in tris:
            b, a, m, c = int(t[0]), int(t[1]), int(t[2]), int(t[3])
            if m < 0 or m >= len(mats):
                m = 0
            mat = mats[m] if mats else {'color': (255, 255, 255, 255), 'texture': None,
                                                 'matfx': 0, 'spec_lvl': 0.0, 'spec_tex': '',
                                                 'refl_i': 0.0}
            key = gkey(mat)
            if key not in groups:
                groups[key] = {'pos': [], 'nor': [], 'uv': [], 'idx': [],
                               'map': {}, 'mat': mat}
                order.append(key)
            gr = groups[key]
            mp = gr['map']
            # порядок вершин треугольника: a, b, c (как в DragonFF bin mesh)
            seq = (a, c, b) if flip else (a, b, c)
            for vi in seq:
                mk = (tag, vi)
                ni = mp.get(mk)
                if ni is None:
                    ni = len(gr['pos'])
                    mp[mk] = ni
                    gr['pos'].append(P[vi].tolist())
                    gr['nor'].append(Nn[vi].tolist())
                    gr['uv'].append(UV[vi].tolist())
            if flip:
                gr['idx'].extend([mp[(tag, a)], mp[(tag, c)], mp[(tag, b)]])
            else:
                gr['idx'].extend([mp[(tag, a)], mp[(tag, b)], mp[(tag, c)]])
            total_tris += 1

    # --- как устроено в GTA SA ---
    # Кузов/салон/двери/бамперы: статика, запекаем мировые трансляции.
    # Колёса: wheel_*_dummy — нода поворота (yaw вокруг Z GTA / Y Three,
    # только передние), wheel mesh — нода вращения (вокруг оси X GTA /
    # X Three; в игре клонируется с RF на 4 угла). Левый борт — зеркально.
    # Руль в салоне: крутится вокруг колонки пропорционально повороту.
    #
    # Резолв деталей — ПО ИМЕНАМ (стандарт SA), без индексов фреймов:
    #  (a) пары X_ok + X_dummy (двери/капот/багажник/бамперы/стекло) —
    #      находятся в любой SA-модели; пивот = дамми, как в игре;
    #  (b) таблица кандидатов для остального (оптика, решётка, руль);
    #  (c) эвристики: колесо = геометрия-ребёнок wheel-дамми, иначе по форме.
    # Всё ненайденное тихо остаётся в статике. Индексов фреймов в коде нет.
    names = dff.get('frame_names', []) or []
    atomics_by_frame = {}
    for at in atomics:
        atomics_by_frame.setdefault(at['frame'], at)

    _low = {}
    for _i, _n in enumerate(names):
        if _n and _n.lower() not in _low:
            _low[_n.lower()] = _i

    def find_exact(*cands):
        """Первый фрейм с именем из кандидатов (точное, потом без регистра)."""
        for c in cands:
            for i, n in enumerate(names):
                if n == c:
                    return i
        for c in cands:
            if c.lower() in _low:
                return _low[c.lower()]
        return None

    def find_all(*cands):
        """Все фреймы с именами из кандидатов."""
        out = []
        for c in cands:
            for i, n in enumerate(names):
                if n == c or (n and n.lower() == c.lower()):
                    if i not in out:
                        out.append(i)
        return out

    def has_geom(fi):
        return (fi is not None and fi in atomics_by_frame
                and geoms[atomics_by_frame[fi]['geom']]['verts'] is not None)

    nframes = len(dff['frames'])
    FI = lambda i: i if (i is not None and 0 <= i < nframes) else None

    # --- колёса: стандартные дамми SA ---
    steer_fis = {}
    for side in ('rf', 'lf', 'rb', 'lb'):
        steer_fis[side] = FI(find_exact(f'wheel_{side}_dummy'))
    # --- колесо-меш: (c1) ребёнок wheel-дамми, (c2) имя, (c3) форма ---
    wheel_mesh_fi = None
    for fi in atomics_by_frame:
        p = dff['frames'][fi]['parent']
        if (p is not None and 0 <= p < len(names)
                and (names[p] or '').lower().startswith('wheel_')
                and (names[p] or '').lower().endswith('_dummy')):
            wheel_mesh_fi = fi
            break
    if wheel_mesh_fi is None:
        wheel_mesh_fi = FI(find_exact('wheel_sr6', 'wheel_ok', 'wheel'))
    # --- руль: кандидаты + подстрока steer/рул (у VehFuncs-моделей f_steer) ---
    steer_wheel_fi = FI(find_exact('руль', 'rul', 'steering', 'steering_wheel',
                                   'f_steer', 'volant'))
    if steer_wheel_fi is None or not has_geom(steer_wheel_fi):
        steer_wheel_fi = None
        for i, n in enumerate(names):
            nl = (n or '').lower()
            if ('steer' in nl or 'рул' in nl) and has_geom(i):
                steer_wheel_fi = i
                break
    print(f"[glb] фреймы: wheel_mesh={wheel_mesh_fi} ({names[wheel_mesh_fi] if wheel_mesh_fi is not None and wheel_mesh_fi < len(names) else '?'}) "
          f"dummies={steer_fis} руль={steer_wheel_fi} "
          f"({names[steer_wheel_fi] if steer_wheel_fi is not None else '?'})")

    def gta_to_three_p(p):
        p = np.asarray(p, dtype=np.float64)
        return [float(p[0]), float(p[2]), float(-p[1])]

    W = lambda i: np.asarray(W_T[i], dtype=np.float64)

    static_groups, static_order = new_store()
    wheelR_groups, wheelR_order = new_store()  # правый борт (оригинал)
    wheelL_groups, wheelL_order = new_store()  # левый борт (зеркало)
    steer_groups, steer_order = new_store()    # руль в салоне

    g_wheel = None
    if wheel_mesh_fi in atomics_by_frame:
        g_wheel = geoms[atomics_by_frame[wheel_mesh_fi]['geom']]
    if g_wheel is None and any(v is not None for v in steer_fis.values()):
        # запасной путь: геометрия колеса по форме —
        # узкая по X (ступица), круглая в Y-Z (диаметр)
        best, best_score = None, 1e9
        for g in geoms:
            if g['verts'] is None:
                continue
            mn, mx = g['verts'].min(axis=0), g['verts'].max(axis=0)
            ext = mx - mn
            if ext[0] < 0.45 and abs(ext[1] - ext[2]) < 0.15 * max(ext[1], ext[2]):
                score = abs(ext[0] - 0.27)
                if score < best_score:
                    best, best_score = g, score
        g_wheel = best
        if best is not None:
            print("[glb] колесо найдено по форме геометрии (имя/родитель не подошли)")

    g_steer = None
    if steer_wheel_fi in atomics_by_frame:
        g_steer = geoms[atomics_by_frame[steer_wheel_fi]['geom']]

    # --- интерактивные детали: каждая — свой меш + нода с пивотом ---
    # pivot ('dummy', fi) = мировая позиция дамми — как в игре, деталь
    # вращается/отрывается вокруг дамми; ('bbox',) = центр геометрии
    # (для оптики/стёкол, где пивот не важен, а важно вкл/выкл).
    # hinge_axis — ось петли в координатах GTA для будущих анимаций
    # (двери — вертикаль, капот/багажник — поперечная).
    # --- детали: резолв по именам ---
    # (a) Пары X_ok + X_dummy — стандартный паттерн SA для навесного
    # (двери/капот/багажник/бамперы/стёкла): пивот = дамми, как в игре.
    # Находится в любой SA-модели с ванильными именами; здесь же покрывает
    # и кривой случай bump_front_ok (родитель — корень, а не дамми).
    # (b) Таблица кандидатов для прочего (оптика, решётка, повторители).
    # Промах = деталь тихо остаётся в статике. Индексов фреймов нет.
    PARTS = []
    for _role, _hinge in (('door_rf', (0, 0, 1)), ('door_lf', (0, 0, 1)),
                          ('door_rr', (0, 0, 1)), ('door_lr', (0, 0, 1)),
                          ('bonnet', (1, 0, 0)), ('boot', (1, 0, 0)),
                          ('bumper_front', None), ('bumper_rear', None),
                          ('windscreen', None)):
        _variants = [_role, _role.replace('bumper_', 'bump_')]
        _fi = _di = None
        for _v in _variants:
            _fi = _fi or find_exact(_v + '_ok')
            _di = _di or find_exact(_v + '_dummy')
        if _fi is None or not has_geom(_fi):
            continue
        _pname = _role  # bumper_* — стабильные имена нод в GLB/вьювере
        if _di is None:
            PARTS.append((_pname, [_fi], ('bbox',), _hinge))
            print(f"[glb] деталь {_pname}: дамми нет — пивот по bbox")
        else:
            PARTS.append((_pname, [_fi], ('dummy', _di), _hinge))
    for _pname, _cands in (
        ('optika_front',    ['optika_front', 'headlight_housing']),
        ('bulbs_front',     ['лампочки', 'bulbs_front', 'bulbs front']),
        ('headlight',       ['фара перед', 'fara pered', 'headlight']),
        ('glass_headlight', ['steklo', 'headlight_glass', 'glass_headlight']),
        ('repeater',        ['povorotnik', 'repeater', 'side_indicator']),
        ('repeater_glass',  ['povorot_steklo', 'repeater_glass']),
        ('grille',          ['reshetka', 'grille']),
        ('front_trim',      ['metal02', 'metal02_trim']),
        ('optika_rear',     ['optika_rear', 'taillight_housing']),
        ('bulbs_rear',      ['лампочки зад', 'bulbs_rear', 'bulbs rear']),
        ('lens_red',        ['glass_red01', 'lens_red']),
        ('lens_clear',      ['glass10', 'lens_clear']),
        ('lens_rear',       ['стекло фара зад', 'lens_rear']),
        ('reflector_rear',  ['glass_red02', 'reflector_rear']),
        ('glass_rear',      ['steklo_zad', 'glass_rear', 'rear_window']),
    ):
        _fis = []
        for _c in _cands:
            _fi = find_exact(_c)
            if _fi is not None and has_geom(_fi) and _fi not in _fis:
                _fis.append(_fi)
        if _pname == 'front_trim':
            for _extra in ('metal_01', 'Plane010', 'metal01', 'plane'):
                _fi = find_exact(_extra)
                if _fi is not None and has_geom(_fi) and _fi not in _fis:
                    _fis.append(_fi)
        if not _fis:
            continue
        PARTS.append((_pname, _fis, ('bbox',), None))
    # маркеры без геометрии — точки привязки для будущего
    # (дым выхлопа, посадка персонажей, крышка бака, габариты света)
    MARKERS = []
    for _m in ('headlights', 'taillights', 'exhaust', 'engine',
               'ped_frontseat', 'ped_backseat', 'petrolcap', 'numberplate'):
        for _fi in find_all(_m):
            if _fi not in atomics_by_frame and _fi not in MARKERS:
                MARKERS.append(_fi)

    frame_to_part = {}
    for pname, frames, _piv, _h in PARTS:
        for fi in frames:
            frame_to_part[fi] = pname

    part_data = {}  # pname -> {groups, order, pivot_gta, frames, hinge}
    for pname, frames, pivspec, hinge in PARTS:
        jobs_p = []
        for fi in frames:
            at = atomics_by_frame.get(fi)
            if at is None:
                print(f"[glb] деталь {pname}: фрейм {fi} без геометрии — пропуск")
                continue
            g = geoms[at['geom']]
            if g['verts'] is None or g['tris'] is None:
                continue
            Vw, Nw, UV = world_verts(g, fi)
            jobs_p.append((g, Vw, Nw, UV, fi))
        if not jobs_p:
            print(f"[glb] деталь {pname}: нет геометрии — пропуск")
            continue
        allv = np.vstack([j[1] for j in jobs_p])
        bbmin, bbmax = allv.min(axis=0), allv.max(axis=0)
        center = (bbmin + bbmax) / 2.0
        if pivspec[0] == 'dummy':
            P = W(pivspec[1])
            # диагностика: дамми должен быть у края детали (петля), а не в центре
            print(f"[glb] деталь {pname}: пивот-дамми {P.round(3).tolist()}, "
                  f"bbox-центр {center.round(3).tolist()}")
        else:
            P = center
        groups, order = new_store()
        for g, Vw, Nw, UV, fi in jobs_p:
            emit_job(groups, order, g, Vw - P, Nw, UV, f'{pname}:{fi}', False)
        part_data[pname] = {'groups': groups, 'order': order,
                            'pivot_gta': [float(v) for v in P],
                            'frames': frames, 'hinge': hinge,
                            'tris': sum(len(gr['idx']) // 3 for gr in groups.values())}

    # --- номера: quads с plaka/plakalk из бамперов — в свои ноды ---
    # (в DFF отдельного фрейма номера нет; игра в SA рисует его поверх).
    # Группы с этими текстурами уже лежат отдельно внутри бамперов —
    # переносим их в свой стор и перебазируем на собственный центр.
    for pname, platename in (('bumper_front', 'plate_front'),
                             ('bumper_rear', 'plate_rear')):
        if pname not in part_data:
            continue
        d = part_data[pname]
        pg, po = new_store()
        for k in list(d['order']):
            if k[0] in ('plaka', 'plakalk'):
                pg[k] = d['groups'].pop(k)
                d['order'].remove(k)
                po.append(k)
        if not po:
            print(f"[glb] {platename}: quads номера не найдены")
            continue
        allp = np.vstack([np.array(pg[k]['pos'], dtype=np.float64) for k in po])
        c = (allp.min(axis=0) + allp.max(axis=0)) / 2.0  # центр (Three, отн. пивота бампера)
        for k in po:
            pg[k]['pos'] = (np.array(pg[k]['pos'], dtype=np.float64) - c).tolist()
        bp = np.asarray(d['pivot_gta'], dtype=np.float64)
        P = bp + np.array([c[0], -c[2], c[1]])  # three->gta смещения
        ntris = sum(len(pg[k]['idx']) // 3 for k in po)
        part_data[platename] = {'groups': pg, 'order': po,
                                'pivot_gta': [float(v) for v in P],
                                'frames': d['frames'], 'hinge': None,
                                'tris': ntris}
        d['tris'] = sum(len(gr['idx']) // 3 for gr in d['groups'].values())
        print(f"[glb] {platename}: {ntris} tris, пивот GTA {P.round(3).tolist()}")

    # --- статика: всё остальное (кузов, салон, подкапотка) одним мешем ---
    static_jobs = 0
    for at in atomics:
        fi = at['frame']
        if fi == wheel_mesh_fi or fi == steer_wheel_fi:
            continue
        if fi in frame_to_part:
            continue
        g = geoms[at['geom']]
        if g['verts'] is None or g['tris'] is None:
            continue
        Vw, Nw, UV = world_verts(g, fi)
        emit_job(static_groups, static_order, g, Vw, Nw, UV, at['geom'], False)
        static_jobs += 1

    # --- руль: центрируем вершины на пивот (bbox-центр), нода несёт позицию ---
    steer_pivot_three = [0.0, 0.0, 0.0]
    steer_axis_three = [0.0, 0.0, -1.0]  # дефолт: как раньше (чистый Z Three)
    if g_steer is not None and g_steer['verts'] is not None:
        Vw_s, Nw_s, UV_s = world_verts(g_steer, steer_wheel_fi)
        mn, mx = Vw_s.min(axis=0), Vw_s.max(axis=0)
        C = (mn + mx) / 2.0
        steer_pivot_three = gta_to_three_p(C)
        # Ось колонки = нормаль к плоскости руля = ось минимальной
        # дисперсии вершин (PCA). Кручение вокруг чистой Y GTA при
        # наклонной колонке давало бы конусное биение — поэтому ось
        # измеряем, а не предполагаем.
        Vc = Vw_s - Vw_s.mean(axis=0)
        cov = np.cov(Vc.T)
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, 0]
        axis = axis / (np.linalg.norm(axis) or 1.0)
        if axis[1] < 0.0:  # ориентируем вперёд (+Y GTA)
            axis = -axis
        tilt = float(np.degrees(np.arccos(np.clip(axis[1], -1.0, 1.0))))
        ax_t = np.array([axis[0], axis[2], -axis[1]])
        ax_t = ax_t / (np.linalg.norm(ax_t) or 1.0)
        steer_axis_three = [float(v) for v in ax_t]
        emit_job(steer_groups, steer_order, g_steer,
                 Vw_s - C, Nw_s, UV_s, 'steering', False)
        print(f"[glb] руль: фрейм {steer_wheel_fi}, пивот GTA {C.round(3).tolist()} "
              f"-> Three {[round(v, 3) for v in steer_pivot_three]}; "
              f"ось колонки GTA {np.round(axis, 3).tolist()} "
              f"(наклон {tilt:.1f}° от +Y)")
    else:
        print("[glb] руль не найден — нода руля будет пустой")

    # --- колёса: локальная геометрия (центр ~0), ноды несут позиции dummy ---
    wheel_ok = g_wheel is not None and g_wheel['verts'] is not None
    if wheel_ok:
        V0 = g_wheel['verts'].astype(np.float64)
        N0 = (g_wheel['normals'].astype(np.float64) if g_wheel['normals'] is not None
              else np.zeros_like(V0))
        UV0 = (g_wheel['uvs'].copy() if g_wheel['uvs'] is not None
               else np.zeros((len(V0), 2), dtype=np.float32))
        mir = np.array([-1.0, 1.0, 1.0])
        emit_job(wheelR_groups, wheelR_order, g_wheel, V0, N0, UV0, 'wheelR', False)
        emit_job(wheelL_groups, wheelL_order, g_wheel,
                 V0 * mir, N0 * mir, UV0, 'wheelL', True)
        print(f"[glb] колесо: вершин {len(V0)}, tris {len(g_wheel['tris'])} "
              f"(R/L варианты собраны)")
    else:
        print("[glb] ВНИМАНИЕ: геометрия колеса не найдена!")

    # (пустой цикл-заглушка удалён; emit идёт по сторам выше)
    print(f"[glb] треугольников (статика/руль/колесоR/колесоL): "
          f"{sum(len(gr['idx']) // 3 for gr in static_groups.values())}/"
          f"{sum(len(gr['idx']) // 3 for gr in steer_groups.values())}/"
          f"{sum(len(gr['idx']) // 3 for gr in wheelR_groups.values())}/"
          f"{sum(len(gr['idx']) // 3 for gr in wheelL_groups.values())}; "
          f"статика: {static_jobs} атомиков")
    stores_in_order = [(static_order, static_groups)]
    for pname, d in part_data.items():
        stores_in_order.append((d['order'], d['groups']))
    stores_in_order += [(steer_order, steer_groups),
                        (wheelR_order, wheelR_groups),
                        (wheelL_order, wheelL_groups)]

    # --- цвет кузова: выбор группы + RGB (применяется при сборке материалов) ---
    # Тот же критерий, что пикер вьювера: крупнейшая группа-kind 'paint'
    # без текстуры. Тёмная окантовка (4,4,4) — тоже paint, но мельче,
    # поэтому не трогается.
    # ВАЖНО: саму группу НЕ мутируем — классификация kind/metallic/roughness
    # должна считаться от ИСХОДНОГО цвета. Иначе белый (255,255,255) после
    # перекраски сам станет 'chrome' (яркий+серый) и запечётся зеркалом
    # вместо белой краски.
    paint_key = None
    paint_rgb = None
    if paint is not None:
        try:
            paint_rgb = parse_paint(paint)
        except ValueError as e:
            raise ValueError(f'плохой цвет кузова: {e}')
        best_tris = -1
        for _o, _g in stores_in_order:
            for k in _o:
                gr = _g[k]
                if classify_material(gr['mat']).get('kind') != 'paint':
                    continue
                t = len(gr['idx']) // 3
                if t > best_tris:
                    paint_key, best_tris = k, t
        if paint_key is None:
            print("[glb] краска не найдена — цвет кузова пропущен")
        else:
            print(f"[glb] краска кузова ({best_tris} tris) -> "
                  f"#{paint_rgb[0]:02X}{paint_rgb[1]:02X}{paint_rgb[2]:02X}")

    # общий порядок материалов: статика, детали, руль, колёса
    order = []
    for _o, _g in stores_in_order:
        for k in _o:
            if k not in order:
                order.append(k)
    groups = {}
    for _o, _g in stores_in_order:
        groups.update(_g)
    for k in order:
        gr = groups[k]
        cls = classify_material(gr['mat'])
        print(f"   {k[0] or '(без текстуры)'} rgba{(k[1],k[2],k[3],k[4])} {cls['kind']} "
              f"met={cls['metallic']} rough={cls['roughness']}: "
              f"verts {len(gr['pos'])} tris {len(gr['idx'])//3}")

    # --- glTF ---
    bin_chunks = []
    buffer_views = []
    accessors = []
    meshes = []
    materials = []
    textures = []
    images_j = []
    samplers = [{'magFilter': 9729, 'minFilter': 9986, 'wrapS': 10497, 'wrapT': 10497}]
    tex_index = {}
    for name, (blob, mime, has_alpha, wh) in images.items():
        tex_index[name] = len(images_j)
        images_j.append({'bufferView': None, 'mimeType': mime, 'name': name})

    mat_index = {}
    gltf_materials = []
    use_clearcoat = False
    used_mat_names = {}
    for key in order:
        gr = groups[key]
        m = gr['mat']
        tex = m.get('texture')
        col = m.get('color', (255, 255, 255, 255))
        cls = classify_material(m)  # всегда от исходного цвета!
        if key == paint_key and paint_rgb is not None:
            col = (paint_rgb[0], paint_rgb[1], paint_rgb[2], col[3])
        base = [col[0] / 255.0, col[1] / 255.0, col[2] / 255.0, cls['opacity']]
        pbr = {'baseColorFactor': base, 'metallicFactor': cls['metallic'],
               'roughnessFactor': cls['roughness']}
        mat_name = f"{cls['kind']}_{tex or 'mat'}_{col[0]}_{col[1]}_{col[2]}_{col[3]}"
        if mat_name in used_mat_names:  # напр. две краски после --paint
            used_mat_names[mat_name] += 1
            mat_name += f"_{used_mat_names[mat_name]}"
        else:
            used_mat_names[mat_name] = 1
        mat_def = {'name': mat_name,
                   'pbrMetallicRoughness': pbr, 'doubleSided': True}
        if cls['alphaMode'] != 'OPAQUE':
            mat_def['alphaMode'] = 'BLEND'
        if any(v > 0.01 for v in cls['emissive']):
            mat_def['emissiveFactor'] = [min(v, 1.0) for v in cls['emissive']]
        if tex and tex in tex_index:
            mat_def['pbrMetallicRoughness']['baseColorTexture'] = {'index': len(textures)}
            textures.append({'source': tex_index[tex], 'sampler': 0, 'name': tex})
            if cls['emissiveTex']:
                mat_def['emissiveTexture'] = {'index': len(textures) - 1}
        else:
            if tex:
                print(f"[glb]   текстура {tex!r} не найдена в TXD — материал без карты")
        if cls['clearcoat'] > 0:
            use_clearcoat = True
            mat_def['extensions'] = {'KHR_materials_clearcoat':
                                     {'clearcoatFactor': cls['clearcoat'],
                                      'clearcoatRoughnessFactor': 0.08}}
        mat_index[key] = len(gltf_materials)
        gltf_materials.append(mat_def)

    # геометрия: по одному mesh на часть (кузов/руль/колёса)
    bin_data = bytearray()

    def add_view(data_bytes, target=None):
        nonlocal bin_data
        # выравнивание 4
        while len(bin_data) % 4:
            bin_data.append(0)
        off = len(bin_data)
        bin_data.extend(data_bytes)
        idx = len(buffer_views)
        d = {'buffer': 0, 'byteOffset': off, 'byteLength': len(data_bytes)}
        if target is not None:
            d['target'] = target
        buffer_views.append(d)
        return idx

    # Собираем геометрию: по одному mesh на часть
    # (кузов / руль / колесо-R / колесо-L), примитивы — по группам материалов.
    def build_primitives(keys, store):
        prims = []
        for key in keys:
            gr = store[key]
            pos_arr = np.array(gr['pos'], dtype=np.float32)
            nor_arr = np.array(gr['nor'], dtype=np.float32)
            uv_arr = np.array(gr['uv'], dtype=np.float32)
            idx_arr = np.array(gr['idx'], dtype=np.uint32)
            # bounds
            mn = pos_arr.min(axis=0).tolist() if len(pos_arr) else [0, 0, 0]
            mx = pos_arr.max(axis=0).tolist() if len(pos_arr) else [0, 0, 0]
            bv_pos = add_view(pos_arr.tobytes(), 34962)
            acc_pos = len(accessors)
            accessors.append({'bufferView': bv_pos, 'componentType': 5126,
                              'count': len(pos_arr), 'type': 'VEC3',
                              'min': mn, 'max': mx})
            bv_nor = add_view(nor_arr.tobytes(), 34962)
            acc_nor = len(accessors)
            accessors.append({'bufferView': bv_nor, 'componentType': 5126,
                              'count': len(nor_arr), 'type': 'VEC3'})
            m = gr['mat']
            tex = m.get('texture')
            prim = {'attributes': {'POSITION': acc_pos, 'NORMAL': acc_nor},
                    'indices': None, 'material': mat_index[key], 'mode': 4}
            if tex and tex in tex_index:
                bv_uv = add_view(uv_arr.tobytes(), 34962)
                acc_uv = len(accessors)
                accessors.append({'bufferView': bv_uv, 'componentType': 5126,
                                  'count': len(uv_arr), 'type': 'VEC2'})
                prim['attributes']['TEXCOORD_0'] = acc_uv
            bv_idx = add_view(idx_arr.tobytes(), 34963)
            acc_idx = len(accessors)
            accessors.append({'bufferView': bv_idx, 'componentType': 5125,
                              'count': len(idx_arr), 'type': 'SCALAR'})
            prim['indices'] = acc_idx
            prims.append(prim)
        return prims

    meshes = []
    mesh_idx = {}  # 'static' | part | 'steer' | 'wheelR' | 'wheelL' -> idx
    mesh_jobs = [('static', static_order, static_groups, 'chassis')]
    for pname, d in part_data.items():
        mesh_jobs.append((pname, d['order'], d['groups'], pname))
    mesh_jobs += [
        ('steer', steer_order, steer_groups, 'steering_wheel'),
        ('wheelR', wheelR_order, wheelR_groups, 'wheel_R'),
        ('wheelL', wheelL_order, wheelL_groups, 'wheel_L'),
    ]
    for tag, keys, store, mname in mesh_jobs:
        if not keys:
            continue
        mesh_idx[tag] = len(meshes)
        meshes.append({'name': mname, 'primitives': build_primitives(keys, store)})

    # изображения
    for name, (blob, mime, has_alpha, wh) in images.items():
        bv = add_view(blob)
        images_j[tex_index[name]]['bufferView'] = bv

    # --- иерархия нод (как в GTA SA) ---
    # chassis (статика, вершины уже в мире) +
    # steering (пивот на центре руля, вращение — ось Z Three = колонка Y GTA) +
    # wheel_{rf,lf,rb,lb}_steer (трансляция dummy, yaw) ->
    #   wheel_*_spin (вращение вокруг X) с мешем колеса.
    nodes = []
    scene_roots = []

    def add_node(d):
        nodes.append(d)
        return len(nodes) - 1

    n_chassis = add_node({'mesh': mesh_idx['static'], 'name': 'chassis'})
    scene_roots.append(n_chassis)

    part_nodes = {}  # pname -> node idx (для будущих анимаций из вьювера)
    for pname, d in part_data.items():
        if pname not in mesh_idx:
            continue
        extra = {'part': pname, 'frames': d['frames'],
                 'pivot_gta': d['pivot_gta']}
        if d['hinge'] is not None:
            extra['hinge_axis'] = list(d['hinge'])
        n_p = add_node({'mesh': mesh_idx[pname], 'name': pname,
                        'translation': gta_to_three_p(d['pivot_gta']),
                        'extras': extra})
        scene_roots.append(n_p)
        part_nodes[pname] = n_p
        print(f"[glb] нода {pname}: tris {d['tris']}, "
              f"translation {[round(v, 3) for v in gta_to_three_p(d['pivot_gta'])]}"
              f"{' hinge ' + str(d['hinge']) if d['hinge'] else ''}")

    # маркеры без геометрии (точки привязки)
    for fi in MARKERS:
        if fi >= len(dff['frames']):
            continue
        mname = (names[fi] if fi < len(names) and names[fi]
                 else f'marker_{fi}')
        n_m = add_node({'name': mname,
                        'translation': gta_to_three_p(W(fi)),
                        'extras': {'marker': mname, 'frame': fi}})
        scene_roots.append(n_m)
        print(f"[glb] маркер {mname}: {[round(v, 3) for v in gta_to_three_p(W(fi))]}")

    if 'steer' in mesh_idx:
        n_sw = add_node({'mesh': mesh_idx['steer'], 'name': 'steering',
                         'translation': steer_pivot_three,
                         'extras': {'steer_axis': steer_axis_three}})
        scene_roots.append(n_sw)
        print(f"[glb] нода steering: translation {steer_pivot_three} "
              f"axis {steer_axis_three}")
    else:
        n_sw = None

    wheel_nodes = {}  # rf/lf/rb/lb -> (steer_idx, spin_idx)
    if 'wheelR' in mesh_idx or 'wheelL' in mesh_idx:
        for side, fi_key in (('rf', 'rf'), ('lf', 'lf'), ('rb', 'rb'), ('lb', 'lb')):
            fi = steer_fis.get(fi_key)
            if fi is None or fi >= len(W_T):
                print(f"[glb] dummy {fi_key} отсутствует — колесо пропущено")
                continue
            is_left = side in ('lf', 'lb')
            mtag = 'wheelL' if is_left else 'wheelR'
            if mtag not in mesh_idx:
                mtag = 'wheelR' if 'wheelR' in mesh_idx else 'wheelL'
            n_steer = add_node({'name': f'wheel_{side}_steer',
                                'translation': gta_to_three_p(W(fi))})
            n_spin = add_node({'mesh': mesh_idx[mtag],
                               'name': f'wheel_{side}_spin'})
            nodes[n_steer]['children'] = [n_spin]
            scene_roots.append(n_steer)
            wheel_nodes[side] = (n_steer, n_spin)
            wpos = gta_to_three_p(W(fi))
            print(f"[glb] нода wheel_{side}_steer: translation "
                  f"{[round(v, 3) for v in wpos]} mesh={mtag}")

    # имя модели — корневой фрейм DFF (раньше было захардкожено)
    model_name = (names[0] if names and names[0] else 'GTA SA vehicle')
    print(f"[glb] модель: {model_name!r}")

    gltf = {
        'asset': {'version': '2.0', 'generator': 'gta-dff-txd-converter'},
        'scene': 0,
        'scenes': [{'nodes': scene_roots, 'name': model_name}],
        'nodes': nodes,
        'meshes': meshes,
        'materials': gltf_materials,
        'textures': textures,
        'images': images_j,
        'samplers': samplers,
        'accessors': accessors,
        'bufferViews': buffer_views,
        'buffers': [{'byteLength': len(bin_data)}],
    }
    if use_clearcoat:
        gltf['extensionsUsed'] = ['KHR_materials_clearcoat']
    # выравнивание bin ДО упаковки JSON, чтобы byteLength был точным
    while len(bin_data) % 4:
        bin_data.append(0)
    gltf['buffers'] = [{'byteLength': len(bin_data)}]
    j = json.dumps(gltf, separators=(',', ':')).encode('utf-8')
    while len(j) % 4:
        j += b' '
    total = 12 + 8 + len(j) + 8 + len(bin_data)
    out = bytearray()
    out += struct.pack('<4sII', b'glTF', 2, total)
    out += struct.pack('<II', len(j), 0x4E4F534A)
    out += j
    out += struct.pack('<II', len(bin_data), 0x004E4942)
    out += bin_data
    return bytes(out)


def build_glb(dff, images, out_path, paint=None):
    data = build_glb_bytes(dff, images, paint=paint)
    with open(out_path, 'wb') as f:
        f.write(data)
    print(f"[glb] записан {out_path} ({os.path.getsize(out_path)//1024//1024} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dff', required=True)
    ap.add_argument('--txd', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pngdir', default=None)
    ap.add_argument('--paint', default=None,
                    help='цвет кузова в GLB, HEX #rrggbb (как carcols.dat)')
    args = ap.parse_args()

    print(f"DFF: {args.dff}\nTXD: {args.txd}")
    tx = parse_txd(args.txd)
    print(f"[txd] текстур: {len(tx)}")
    images = txd_to_images(tx, out_dir=args.pngdir)
    dff = parse_dff(args.dff)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    build_glb(dff, images, args.out, paint=args.paint)


if __name__ == '__main__':
    main()
