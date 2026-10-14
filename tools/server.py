#!/usr/bin/env python3
"""Веб-конвертер GTA SA DFF+TXD -> GLB.

Раздаёт `web/` как статику и принимает конвертацию:
    POST /api/convert  (multipart: dff=<.dff>, txd=<.txd>) -> model/gltf-binary
    GET  /api/health   -> {"ok": true}

Запуск:
    python3 tools/server.py [--port 8000] [--dir web]

Только стандартная библиотека (cgi убран — свой мини-парсер multipart).
Инструмент для localhost, без аутентификации.
"""
import argparse
import json
import os
import re
import sys
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import build_glb_bytes, parse_dff_data, parse_txd_data, txd_to_images

MAX_BYTES = 300 * 1024 * 1024


def parse_multipart(body: bytes, boundary: bytes):
    """Минимум multipart/form-data: {name: (filename, bytes)}."""
    res = {}
    for ch in body.split(b'--' + boundary):
        if ch in (b'', b'\r\n', b'--', b'--\r\n'):
            continue
        head, sep, payload = ch.partition(b'\r\n\r\n')
        if not sep:
            continue
        if payload.endswith(b'\r\n'):
            payload = payload[:-2]
        m = re.search(br'name="([^"]+)"', head)
        if not m:
            continue
        name = m.group(1).decode('utf-8', 'ignore')
        mf = re.search(br'filename="([^"]*)"', head)
        fname = mf.group(1).decode('utf-8', 'ignore') if mf else ''
        res[name] = (fname, payload)
    return res


def safe_stem(filename, fallback='model'):
    stem = os.path.splitext(os.path.basename(filename or ''))[0].strip()
    stem = re.sub(r'[^\w\-.]+', '_', stem, flags=re.UNICODE) or fallback
    return stem[:80]


class Handler(SimpleHTTPRequestHandler):
    server_version = 'GTAConvert/1.0'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.web_dir, **kwargs)

    def log_message(self, fmt, *args):
        sys.stderr.write('[http] ' + fmt % args + '\n')

    def _json(self, code, obj):
        data = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if urlparse(self.path).path == '/api/health':
            self._json(200, {'ok': True, 'service': 'gta-dff-txd-converter'})
            return
        super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != '/api/convert':
            self._json(404, {'error': 'unknown endpoint'})
            return
        try:
            length = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BYTES:
            self._json(413 if length > MAX_BYTES else 400,
                       {'error': 'пустое тело или больше 300 МБ'})
            return
        ctype = self.headers.get('Content-Type', '')
        m = re.search(r'boundary=([^;]+)', ctype)
        if not m:
            self._json(400, {'error': 'нужен multipart/form-data'})
            return
        try:
            body = self.rfile.read(length)
            files = parse_multipart(body, m.group(1).strip().encode('latin-1'))
        except Exception as e:
            self._json(400, {'error': f'не разобрать multipart: {e}'})
            return
        dff_name, dff_data = files.get('dff', ('', b''))
        txd_name, txd_data = files.get('txd', ('', b''))
        paint_name, paint_data = files.get('paint', ('', b''))
        paint = paint_data.decode('utf-8', 'ignore').strip() or None
        if paint and (len(paint) > 7 or not re.fullmatch(r'#?[0-9a-fA-F]{6}', paint)):
            self._json(400, {'error': 'цвет — HEX вида #rrggbb'})
            return
        if not dff_data or not txd_data:
            self._json(400, {'error': 'нужны оба файла: dff и txd'})
            return
            self._json(400, {'error': 'нужны оба файла: dff и txd'})
            return
        if len(dff_data) < 12 or len(txd_data) < 12:
            self._json(400, {'error': 'файлы слишком маленькие'})
            return
        try:
            print(f"[api] convert {dff_name} ({len(dff_data)//1024}K) + "
                  f"{txd_name} ({len(txd_data)//1024}K)"
                  + (f" paint {paint}" if paint else ""), flush=True)
            tx = parse_txd_data(txd_data)
            images = txd_to_images(tx)
            dff = parse_dff_data(dff_data)
            blob = build_glb_bytes(dff, images, paint=paint)
        except AssertionError as e:
            self._json(422, {'error': f'не похоже на DFF/TXD: {e}'})
            return
        except Exception as e:
            traceback.print_exc()
            self._json(422, {'error': f'ошибка конвертации: {e}'})
            return
        out_name = safe_stem(dff_name) + '.glb'
        self.send_response(200)
        self.send_header('Content-Type', 'model/gltf-binary')
        self.send_header('Content-Length', str(len(blob)))
        self.send_header('Content-Disposition',
                         f'attachment; filename="{out_name}"')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(blob)
        print(f"[api] -> {out_name} ({len(blob)//1024//1024} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser(description='Веб-конвертер DFF+TXD -> GLB')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--dir', default=None,
                    help='корень статики (по умолчанию web/ рядом с проектом)')
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    web_dir = os.path.abspath(args.dir or os.path.join(root, 'web'))
    Handler.web_dir = web_dir
    srv = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f"Статика: {web_dir}\nAPI: http://localhost:{args.port}/api/convert")
    print("Открыть: http://localhost:8000 (блок «Конвертер DFF+TXD»)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
