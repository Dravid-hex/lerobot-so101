#!/usr/bin/env python3
"""
Calibracion del detector de reward para apilar borradores (eraser stacking).

Que hace:
  1. Abre las dos camaras (scene = frontal, side = lateral 3/4).
  2. Te deja seleccionar la ROI (zona fija donde siempre estan los borradores)
     para cada camara.
  3. Te deja ajustar el rango HSV del verde del fieltro con sliders, viendo la
     mascara en vivo.
  4. Capturas varias referencias del estado "NO apilado" (tecla F) y del estado
     "APILADO" (tecla S). El script mide la altura del verde, el area y el numero
     de blobs en cada estado.
  5. Guarda todo en reward_calibration.json para que lo use eraser_reward.py.

Uso:
  python calibrate_eraser_reward.py \
      --scene-cam /dev/video3 \
      --side-cam  /dev/video2 \
      --width 1280 --height 720

Teclas (en la ventana de mascara):
  F  -> capturar frame actual como NO apilado (pon los borradores separados)
  S  -> capturar frame actual como APILADO   (pon los borradores apilados)
  R  -> volver a seleccionar las ROIs
  Q  -> guardar calibracion y salir
"""

import argparse
import json
import time

import cv2
import numpy as np


def open_camera(path, width, height, fps=30):
    """Abre una camara con la misma config que usa lerobot (V4L2 + MJPG)."""
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir la camara {path}")
    # Warmup: descarta unos frames iniciales
    for _ in range(10):
        cap.read()
        time.sleep(0.02)
    return cap


def read_frame(cap, path):
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"No pude leer frame de {path}")
    return frame


def green_mask(hsv_roi, lower, upper):
    """Devuelve la mascara binaria del verde dentro de la ROI ya en HSV."""
    mask = cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
    # Limpieza morfologica para quitar ruido
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def measure(mask, min_blob_area):
    """
    Extrae las 3 senales de la mascara:
      top_y_norm : altura del pixel verde mas alto, normalizada (0=arriba, 1=abajo)
      area_frac  : fraccion de la ROI que es verde
      num_blobs  : numero de manchas verdes con area >= min_blob_area
    """
    h, w = mask.shape[:2]
    total = float(h * w)
    area_frac = float(mask.sum() / 255) / total if total > 0 else 0.0

    ys, _ = np.where(mask > 0)
    top_y_norm = float(ys.min()) / h if ys.size > 0 else 1.0

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    num_blobs = 0
    for i in range(1, n_labels):  # 0 es el fondo
        if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
            num_blobs += 1

    return {"top_y_norm": top_y_norm, "area_frac": area_frac, "num_blobs": num_blobs}


def nothing(_):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-cam", default="/dev/video3")
    ap.add_argument("--side-cam", default="/dev/video2")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", default="reward_calibration.json")
    ap.add_argument("--min-blob-area", type=int, default=500)
    args = ap.parse_args()

    cams = {
        "scene": open_camera(args.scene_cam, args.width, args.height),
        "side": open_camera(args.side_cam, args.width, args.height),
    }
    cam_paths = {"scene": args.scene_cam, "side": args.side_cam}

    # --- 1. Seleccion de ROI por camara ---
    rois = {}

    def select_rois():
        for name, cap in cams.items():
            frame = read_frame(cap, cam_paths[name])
            r = cv2.selectROI(
                f"Selecciona la zona de los borradores ({name}) y ENTER",
                frame, showCrosshair=True, fromCenter=False,
            )
            cv2.destroyAllWindows()
            if r[2] == 0 or r[3] == 0:  # si no selecciono nada, usa el frame completo
                r = (0, 0, frame.shape[1], frame.shape[0])
            rois[name] = tuple(int(v) for v in r)
            print(f"ROI {name}: {rois[name]}")

    select_rois()

    # --- 2. Sliders HSV ---
    win = "mascara (F=no apilado, S=apilado, R=ROI, Q=guardar)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # Defaults pensados para verde olivo/oscuro del fieltro
    defaults = {"Hmin": 30, "Smin": 25, "Vmin": 20, "Hmax": 95, "Smax": 255, "Vmax": 230}
    for k, v in defaults.items():
        maxv = 179 if k.startswith("H") else 255
        cv2.createTrackbar(k, win, v, maxv, nothing)

    flat_caps = {"scene": [], "side": []}
    stacked_caps = {"scene": [], "side": []}

    print("\n--- Calibracion en vivo ---")
    print("Ajusta los sliders hasta que SOLO el fieltro verde quede blanco en la mascara.")
    print("Pon los borradores SEPARADOS y presiona F varias veces (5-10).")
    print("Luego apilalos y presiona S varias veces (5-10).")
    print("Presiona Q para guardar.\n")

    while True:
        lower = [cv2.getTrackbarPos("Hmin", win),
                 cv2.getTrackbarPos("Smin", win),
                 cv2.getTrackbarPos("Vmin", win)]
        upper = [cv2.getTrackbarPos("Hmax", win),
                 cv2.getTrackbarPos("Smax", win),
                 cv2.getTrackbarPos("Vmax", win)]

        vis_rows = []
        cur = {}
        for name, cap in cams.items():
            frame = read_frame(cap, cam_paths[name])
            x, y, w, h = rois[name]
            roi = frame[y:y + h, x:x + w]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = green_mask(hsv, lower, upper)
            cur[name] = (roi, mask)

            m = measure(mask, args.min_blob_area)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            label = f"{name} blobs={m['num_blobs']} top={m['top_y_norm']:.2f} area={m['area_frac']:.3f}"
            roi_small = cv2.resize(roi, (320, 240))
            mask_small = cv2.resize(mask_bgr, (320, 240))
            row = np.hstack([roi_small, mask_small])
            cv2.putText(row, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)
            vis_rows.append(row)

        cv2.imshow(win, np.vstack(vis_rows))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("f"):
            for name in cams:
                _, mask = cur[name]
                flat_caps[name].append(measure(mask, args.min_blob_area))
            print(f"NO apilado capturado (scene={len(flat_caps['scene'])}, side={len(flat_caps['side'])})")
        elif key == ord("s"):
            for name in cams:
                _, mask = cur[name]
                stacked_caps[name].append(measure(mask, args.min_blob_area))
            print(f"APILADO capturado (scene={len(stacked_caps['scene'])}, side={len(stacked_caps['side'])})")
        elif key == ord("r"):
            cv2.destroyWindow(win)
            select_rois()
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            for k, v in defaults.items():
                maxv = 179 if k.startswith("H") else 255
                cv2.createTrackbar(k, win, lower[0] if False else v, maxv, nothing)
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    for cap in cams.values():
        cap.release()

    # --- 3. Resumir referencias y guardar ---
    def avg(caps, field):
        return float(np.mean([c[field] for c in caps])) if caps else None

    calib = {
        "hsv_lower": lower,
        "hsv_upper": upper,
        "min_blob_area": args.min_blob_area,
        "cameras": {},
    }
    for name in cams:
        f_caps, s_caps = flat_caps[name], stacked_caps[name]
        calib["cameras"][name] = {
            "roi": list(rois[name]),
            "flat_top_y": avg(f_caps, "top_y_norm"),
            "stacked_top_y": avg(s_caps, "top_y_norm"),
            "flat_area": avg(f_caps, "area_frac"),
            "stacked_area": avg(s_caps, "area_frac"),
            "flat_blobs": avg(f_caps, "num_blobs"),
            "stacked_blobs": avg(s_caps, "num_blobs"),
        }

    with open(args.out, "w") as fp:
        json.dump(calib, fp, indent=2)

    print(f"\nCalibracion guardada en {args.out}")
    print(json.dumps(calib, indent=2))

    # Aviso de separabilidad por camara
    for name, c in calib["cameras"].items():
        if c["flat_top_y"] is not None and c["stacked_top_y"] is not None:
            sep = abs(c["flat_top_y"] - c["stacked_top_y"])
            print(f"  {name}: separabilidad por altura = {sep:.3f} "
                  f"({'buena' if sep > 0.1 else 'debil, esta camara votara poco'})")
        else:
            print(f"  {name}: faltan capturas (F y S) para esta camara")


if __name__ == "__main__":
    main()
