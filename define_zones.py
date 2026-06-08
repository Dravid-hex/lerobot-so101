#!/usr/bin/env python3
"""
Herramienta para definir y visualizar las 3 zonas objetivo (izquierda, centro, derecha)
en el feed de la camara. Las zonas se guardan en zones.json y se muestran como
circulos semitransparentes en el controller.

Uso:
  python define_zones.py --device /dev/video2 --width 1280 --height 720

Instrucciones:
  1. Se abre la camara en vivo.
  2. Haz click en cada marca negra de la mesa en este orden:
       Primer click  -> zona IZQUIERDA
       Segundo click -> zona CENTRO
       Tercer click  -> zona DERECHA
  3. Presiona Q para guardar y salir.
  4. Usa zones.json con aruco_cube_controller.py --zones zones.json
"""

import argparse
import json
import time

import cv2
import numpy as np

# Colores por zona (BGR)
ZONE_COLORS = {
    "izquierda": (255, 80,  80),   # azul
    "centro":    (80,  255, 80),   # verde
    "derecha":   (80,  80,  255),  # rojo
}
ZONE_LABELS = ["izquierda", "centro", "derecha"]


def open_camera(path, width, height):
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir camara: {path}")
    for _ in range(10):
        cap.read()
        time.sleep(0.02)
    return cap


def draw_zones(frame, zones, active_zone=None, radius=40, alpha=0.4):
    """Dibuja circulos semitransparentes en las zonas definidas."""
    overlay = frame.copy()
    for name, pos in zones.items():
        color = ZONE_COLORS.get(name, (200, 200, 200))
        thickness = -1  # relleno
        cv2.circle(overlay, tuple(pos), radius, color, thickness)
        cv2.circle(frame, tuple(pos), radius, color, 2)  # borde solido

        # Si es la zona activa, hacer mas grande y brillante
        if name == active_zone:
            cv2.circle(overlay, tuple(pos), radius + 10, color, -1)
            cv2.circle(frame, tuple(pos), radius + 10, (255, 255, 255), 3)

        # Etiqueta
        label_pos = (pos[0] - 40, pos[1] + radius + 20)
        cv2.putText(frame, name.upper(), label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Blend semitransparente
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video2")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", default="zones.json")
    ap.add_argument("--radius", type=int, default=40,
                    help="Radio del circulo de zona en pixeles")
    args = ap.parse_args()

    cap = open_camera(args.device, args.width, args.height)

    zones = {}
    click_queue = list(ZONE_LABELS)  # orden de clicks

    win = "Definir zonas - Haz click en cada marca negra"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and click_queue:
            label = click_queue.pop(0)
            zones[label] = [x, y]
            print(f"Zona '{label}' definida en ({x}, {y})")

    cv2.setMouseCallback(win, on_click)

    print("\n=== DEFINIR ZONAS ===")
    print("Haz click en las marcas negras en este orden:")
    print("  1er click -> IZQUIERDA")
    print("  2do click -> CENTRO")
    print("  3er click -> DERECHA")
    print("Presiona Q para guardar.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        display = frame.copy()

        # Mostrar zonas ya definidas
        if zones:
            draw_zones(display, zones, radius=args.radius)

        # Instruccion actual
        if click_queue:
            msg = f"Click en la marca: {click_queue[0].upper()}"
            cv2.putText(display, msg, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        else:
            cv2.putText(display, "Todas las zonas definidas - presiona Q para guardar",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()

    if len(zones) < 3:
        print(f"Solo definiste {len(zones)} zonas. Necesitas 3.")
    else:
        zones["radius"] = args.radius
        with open(args.out, "w") as f:
            json.dump(zones, f, indent=2)
        print(f"\nZonas guardadas en {args.out}:")
        for k, v in zones.items():
            print(f"  {k}: {v}")
        print(f"\nUsa: python aruco_cube_controller.py --zones {args.out}")


if __name__ == "__main__":
    main()
