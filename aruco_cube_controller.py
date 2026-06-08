#!/usr/bin/env python3
"""
Detector de orientacion del cubo ArUco + lanzador de politica lerobot.

Uso:
  python aruco_cube_controller.py \
      --calib camera_calibration.json \
      --device /dev/video2 \
      --zones zones.json \
      --policy-left  /home/usuario/outputs/train/act_aruco_left/checkpoints/last/pretrained_model \
      --policy-right /home/usuario/outputs/train/act_aruco_right/checkpoints/last/pretrained_model \
      --policy-center /home/usuario/outputs/train/act_aruco_center/checkpoints/last/pretrained_model \
      --follower-port /dev/ttyACM0 \
      --scene-cam /dev/video2 \
      --side-cam /dev/video4

Modo calibracion de orientaciones:
  python aruco_cube_controller.py --calib camera_calibration.json --device /dev/video2 --calibrate-orientations

Teclas:
  ESPACIO -> lanzar politica segun orientacion confirmada
  Q       -> salir
"""

import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np


ZONE_COLORS = {
    "izquierda": (255, 80,  80),
    "centro":    (80,  255, 80),
    "derecha":   (80,  80,  255),
}


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


def get_aruco_detector(dict_name="DICT_4X4_50"):
    dict_id = getattr(cv2.aruco, dict_name, None)
    if dict_id is None:
        raise ValueError(f"Diccionario no encontrado: {dict_name}")
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        use_new_api = True
    except AttributeError:
        detector = (aruco_dict, params)
        use_new_api = False
    return detector, use_new_api


def detect_aruco(detector, use_new_api, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if use_new_api:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        aruco_dict, params = detector
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    return corners, ids


def compute_orientation_axis(rvec):
    """Eje X del marcador en frame de camara — cambia al rotar el cubo."""
    R, _ = cv2.Rodrigues(rvec)
    axis = R[:, 0]
    return axis / np.linalg.norm(axis)


def classify(axis, thresholds):
    if thresholds is None:
        nx, ny, nz = axis
        if nx < -0.5:
            return "izquierda"
        elif nx > 0.5:
            return "derecha"
        elif ny > 0.4:
            return "centro"
        return "desconocido"
    best_label, best_dot = "desconocido", -2.0
    for label, ref in thresholds.items():
        dot = float(np.dot(axis, np.array(ref)))
        if dot > best_dot:
            best_dot = dot
            best_label = label
    return best_label if best_dot >= 0.7 else "desconocido"


def draw_zones(frame, zones, zone_radius, active=None):
    overlay = frame.copy()
    for name, pos in zones.items():
        color = ZONE_COLORS.get(name, (200, 200, 200))
        r = zone_radius + (15 if name == active else 0)
        cv2.circle(overlay, tuple(pos), r, color, -1)
        cv2.circle(frame, tuple(pos), r, color, 2)
        cv2.putText(frame, name.upper(), (pos[0] - 35, pos[1] + r + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)


def calibrate_orientations_mode(cap, detector, use_new_api,
                                 camera_matrix, dist_coeffs, marker_size, out_path):
    labels = {"1": "izquierda", "2": "derecha", "3": "centro"}
    captured = {}

    print("\n=== CALIBRACION DE ORIENTACIONES ===")
    print("Rota el cubo 90 grados para cada posicion:")
    print("  1 -> IZQUIERDA | 2 -> DERECHA | 3 -> CENTRO | Q -> guardar\n")

    win = "calibrar (1=izq, 2=der, 3=centro, Q=guardar)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    obj_pts = np.array([[-marker_size/2, marker_size/2, 0],
                        [ marker_size/2, marker_size/2, 0],
                        [ marker_size/2,-marker_size/2, 0],
                        [-marker_size/2,-marker_size/2, 0]], dtype=np.float32)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        corners, ids = detect_aruco(detector, use_new_api, frame)
        display = frame.copy()
        axis = None

        if ids is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            _, rvec, tvec = cv2.solvePnP(obj_pts, corners[0][0], camera_matrix, dist_coeffs)
            tvec = tvec.ravel()
            cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, rvec, tvec, marker_size * 0.5)
            axis = compute_orientation_axis(rvec)
            nx, ny, nz = axis
            cv2.putText(display, f"Eje X: x={nx:.2f} y={ny:.2f} z={nz:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Buscando ArUco...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        for i, lbl in enumerate(captured):
            cv2.putText(display, f"OK: {lbl}", (10, 65 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(win, display)
        key = chr(cv2.waitKey(1) & 0xFF)

        if key in labels and axis is not None:
            label = labels[key]
            captured[label] = axis.tolist()
            print(f"Guardado: {label} -> {axis.round(3)}")
        elif key == "q":
            break

    cv2.destroyWindow(win)
    with open(out_path, "w") as f:
        json.dump(captured, f, indent=2)
    print(f"Orientaciones guardadas en {out_path}")
    return captured


def launch_policy(orientation, args):
    policy_map = {
        "izquierda": args.policy_left,
        "derecha":   args.policy_right,
        "centro":    args.policy_center,
    }
    policy_path = policy_map.get(orientation, "")
    if not policy_path:
        print(f"Sin politica para '{orientation}'.")
        return
    if not os.path.exists(policy_path):
        print(f"Ruta no encontrada: {policy_path}")
        return

    cameras = (
        f"{{scene: {{type: opencv, index_or_path: {args.scene_cam}, "
        f"width: {args.width}, height: {args.height}, fps: 30, fourcc: MJPG, backend: V4L2}}, "
        f"side: {{type: opencv, index_or_path: {args.side_cam}, "
        f"width: {args.width}, height: {args.height}, fps: 30, fourcc: MJPG, backend: V4L2}}}}"
    )
    task_map = {
        "izquierda": "Pon el cubo en el post-it de la izquierda",
        "derecha":   "Pon el cubo en el post-it de la derecha",
        "centro":    "Pon el cubo en el post-it del centro",
    }
    cmd = [
        "lerobot-rollout",
        "--robot.type=so101_follower",
        f"--robot.port={args.follower_port}",
        "--robot.id=my_awesome_follower_arm",
        f"--robot.cameras={cameras}",
        "--policy.type=act",
        f"--policy.pretrained_path={policy_path}",
        "--strategy.type=sentry",
        f"--task={task_map[orientation]}",
        "--dataset.repo_id=Dravid419/rollout_aruco_cube",
        "--dataset.push_to_hub=false",
        "--duration=60",
    ]
    print(f"Lanzando politica '{orientation}'...")
    # input=b"\n" acepta automaticamente la pregunta de calibracion
    subprocess.run(cmd, input=b"\n")
    print("Rollout terminado.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="camera_calibration.json")
    ap.add_argument("--device", default="/dev/video2")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--aruco-dict", default="DICT_4X4_50")
    ap.add_argument("--marker-size", type=float, default=0.05)
    ap.add_argument("--orientation-calib", default="orientation_calibration.json")
    ap.add_argument("--calibrate-orientations", action="store_true")
    ap.add_argument("--zones", default="zones.json")
    ap.add_argument("--policy-left", default="")
    ap.add_argument("--policy-right", default="")
    ap.add_argument("--policy-center", default="")
    ap.add_argument("--follower-port", default="/dev/ttyACM0")
    ap.add_argument("--scene-cam", default="/dev/video2")
    ap.add_argument("--side-cam", default="/dev/video4")
    args = ap.parse_args()

    if not os.path.exists(args.calib):
        print(f"ERROR: No se encontro {args.calib}")
        sys.exit(1)
    with open(args.calib) as f:
        calib = json.load(f)
    camera_matrix = np.array(calib["camera_matrix"])
    dist_coeffs = np.array(calib["dist_coeffs"])
    print(f"Calibracion cargada (RMS={calib.get('rms_error', '?'):.3f})")

    thresholds = None
    if os.path.exists(args.orientation_calib):
        with open(args.orientation_calib) as f:
            thresholds = json.load(f)
        print(f"Orientaciones cargadas: {list(thresholds.keys())}")

    zones, zone_radius = {}, 40
    if os.path.exists(args.zones):
        with open(args.zones) as f:
            zones_raw = json.load(f)
        zone_radius = zones_raw.pop("radius", 40)
        zones = {k: v for k, v in zones_raw.items()}
        print(f"Zonas cargadas: {list(zones.keys())}")

    cap = open_camera(args.device, args.width, args.height)
    detector, use_new_api = get_aruco_detector(args.aruco_dict)
    print(f"Detector ArUco listo ({args.aruco_dict})")

    if args.calibrate_orientations:
        thresholds = calibrate_orientations_mode(
            cap, detector, use_new_api,
            camera_matrix, dist_coeffs,
            args.marker_size, args.orientation_calib
        )

    obj_pts = np.array([[-args.marker_size/2, args.marker_size/2, 0],
                        [ args.marker_size/2, args.marker_size/2, 0],
                        [ args.marker_size/2,-args.marker_size/2, 0],
                        [-args.marker_size/2,-args.marker_size/2, 0]], dtype=np.float32)

    win = "ArUco detector (ESPACIO=lanzar, Q=salir)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    history, HISTORY_LEN, confirmed = [], 10, None

    print("\nDetectando... Presiona ESPACIO para lanzar la politica.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        corners, ids = detect_aruco(detector, use_new_api, frame)
        display = frame.copy()
        current = None

        if ids is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(display, corners, ids)
            _, rvec, tvec = cv2.solvePnP(obj_pts, corners[0][0], camera_matrix, dist_coeffs)
            tvec = tvec.ravel()
            cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, rvec, tvec, args.marker_size * 0.5)

            axis = compute_orientation_axis(rvec)
            nx, ny, nz = axis

            current = classify(axis, thresholds)
            history.append(current)
            if len(history) > HISTORY_LEN:
                history.pop(0)
            if len(history) == HISTORY_LEN:
                counts = {}
                for o in history:
                    counts[o] = counts.get(o, 0) + 1
                dom = max(counts, key=counts.get)
                confirmed = dom if counts[dom] >= HISTORY_LEN * 0.7 else None

            color = ZONE_COLORS.get(current, (128, 128, 128))
            cv2.putText(display, f"Eje X: x={nx:.2f} y={ny:.2f} z={nz:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(display, f"Orientacion: {current}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display, f"Confirmada:  {confirmed or '-'}",
                        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if confirmed else (128, 128, 128), 2)
        else:
            history.clear()
            confirmed = None
            cv2.putText(display, "Buscando ArUco...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if zones:
            draw_zones(display, zones, zone_radius,
                       active=confirmed if confirmed != "desconocido" else None)

        cv2.putText(display, "ESPACIO=lanzar | Q=salir",
                    (10, args.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if confirmed and confirmed != "desconocido":
                any_policy = any([args.policy_left, args.policy_right, args.policy_center])
                if any_policy:
                    # Liberar camara antes del rollout para evitar conflicto
                    cap.release()
                    cv2.destroyAllWindows()
                    launch_policy(confirmed, args)
                    # Reabrir camara y ventana despues del rollout
                    cap = open_camera(args.device, args.width, args.height)
                    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                    history.clear()
                    confirmed = None
                else:
                    print(f"Orientacion: {confirmed} (sin politicas configuradas)")
            else:
                print("Orientacion no confirmada, espera un momento.")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()
    print("Detector cerrado.")


if __name__ == "__main__":
    main()