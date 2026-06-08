#!/usr/bin/env python3
"""
Calibracion de camara con tablero de ajedrez (checkerboard).
Version mejorada: filtro de movimiento, rechazo de outliers, guia de cobertura.

Uso:
  python calibrate_camera.py --device /dev/video2 --auto
  python calibrate_camera.py --device /dev/video2 --chess-cols 7 --chess-rows 5

Teclas:
  C     -> capturar frame (solo si el tablero se movio lo suficiente)
  SPACE -> captura automatica cuando hay suficiente movimiento
  Q     -> calcular calibracion con rechazo de outliers y guardar
  R     -> resetear capturas

Estrategia para RMS bajo (<0.5):
  - Captura 15-25 poses MUY DISTINTAS
  - Varia: distancia, inclinacion X, inclinacion Y, rotacion
  - Llena todos los cuadrantes de la imagen con el tablero
  - Evita capturas con blur o angulo extremo (>60 grados)
"""

import argparse
import json
import time

import cv2
import numpy as np


CHESS_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_FAST_CHECK
)

AUTO_SIZES = [
    (7, 5), (8, 6), (6, 5), (9, 6), (6, 4), (5, 4),
    (10, 7), (9, 7), (11, 8),
]

# Umbral de movimiento minimo entre capturas (pixeles promedio de desplazamiento de esquinas)
MIN_CORNER_MOVEMENT = 20.0

# Umbral para rechazar capturas con error alto en post-procesado
OUTLIER_THRESHOLD_MULTIPLIER = 1.5  # rechaza si error > media * factor


def open_camera(path, width, height):
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir la camara: {path}")
    for _ in range(10):
        cap.read()
        time.sleep(0.02)
    return cap


def preprocess(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def find_board(gray, chess_size):
    proc = preprocess(gray)
    found, corners = cv2.findChessboardCorners(proc, chess_size, CHESS_FLAGS)
    if found:
        return found, corners
    found, corners = cv2.findChessboardCorners(gray, chess_size, CHESS_FLAGS)
    if found:
        return found, corners
    found, corners = cv2.findChessboardCorners(gray, chess_size, None)
    return found, corners


def find_board_auto(gray):
    for size in AUTO_SIZES:
        found, corners = find_board(gray, size)
        if found:
            return found, corners, size
    return False, None, None


def make_objp(size, square_size):
    objp = np.zeros((size[0] * size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


def corners_moved_enough(new_corners, last_corners, threshold):
    """Verifica que las esquinas se hayan movido lo suficiente respecto a la ultima captura."""
    if last_corners is None:
        return True
    if new_corners.shape != last_corners.shape:
        return True
    movement = np.mean(np.linalg.norm(new_corners.reshape(-1, 2) - last_corners.reshape(-1, 2), axis=1))
    return movement >= threshold


def compute_reprojection_errors(obj_points, img_points, camera_matrix, dist_coeffs, rvecs, tvecs):
    """Calcula el error de reproyeccion por captura."""
    errors = []
    for i, (objp, imgp) in enumerate(zip(obj_points, img_points)):
        projected, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = cv2.norm(imgp, projected, cv2.NORM_L2) / len(projected)
        errors.append(err)
    return np.array(errors)


def calibrate_with_outlier_rejection(obj_points, img_points, img_size, square_size, chess_size, max_iter=5):
    """
    Calibra iterativamente rechazando capturas con error de reproyeccion alto.
    Retorna (rms, camera_matrix, dist_coeffs, rvecs, tvecs, indices_usados)
    """
    indices = list(range(len(obj_points)))

    for iteration in range(max_iter):
        current_obj = [obj_points[i] for i in indices]
        current_img = [img_points[i] for i in indices]

        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            current_obj, current_img, img_size, None, None,
            flags=cv2.CALIB_RATIONAL_MODEL  # modelo mas completo para distorsion
        )

        errors = compute_reprojection_errors(current_obj, current_img, K, D, rvecs, tvecs)
        mean_err = np.mean(errors)
        threshold = mean_err * OUTLIER_THRESHOLD_MULTIPLIER

        outliers = np.where(errors > threshold)[0]
        if len(outliers) == 0:
            print(f"  Convergido en iteracion {iteration+1}: RMS={rms:.4f}, capturas={len(indices)}")
            return rms, K, D, rvecs, tvecs, indices

        # Eliminar solo el peor outlier por iteracion para no perder demasiadas
        worst = outliers[np.argmax(errors[outliers])]
        removed_idx = indices[worst]
        print(f"  Iter {iteration+1}: RMS={rms:.4f}, eliminando captura {removed_idx+1} (err={errors[worst]:.3f})")
        indices.pop(worst)

        if len(indices) < 5:
            print("  Quedan menos de 5 capturas validas, deteniendo.")
            break

    return rms, K, D, rvecs, tvecs, indices


def draw_coverage_grid(display, img_points_list, w, h, grid=4):
    """Dibuja una cuadricula mostrando que zonas de la imagen han sido cubiertas."""
    cell_w, cell_h = w // grid, h // grid
    covered = np.zeros((grid, grid), dtype=bool)

    for corners in img_points_list:
        cx = int(np.mean(corners[:, 0, 0]))
        cy = int(np.mean(corners[:, 0, 1]))
        gx = min(cx // cell_w, grid - 1)
        gy = min(cy // cell_h, grid - 1)
        covered[gy, gx] = True

    for gy in range(grid):
        for gx in range(grid):
            x1, y1 = gx * cell_w, gy * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            color = (0, 180, 0) if covered[gy, gx] else (60, 60, 60)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

    pct = int(100 * covered.sum() / (grid * grid))
    return pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video2")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--chess-cols", type=int, default=7,
                    help="Esquinas interiores en columnas (default 7)")
    ap.add_argument("--chess-rows", type=int, default=5,
                    help="Esquinas interiores en filas (default 5)")
    ap.add_argument("--square-size", type=float, default=0.025,
                    help="Tamano del cuadro en metros (default 0.025 = 25mm)")
    ap.add_argument("--out", default="camera_calibration.json")
    ap.add_argument("--auto", action="store_true",
                    help="Auto-detectar tamano del tablero")
    ap.add_argument("--min-movement", type=float, default=MIN_CORNER_MOVEMENT,
                    help=f"Movimiento minimo de esquinas para captura (px, default {MIN_CORNER_MOVEMENT})")
    ap.add_argument("--max-captures", type=int, default=30,
                    help="Maximo de capturas a tomar (default 30, mas no siempre es mejor)")
    args = ap.parse_args()

    chess_size = (args.chess_cols, args.chess_rows)
    auto_mode = args.auto

    obj_points = []
    img_points = []
    last_corners = None
    auto_capture = False  # modo captura automatica con SPACE

    cap = open_camera(args.device, args.width, args.height)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if auto_mode:
        print(f"\nModo AUTO: probando tamanios {AUTO_SIZES}")
    else:
        print(f"\nCalibrando camara {args.device}")
        print(f"Tablero: {chess_size[0]}x{chess_size[1]} esquinas interiores")
    print(f"Movimiento minimo entre capturas: {args.min_movement:.0f} px")
    print("C = capturar | SPACE = auto-captura | Q = guardar | R = resetear\n")
    print("CONSEJO: Toma 15-25 capturas MUY DISTINTAS en angulo, distancia y posicion.\n")

    win = "calibracion (C=capturar, SPACE=auto, Q=guardar)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if auto_mode:
            found, corners, detected_size = find_board_auto(gray)
            current_size = detected_size if (found and detected_size) else chess_size
        else:
            found, corners = find_board(gray, chess_size)
            current_size = chess_size

        # Comprobar si hay suficiente movimiento para capturar
        enough_movement = found and corners_moved_enough(corners, last_corners, args.min_movement)

        display = frame.copy()
        h_img, w_img = display.shape[:2]

        # Cuadricula de cobertura
        if img_points:
            cov_pct = draw_coverage_grid(display, img_points, w_img, h_img)
        else:
            cov_pct = 0

        if found:
            cv2.drawChessboardCorners(display, current_size, corners, found)
            if enough_movement:
                status = f"LISTO - presiona C ({current_size[0]}x{current_size[1]})"
                color = (0, 255, 0)
                # Auto-captura si SPACE activo
                if auto_capture and len(obj_points) < args.max_captures:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    obj_points.append(make_objp(current_size, args.square_size))
                    img_points.append(corners2)
                    last_corners = corners2.copy()
                    if auto_mode and detected_size:
                        chess_size = detected_size
                        auto_mode = False
                    print(f"Auto-captura {len(obj_points)} ({current_size[0]}x{current_size[1]})")
                    time.sleep(0.5)  # pausa entre auto-capturas
            else:
                status = "Mueve mas el tablero..."
                color = (0, 165, 255)
        else:
            status = "Buscando tablero... (mejor luz, menos angulo)"
            color = (0, 0, 255)

        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, f"Capturas: {len(obj_points)}/{args.max_captures} | Cobertura: {cov_pct}%",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        auto_str = "AUTO-ON" if auto_capture else "auto-off"
        cv2.putText(display, f"[SPACE={auto_str}] [R=reset] [Q=calibrar]",
                    (10, h_img - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c") and found and enough_movement:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(make_objp(current_size, args.square_size))
            img_points.append(corners2)
            last_corners = corners2.copy()
            if auto_mode and detected_size:
                chess_size = detected_size
                auto_mode = False
                print(f"Tamano fijado a {chess_size[0]}x{chess_size[1]}")
            print(f"Captura {len(obj_points)} guardada ({current_size[0]}x{current_size[1]})")

        elif key == ord(" "):
            auto_capture = not auto_capture
            print(f"Auto-captura: {'ON' if auto_capture else 'OFF'}")

        elif key == ord("r"):
            obj_points.clear()
            img_points.clear()
            last_corners = None
            print("Capturas reseteadas.")

        elif key == ord("q"):
            if len(obj_points) < 5:
                print(f"Necesitas al menos 5 capturas (tienes {len(obj_points)})")
                continue
            break

    cv2.destroyAllWindows()
    cap.release()

    print(f"\n=== Calibrando con rechazo de outliers ({len(obj_points)} capturas iniciales) ===")
    img_size = (args.width, args.height)

    rms, camera_matrix, dist_coeffs, rvecs, tvecs, used_indices = calibrate_with_outlier_rejection(
        obj_points, img_points, img_size, args.square_size, chess_size
    )

    n_used = len(used_indices)
    n_removed = len(obj_points) - n_used
    print(f"\nResultado final:")
    print(f"  Capturas usadas:   {n_used} (eliminadas: {n_removed} outliers)")
    print(f"  RMS error:         {rms:.4f} (bueno < 0.5, aceptable < 1.0)")
    print(f"  Matriz de camara:\n{camera_matrix}")
    print(f"  Distorsion: {dist_coeffs.ravel()}")

    # Errores individuales de las capturas finales
    used_obj = [obj_points[i] for i in used_indices]
    used_img = [img_points[i] for i in used_indices]
    errors = compute_reprojection_errors(used_obj, used_img, camera_matrix, dist_coeffs, rvecs, tvecs)
    print(f"\n  Error por captura (px): min={errors.min():.3f} max={errors.max():.3f} mean={errors.mean():.3f}")

    calib = {
        "rms_error": float(rms),
        "image_width": args.width,
        "image_height": args.height,
        "chess_cols": chess_size[0],
        "chess_rows": chess_size[1],
        "square_size_m": args.square_size,
        "captures_used": n_used,
        "captures_removed_as_outliers": n_removed,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
    }
    with open(args.out, "w") as f:
        json.dump(calib, f, indent=2)

    print(f"\nCalibracion guardada en {args.out}")

    if rms < 0.5:
        print("Excelente calibracion!")
    elif rms < 1.0:
        print("Calibracion aceptable. Puedes mejorar con poses mas diversas.")
    else:
        print("ADVERTENCIA: RMS > 1.0. Consejos:")
        print("  - Usa solo 15-20 capturas MUY DISTINTAS entre si")
        print("  - Varia angulo (hasta 45 grados), distancia y posicion en la imagen")
        print("  - Asegurate que el tablero este completamente visible y sin blur")
        print("  - Imprime el tablero en papel rigido (no curvo)")
        print("  - Mejora la iluminacion: luz difusa, sin reflejo")


if __name__ == "__main__":
    main()
