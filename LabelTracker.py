import cv2
import os
import tkinter as tk
from tkinter import simpledialog, StringVar, OptionMenu, messagebox
from PIL import Image, ImageTk
import threading
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import Counter
import random  # Importar random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Directorios para guardar imágenes y etiquetas
images_dir = "images"
labels_dir = "labels"
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Variables globales
trackers = []
bboxes = []
instance_ids = []
class_names = []
kalman_filters = []
optical_flow_points = []
initial_histograms = []
colors = {}
tracking = False
current_bbox = None
current_class = ""
recording = False
frame_count = 0
frame = None
clean_frame = None
prev_gray = None
camera_index = 0
unique_id_counter = 0
HIST_SIMILARITY_THRESHOLD = 1

TRACKER_TYPE = "CSRT"
REDETECTION_INTERVAL = 5
MAX_ALLOWED_DRIFT = 60

orb = cv2.ORB_create()
lk_params = dict(winSize=(15, 15), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Variable para controlar el intervalo de muestreo en milisegundos
SAMPLE_INTERVAL_MS = 500  # Por defecto, 500 ms (2 muestras por segundo)

# Variables globales para revisión y navegación
review_mode = False
image_list = []
current_image_index = 0
selected_bbox_index = None  # Índice de la etiqueta seleccionada para edición
resize_mode = False  # Indica si se está redimensionando una etiqueta
resize_corner = None  # Esquina que se está ajustando

# Funciones para el filtro de Kalman
def create_kalman_filter():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 0.01, 0],
                                       [0, 0, 0, 0.01]], np.float32) * 0.03
    return kalman

# Funciones auxiliares
def calculate_histogram(frame, bbox):
    x, y, w, h = map(int, bbox)
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return None
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_roi], [0, 1], None,
                        [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist

def is_histogram_similar(hist1, hist2, threshold=HIST_SIMILARITY_THRESHOLD):
    if hist1 is None or hist2 is None:
        return False
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return similarity < (1 - threshold)

def is_overlapping(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    overlap_x = max(0, min(x1_max, x2_max) - max(x1, x2))
    overlap_y = max(0, min(y1_max, y2_max) - max(y1, y2))
    overlap_area = overlap_x * overlap_y

    area1 = w1 * h1
    area2 = w2 * h2

    return overlap_area > 0.5 * min(area1, area2)

def save_labels(filename, bboxes, class_names, width, height, timestamp):
    with open(filename, 'w') as f:
        for bbox, class_name in zip(bboxes, class_names):
            x, y, w, h = bbox
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height
            f.write(f"{class_name} {x_center} {y_center} {w_norm} {h_norm}\n")

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def create_tracker():
    if TRACKER_TYPE == "CSRT":
        return cv2.legacy.TrackerCSRT_create()
    elif TRACKER_TYPE == "KCF":
        return cv2.legacy.TrackerKCF_create()
    elif TRACKER_TYPE == "MIL":
        return cv2.legacy.TrackerMIL_create()
    elif TRACKER_TYPE == "MedianFlow":
        return cv2.legacy.TrackerMedianFlow_create()
    elif TRACKER_TYPE == "MOSSE":
        return cv2.legacy.TrackerMOSSE_create()
    else:
        raise ValueError(f"Tipo de tracker desconocido: {TRACKER_TYPE}")

def change_camera(selected_index):
    global cap, camera_index, frame
    camera_index = int(selected_index)
    cap.release()
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    if not ret:
        print(f"No se pudo abrir la cámara {camera_index}")

# ... (el código inicial permanece igual)

def on_mouse_down(event):
    global current_bbox, tracking, selected_bbox_index, start_x, start_y, resize_mode, resize_corner
    if review_mode:
        start_x, start_y = event.x, event.y
        selected_bbox_index = None
        resize_mode = False
        resize_corner = None
        # Verificar si se ha hecho clic en alguna esquina de una etiqueta para redimensionar
        for i, bbox in enumerate(bboxes):
            x, y, w, h = bbox
            corners = {
                'tl': (x, y),
                'tr': (x + w, y),
                'bl': (x, y + h),
                'br': (x + w, y + h)
            }
            for corner_name, (cx, cy) in corners.items():
                if abs(event.x - cx) <= 5 and abs(event.y - cy) <= 5:
                    selected_bbox_index = i
                    resize_mode = True
                    resize_corner = corner_name
                    tracking = True  # **Agregamos esta línea**
                    break
            if resize_mode:
                break
        if not resize_mode:
            # Verificar si se ha hecho clic dentro de una etiqueta para moverla
            for i, bbox in enumerate(bboxes):
                x, y, w, h = bbox
                if x <= event.x <= x + w and y <= event.y <= y + h:
                    selected_bbox_index = i
                    tracking = True  # Iniciar movimiento de etiqueta
                    break
            if selected_bbox_index is None:
                # Crear nueva etiqueta
                current_bbox = (event.x, event.y, 1, 1)
                tracking = True
    else:
        current_bbox = (event.x, event.y, 1, 1)
        tracking = True

# ... (el resto del código permanece igual)

def on_mouse_move(event):
    global current_bbox, tracking, selected_bbox_index, resize_mode, resize_corner
    if tracking:
        if review_mode:
            if selected_bbox_index is not None:
                if resize_mode:
                    # Redimensionar etiqueta
                    x, y, w, h = bboxes[selected_bbox_index]
                    if resize_corner == 'tl':
                        new_x = event.x
                        new_y = event.y
                        new_w = (x + w) - new_x
                        new_h = (y + h) - new_y
                        bboxes[selected_bbox_index] = [new_x, new_y, new_w, new_h]
                    elif resize_corner == 'tr':
                        new_x = x
                        new_y = event.y
                        new_w = event.x - x
                        new_h = (y + h) - new_y
                        bboxes[selected_bbox_index] = [new_x, new_y, new_w, new_h]
                    elif resize_corner == 'bl':
                        new_x = event.x
                        new_y = y
                        new_w = (x + w) - new_x
                        new_h = event.y - y
                        bboxes[selected_bbox_index] = [new_x, new_y, new_w, new_h]
                    elif resize_corner == 'br':
                        new_x = x
                        new_y = y
                        new_w = event.x - x
                        new_h = event.y - y
                        bboxes[selected_bbox_index] = [new_x, new_y, new_w, new_h]
                else:
                    # Mover etiqueta existente
                    dx = event.x - start_x
                    dy = event.y - start_y
                    x, y, w, h = bboxes[selected_bbox_index]
                    bboxes[selected_bbox_index] = [x + dx, y + dy, w, h]
                    start_x, start_y = event.x, event.y
            else:
                # Ajustar nueva etiqueta
                current_bbox = (current_bbox[0], current_bbox[1],
                                event.x - current_bbox[0], event.y - current_bbox[1])
            draw_labels_on_frame()
        else:
            current_bbox = (current_bbox[0], current_bbox[1],
                            event.x - current_bbox[0], event.y - current_bbox[1])

def on_mouse_up(event):
    global current_bbox, tracking, optical_flow_points, prev_gray, unique_id_counter, selected_bbox_index, resize_mode, resize_corner
    if tracking or resize_mode:
        if review_mode:
            if selected_bbox_index is not None:
                # Finalizar movimiento o redimensionamiento de etiqueta
                tracking = False
                resize_mode = False
                resize_corner = None
                selected_bbox_index = None
                save_current_labels()
            else:
                # Finalizar creación de nueva etiqueta
                if current_bbox[2] > 10 and current_bbox[3] > 10:
                    if not current_class:
                        messagebox.showerror("Error", "Por favor, establece una clase antes de marcar el objeto.")
                        current_bbox = None
                        tracking = False
                        return
                    bboxes.append([int(current_bbox[0]), int(current_bbox[1]), int(current_bbox[2]), int(current_bbox[3])])
                    class_names.append(current_class)
                    if current_class not in colors:
                        colors[current_class] = get_random_color()
                    update_class_list()
                    save_current_labels()
                current_bbox = None
                tracking = False
                draw_labels_on_frame()
        else:
            if current_bbox[2] > 10 and current_bbox[3] > 10:
                if not current_class:
                    messagebox.showerror("Error", "Por favor, establece una clase antes de marcar el objeto.")
                    current_bbox = None
                    tracking = False
                    return
                tracker = create_tracker()
                tracker.init(frame, tuple(map(int, current_bbox)))
                trackers.append(tracker)
                bboxes.append(tuple(map(int, current_bbox)))
                class_names.append(current_class)
                instance_ids.append(unique_id_counter)
                unique_id_counter += 1

                initial_hist = calculate_histogram(frame, current_bbox)
                initial_histograms.append(initial_hist)

                kalman_filters.append(create_kalman_filter())

                if current_class not in colors:
                    colors[current_class] = get_random_color()

                x, y, w, h = map(int, current_bbox)
                roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                keypoints, descriptors = orb.detectAndCompute(roi_gray, None)
                if keypoints is not None and descriptors is not None:
                    # Ajustar las coordenadas de los keypoints al marco completo
                    for kp in keypoints:
                        kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
                    optical_flow_points.append((keypoints, descriptors))
                else:
                    optical_flow_points.append((None, None))

                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                update_class_list()

            current_bbox = None
            tracking = False

# Grabación
def toggle_recording():
    global recording
    recording = not recording
    if recording:
        button_record.config(bg="#ff4444", text="Grabando...", fg="white")
        threading.Thread(target=record_loop).start()
    else:
        button_record.config(bg="#00c853", text="Iniciar Grabación", fg="white")

def record_loop():
    global frame_count, clean_frame
    while recording:
        if not recording:
            break
        if clean_frame is not None:
            timestamp = int(time.time() * 1000)  # Marca de tiempo en milisegundos
            img_clean = clean_frame.copy()
            img_filename = os.path.join(images_dir, f"frame_{frame_count}_{timestamp}.jpg")
            label_filename = os.path.join(labels_dir, f"frame_{frame_count}_{timestamp}.txt")
            
            # Guardar la imagen
            cv2.imwrite(img_filename, img_clean)
            
            # Guardar etiquetas junto con la marca de tiempo y dimensiones de imagen
            save_labels(label_filename, bboxes, class_names, clean_frame.shape[1], clean_frame.shape[0], timestamp)
            
            frame_count += 1
            # Esperar el intervalo definido por el usuario
            time.sleep(SAMPLE_INTERVAL_MS / 1000.0)  # Convertir a segundos

def set_class():
    global current_class
    current_class = entry_class.get()
    label_status.config(text=f"Clase actual: {current_class}")

def update_class_list():
    listbox_classes.delete(0, tk.END)
    unique_classes = set(class_names)
    for class_name in unique_classes:
        count = class_names.count(class_name)
        color = colors.get(class_name, (0, 255, 0))
        listbox_classes.insert(tk.END, f"{class_name} - ({count})")
        listbox_classes.itemconfig(tk.END, {'fg': '#%02x%02x%02x' % color})

def delete_selected_class():
    selected_index = listbox_classes.curselection()
    if selected_index:
        selected_text = listbox_classes.get(selected_index)
        class_name = selected_text.split(" - ")[0]
        indices_to_remove = [i for i, name in enumerate(class_names) if name == class_name]
        for index in sorted(indices_to_remove, reverse=True):
            del bboxes[index]
            del class_names[index]
        if class_name in colors:
            del colors[class_name]
        update_class_list()
        save_current_labels()
        draw_labels_on_frame()

def edit_selected_class():
    selected_index = listbox_classes.curselection()
    if selected_index:
        selected_text = listbox_classes.get(selected_index)
        old_class_name = selected_text.split(" - ")[0]
        new_class_name = simpledialog.askstring("Editar Clase", f"Nuevo nombre para la clase '{old_class_name}':")
        if new_class_name and new_class_name != old_class_name:
            # Actualizar en class_names
            for i in range(len(class_names)):
                if class_names[i] == old_class_name:
                    class_names[i] = new_class_name
            # Actualizar en colors
            colors[new_class_name] = colors.pop(old_class_name)
            update_class_list()
            save_current_labels()
            draw_labels_on_frame()

def clear_directories():
    answer = messagebox.askyesno("Confirmación", "¿Estás seguro de que deseas eliminar todos los archivos de las carpetas de imágenes y etiquetas?")
    if answer:
        for directory in [images_dir, labels_dir]:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    os.remove(file_path)
                    print(f"Eliminado: {file_path}")
                except Exception as e:
                    print(f"No se pudo eliminar {file_path}. Error: {e}")

def update_video():
    global frame, frame_count, prev_gray, optical_flow_points, clean_frame
    if review_mode:
        # En modo revisión, mostrar la imagen actual con las etiquetas
        if frame is not None:
            # Dibujar las etiquetas en la imagen
            draw_labels_on_frame()
        video_label.after(20, update_video)
        return

    ret, frame = cap.read()
    if ret:
        clean_frame = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_bboxes = []
        new_class_names = []
        new_histograms = []

        if current_bbox:
            x, y, w, h = map(int, current_bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Comprobación de longitudes
        if len(trackers) != len(optical_flow_points):
            print(f"Advertencia: trackers y optical_flow_points no tienen la misma longitud.")

        for i in range(len(trackers)):
            if i >= len(optical_flow_points):
                print(f"Advertencia: optical_flow_points tiene menos elementos que trackers.")
                continue

            tracker = trackers[i]
            bbox = bboxes[i]
            class_name = class_names[i]
            kalman = kalman_filters[i]
            keypoints_prev, descriptors_prev = optical_flow_points[i]
            initial_hist = initial_histograms[i]

            success, new_bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, new_bbox)
                current_hist = calculate_histogram(frame, new_bbox)

                # Validación del histograma
                if not is_histogram_similar(initial_hist, current_hist):
                    new_bbox = bbox
                    tracker.init(frame, bbox)
                    current_hist = initial_hist

                # Seguimiento con Optical Flow
                if keypoints_prev is not None and descriptors_prev is not None:
                    p0 = np.array([kp.pt for kp in keypoints_prev], dtype=np.float32).reshape(-1, 1, 2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]

                    if len(good_new) > 0:
                        dx = np.mean(good_new[:, 0] - good_old[:, 0])
                        dy = np.mean(good_new[:, 1] - good_old[:, 1])
                        x += int(dx)
                        y += int(dy)
                        new_bbox = (x, y, w, h)
                        # Actualizar keypoints
                        keypoints_new = [cv2.KeyPoint(pt[0], pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                                         for pt, kp in zip(good_new, keypoints_prev)]
                        optical_flow_points[i] = (keypoints_new, descriptors_prev)
                    else:
                        # Re-extraer keypoints si no hay buenos puntos
                        roi_gray = frame_gray[y:y+h, x:x+w]
                        keypoints_new, descriptors_new = orb.detectAndCompute(roi_gray, None)
                        if keypoints_new is not None and descriptors_new is not None:
                            for kp in keypoints_new:
                                kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
                            optical_flow_points[i] = (keypoints_new, descriptors_new)
                        else:
                            optical_flow_points[i] = (None, None)
                else:
                    # Re-extraer keypoints si no existen
                    roi_gray = frame_gray[y:y+h, x:x+w]
                    keypoints_new, descriptors_new = orb.detectAndCompute(roi_gray, None)
                    if keypoints_new is not None and descriptors_new is not None:
                        for kp in keypoints_new:
                            kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
                        optical_flow_points[i] = (keypoints_new, descriptors_new)
                    else:
                        optical_flow_points[i] = (None, None)

                # Dibujar el bounding box
                color = colors.get(class_name, (0, 255, 0))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, class_name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                new_bboxes.append((x, y, w, h))
                new_class_names.append(class_name)
                new_histograms.append(current_hist)
            else:
                # Si el tracker falla, mantener el bounding box anterior
                new_bboxes.append(bbox)
                new_class_names.append(class_name)
                new_histograms.append(initial_histograms[i])

        bboxes[:] = new_bboxes
        class_names[:] = new_class_names
        initial_histograms[:] = new_histograms
        frame_count += 1
        prev_gray = frame_gray.copy()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        video_label.imgtk = img
        video_label.config(image=img)

    video_label.after(20, update_video)

def close_program():
    global recording
    recording = False
    root.quit()
    cap.release()
    cv2.destroyAllWindows()

### Funciones adicionales: Monitoreo de etiquetas y gráfica en tiempo real ###
def count_labels():
    """
    Cuenta las etiquetas en los archivos de la carpeta de etiquetas.
    """
    label_counts = Counter()
    if not os.path.exists(labels_dir):
        print(f"La carpeta '{labels_dir}' no existe.")
        return label_counts

    for label_file in os.listdir(labels_dir):
        if label_file.endswith(".txt"):
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, "r") as file:
                for line in file:
                    if not line.startswith("#"):  # Ignorar comentarios
                        label = line.split()[0]  # Primera palabra es la etiqueta
                        label_counts[label] += 1
    return label_counts

### Funciones para el modo de revisión ###
def load_image_and_labels(index):
    """Carga la imagen y sus etiquetas correspondientes."""
    global frame, image_list, labels_dir, bboxes, class_names
    if 0 <= index < len(image_list):
        img_path = os.path.join(images_dir, image_list[index])
        label_path = os.path.join(labels_dir, image_list[index].replace(".jpg", ".txt"))

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"No se pudo cargar la imagen: {img_path}")
            return
        
        # Cargar las etiquetas
        bboxes = []
        class_names = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) != 5:
                        continue  # Saltar líneas mal formateadas
                    class_name = data[0]
                    x_center, y_center, w_norm, h_norm = map(float, data[1:])
                    w = w_norm * frame.shape[1]
                    h = h_norm * frame.shape[0]
                    x = x_center * frame.shape[1] - w / 2
                    y = y_center * frame.shape[0] - h / 2
                    bboxes.append([int(x), int(y), int(w), int(h)])
                    class_names.append(class_name)
        else:
            print(f"No se encontró el archivo de etiquetas para {img_path}")

        update_class_list()
    else:
        print("Índice fuera de rango.")

def draw_labels_on_frame():
    """Dibuja las etiquetas actuales en el frame."""
    global frame, bboxes, class_names, colors, current_bbox
    if frame is None:
        return
    temp_frame = frame.copy()
    for bbox, class_name in zip(bboxes, class_names):
        x, y, w, h = bbox
        color = colors.get(class_name, (0, 255, 0))
        cv2.rectangle(temp_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(temp_frame, class_name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # Dibujar los puntos de las esquinas para redimensionar
        corners = [
            (x, y),  # Top-left
            (x + w, y),  # Top-right
            (x, y + h),  # Bottom-left
            (x + w, y + h)  # Bottom-right
        ]
        for corner in corners:
            cv2.circle(temp_frame, corner, 5, color, -1)
    if current_bbox and tracking:
        x, y, w, h = map(int, current_bbox)
        cv2.rectangle(temp_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    img = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    video_label.imgtk = img
    video_label.config(image=img)

def save_current_labels():
    """Guarda las etiquetas actuales al archivo correspondiente."""
    global image_list, current_image_index, bboxes, class_names, frame
    label_path = os.path.join(labels_dir, image_list[current_image_index].replace(".jpg", ".txt"))
    save_labels(label_path, bboxes, class_names, frame.shape[1], frame.shape[0], int(time.time() * 1000))

def next_image(event=None):
    """Carga la siguiente imagen en la lista."""
    global current_image_index
    save_current_labels()  # Guardar cambios antes de cambiar de imagen
    if current_image_index < len(image_list) - 1:
        current_image_index += 1
        load_image_and_labels(current_image_index)
    else:
        messagebox.showinfo("Información", "Esta es la última imagen.")

def prev_image(event=None):
    """Carga la imagen anterior en la lista."""
    global current_image_index
    save_current_labels()  # Guardar cambios antes de cambiar de imagen
    if current_image_index > 0:
        current_image_index -= 1
        load_image_and_labels(current_image_index)
    else:
        messagebox.showinfo("Información", "Esta es la primera imagen.")

def toggle_review_mode():
    """Activa o desactiva el modo de revisión."""
    global review_mode, image_list, current_image_index
    review_mode = not review_mode
    if review_mode:
        # Cargar lista de imágenes etiquetadas
        image_list = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
        if image_list:
            current_image_index = 0
            load_image_and_labels(current_image_index)
            label_status.config(text="Modo Revisión Activado")
        else:
            messagebox.showinfo("Información", "No hay imágenes etiquetadas para revisar.")
            review_mode = False
    else:
        label_status.config(text=f"Clase actual: {current_class}")

# Agregar bindings de teclado
def on_key_press(event):
    if review_mode:
        if event.keysym.lower() == 'd':
            next_image()
        elif event.keysym.lower() == 'a':
            prev_image()

root = tk.Tk()
root.title("Aplicación de Etiquetado con Gráfica en Tiempo Real")

# Iniciar la captura de video
cap = cv2.VideoCapture(camera_index)
ret, frame = cap.read()
if not ret:
    print(f"No se pudo abrir la cámara {camera_index}")

# Crear la figura de matplotlib
fig = plt.Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

def plot_labels():
    label_counts = count_labels()
    labels = list(label_counts.keys())
    counts = list(label_counts.values())

    ax.clear()
    ax.bar(labels, counts, color='blue', alpha=0.7)
    ax.set_xlabel("Etiquetas")
    ax.set_ylabel("Conteo")
    ax.set_title("Distribución de Etiquetas en Tiempo Real")
    canvas.draw()

    # Actualizar después de 1000 ms (1 segundo)
    root.after(1000, plot_labels)

# Estilos personalizados
style_frame_bg = "#252525"
style_frame_highlight = "#3c3c3c"
style_button_bg = "#00c853"
style_button_fg = "white"
style_button_hover_bg = "#76ff03"
style_text_fg = "white"
style_label_bg = "#1a1a1a"
style_entry_bg = "#2b2b2b"
style_entry_fg = "white"

# Marco para el video y panel de configuración
video_frame = tk.Frame(root, bg=style_frame_bg, highlightbackground=style_frame_highlight, highlightthickness=1)
video_frame.pack(side=tk.LEFT, padx=10, pady=10)

video_label = tk.Label(video_frame)
video_label.pack()

video_label.bind("<Button-1>", on_mouse_down)
video_label.bind("<B1-Motion>", on_mouse_move)
video_label.bind("<ButtonRelease-1>", on_mouse_up)

# Vincular las teclas A y D
root.bind("<Key>", on_key_press)

camera_var = StringVar(root)
camera_var.set("0")
camera_selector = OptionMenu(root, camera_var, "0", "1", command=change_camera)
camera_selector.config(font=("Arial", 12), bg=style_button_bg, fg=style_button_fg, activebackground=style_button_hover_bg)
camera_selector.pack(side=tk.TOP, padx=10, pady=10)

frame_tk = tk.Frame(root, bg=style_frame_bg, highlightbackground=style_frame_highlight, highlightthickness=1, padx=10, pady=10)
frame_tk.pack(side=tk.RIGHT, fill=tk.Y)

label_class = tk.Label(frame_tk, text="Nombre de la clase:", font=("Arial", 12, "bold"), fg=style_text_fg, bg=style_label_bg)
label_class.pack(pady=(0, 5))
entry_class = tk.Entry(frame_tk, font=("Arial", 12), bg=style_entry_bg, fg=style_entry_fg)
entry_class.pack(pady=(0, 10), fill=tk.X)
button_set_class = tk.Button(frame_tk, text="Set Clase", command=set_class, font=("Arial", 12), bg=style_button_bg, fg=style_button_fg)
button_set_class.pack(pady=(0, 15), fill=tk.X)

label_status = tk.Label(frame_tk, text="Clase actual: ", font=("Arial", 10), fg=style_text_fg, bg=style_label_bg)
label_status.pack(pady=(0, 15))

label_list = tk.Label(frame_tk, text="Clases en la imagen:", font=("Arial", 12, "bold"), fg=style_text_fg, bg=style_label_bg)
label_list.pack(pady=(0, 5))
listbox_classes = tk.Listbox(frame_tk, font=("Arial", 10), height=8, bg=style_entry_bg, fg=style_text_fg, selectbackground=style_button_bg)
listbox_classes.pack(pady=(0, 10), fill=tk.BOTH)

button_delete_class = tk.Button(frame_tk, text="Eliminar Clase", command=delete_selected_class, font=("Arial", 12), bg="#ff5252", fg="white")
button_delete_class.pack(pady=5, fill=tk.X)
button_edit_class = tk.Button(frame_tk, text="Renombrar Clase", command=edit_selected_class, font=("Arial", 12), bg="#ffab40", fg="white")
button_edit_class.pack(pady=5, fill=tk.X)
button_record = tk.Button(frame_tk, text="Iniciar Grabación", command=toggle_recording, bg=style_button_bg, fg=style_button_fg, font=("Arial", 12, "bold"), width=15)
button_record.pack(pady=10, fill=tk.X)
button_clear_directories = tk.Button(frame_tk, text="Limpiar Carpetas", command=clear_directories, font=("Arial", 12), bg="#ff5252", fg="white")
button_clear_directories.pack(pady=10, fill=tk.X)
button_review_mode = tk.Button(frame_tk, text="Revisar Etiquetas", command=toggle_review_mode, font=("Arial", 12), bg=style_button_bg, fg=style_button_fg)
button_review_mode.pack(pady=10, fill=tk.X)
button_close = tk.Button(frame_tk, text="Cerrar Programa", command=close_program, font=("Arial", 12), bg="#ff5252", fg="white")
button_close.pack(pady=10, fill=tk.X)

# Control deslizante para ajustar el intervalo de muestreo
def update_sample_interval(value):
    global SAMPLE_INTERVAL_MS
    SAMPLE_INTERVAL_MS = int(1000 / float(value))  # Convertir frecuencia a intervalo

label_slider = tk.Label(frame_tk, text="Velocidad de muestreo (muestras/segundo):", font=("Arial", 12), fg=style_text_fg, bg=style_label_bg)
label_slider.pack(pady=(20, 5))
sample_rate_slider = tk.Scale(frame_tk, from_=1, to=10, orient=tk.HORIZONTAL, command=update_sample_interval, bg=style_frame_bg, fg=style_text_fg, highlightbackground=style_frame_bg)
sample_rate_slider.set(2)  # Valor inicial de 2 muestras por segundo
sample_rate_slider.pack(pady=(0, 10), fill=tk.X)

# Iniciar la actualización periódica de la gráfica
plot_labels()

update_video()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
