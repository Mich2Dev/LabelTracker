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
import random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Directorios para guardar imágenes y etiquetas
images_dir = "images"
labels_dir = "labels"
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Variables globales para el etiquetado y seguimiento
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
HIST_SIMILARITY_THRESHOLD = 0.5  # Ajustado para una comparación más realista

TRACKER_TYPE = "CSRT"

orb = cv2.ORB_create()
lk_params = dict(winSize=(15, 15), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Variable para controlar el intervalo de muestreo en milisegundos
SAMPLE_INTERVAL_MS = 500  # Por defecto, 500 ms (2 muestras por segundo)

# Variables globales para revisión y navegación
review_mode = False
image_list = []
current_image_index = 0
previously_recording = False  # Para recordar si estaba grabando antes de entrar en revisión

# Variables para edición en modo revisión
selected_bbox = None
resize_mode = None

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
    return similarity < threshold  # Ajustado para que valores más bajos signifiquen mayor similitud

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

# Definición de la función on_key_press
def on_key_press(event):
    if review_mode:
        if event.keysym.lower() == 'd':
            next_image()
        elif event.keysym.lower() == 'a':
            prev_image()

# Funciones de manejo del mouse
def on_mouse_down(event):
    global current_bbox, tracking, selected_bbox, resize_mode
    x, y = event.x, event.y
    if review_mode:
        selected_bbox = None
        resize_mode = None

        # Verificar si el clic está cerca de alguna esquina de las cajas existentes
        for i, bbox in enumerate(bboxes):
            bx, by, bw, bh = bbox
            corners = {
                "top_left": (bx, by),
                "top_right": (bx + bw, by),
                "bottom_left": (bx, by + bh),
                "bottom_right": (bx + bw, by + bh),
            }
            for corner_name, corner_coords in corners.items():
                if abs(corner_coords[0] - x) <= 10 and abs(corner_coords[1] - y) <= 10:
                    selected_bbox = i
                    resize_mode = corner_name
                    break
            if selected_bbox is not None:
                break

        # Si no se selecciona una esquina, verificar si el clic está dentro de la caja
        if selected_bbox is None:
            for i, bbox in enumerate(bboxes):
                bx, by, bw, bh = bbox
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    selected_bbox = i
                    resize_mode = "move"
                    break

            # Si no se selecciona ninguna caja existente, iniciar una nueva caja
            if selected_bbox is None:
                current_bbox = (x, y, 1, 1)
                tracking = True
    else:
        # Modo normal para dibujar una nueva caja
        current_bbox = (x, y, 1, 1)
        tracking = True

def on_mouse_move(event):
    global current_bbox, tracking, selected_bbox, resize_mode, unique_id_counter
    x, y = event.x, event.y
    if tracking and current_bbox and not review_mode:
        # Modo normal de creación de cajas
        current_bbox = (current_bbox[0], current_bbox[1],
                        event.x - current_bbox[0], event.y - current_bbox[1])
    elif tracking and current_bbox and review_mode:
        # Modo de creación de nuevas cajas en revisión
        current_bbox = (current_bbox[0], current_bbox[1],
                        event.x - current_bbox[0], event.y - current_bbox[1])
    elif review_mode and selected_bbox is not None:
        # Modo revisión: Redimensionar o mover la caja seleccionada
        bx, by, bw, bh = bboxes[selected_bbox]

        if resize_mode == "top_left":
            new_bx = x
            new_by = y
            new_bw = bw + (bx - new_bx)
            new_bh = bh + (by - new_by)
            bboxes[selected_bbox] = (int(new_bx), int(new_by), int(max(1, new_bw)), int(max(1, new_bh)))
        elif resize_mode == "top_right":
            new_bw = x - bx
            new_by = y
            new_bh = bh + (by - new_by)
            bboxes[selected_bbox] = (int(bx), int(new_by), int(max(1, new_bw)), int(max(1, new_bh)))
        elif resize_mode == "bottom_left":
            new_bx = x
            new_bw = bw + (bx - new_bx)
            new_bh = y - by
            bboxes[selected_bbox] = (int(new_bx), int(by), int(max(1, new_bw)), int(max(1, new_bh)))
        elif resize_mode == "bottom_right":
            new_bw = x - bx
            new_bh = y - by
            bboxes[selected_bbox] = (int(bx), int(by), int(max(1, new_bw)), int(max(1, new_bh)))
        elif resize_mode == "move":
            # Calcular el desplazamiento basado en el movimiento del cursor
            center_x = bx + bw / 2
            center_y = by + bh / 2
            dx = x - center_x
            dy = y - center_y
            new_bx = bx + dx
            new_by = by + dy
            bboxes[selected_bbox] = (int(new_bx), int(new_by), int(bw), int(bh))

def on_mouse_up(event):
    global current_bbox, tracking, selected_bbox, resize_mode, unique_id_counter
    x_end, y_end = event.x, event.y

    if tracking and current_bbox:
        # Finalizar el dibujo de una nueva caja
        x_start, y_start, _, _ = current_bbox
        new_w = x_end - x_start
        new_h = y_end - y_start
        if abs(new_w) > 10 and abs(new_h) > 10:
            if not current_class:
                messagebox.showerror("Error", "Por favor, establece una clase antes de marcar el objeto.")
            else:
                # Normalizar las coordenadas en caso de que el usuario dibuje hacia arriba o hacia la izquierda
                x, y, w, h = (min(x_start, x_end), min(y_start, y_end), abs(new_w), abs(new_h))

                # Crear y agregar el tracker
                tracker = create_tracker()
                tracker.init(frame, (x, y, w, h))
                trackers.append(tracker)
                bboxes.append((x, y, w, h))
                class_names.append(current_class)
                instance_ids.append(unique_id_counter)
                unique_id_counter += 1  # Incrementar el contador global

                initial_hist = calculate_histogram(frame, (x, y, w, h))
                initial_histograms.append(initial_hist)

                kalman_filters.append(create_kalman_filter())

                if current_class not in colors:
                    colors[current_class] = get_random_color()

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

    elif review_mode and selected_bbox is not None:
        # Terminar el ajuste de una caja existente
        save_current_labels()

    selected_bbox = None
    resize_mode = None

# Funciones de Grabación
def toggle_recording():
    global recording
    if not review_mode:
        recording = not recording
        if recording:
            button_record.config(bg="#ff4444", text="Grabando...", fg="white")
            threading.Thread(target=record_loop, daemon=True).start()
        else:
            button_record.config(bg="#00c853", text="Iniciar Grabación", fg="white")
    else:
        messagebox.showinfo("Información", "No se puede grabar mientras estás en modo de revisión.")

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

# Funciones de Clase
def set_class():
    global current_class
    current_class = entry_class.get().strip()
    if current_class:
        label_status.config(text=f"Clase actual: {current_class}")
    else:
        messagebox.showwarning("Advertencia", "El nombre de la clase no puede estar vacío.")

def update_class_list():
    listbox_classes.delete(0, tk.END)
    for class_name, color in colors.items():
        listbox_classes.insert(tk.END, f"{class_name} - Color: {color}")

def delete_selected_class():
    selected_index = listbox_classes.curselection()
    if selected_index:
        selected_text = listbox_classes.get(selected_index)
        class_name = selected_text.split(" - ")[0]
        if class_name in colors:
            del colors[class_name]
        # Encontrar todos los índices que coinciden con la clase a eliminar
        indices_to_remove = [i for i, name in enumerate(class_names) if name == class_name]
        # Eliminar en orden inverso para evitar reindexación
        for index in sorted(indices_to_remove, reverse=True):
            del trackers[index]
            del bboxes[index]
            del class_names[index]
            del kalman_filters[index]
            del optical_flow_points[index]
            del instance_ids[index]
            del initial_histograms[index]
        update_class_list()
    else:
        messagebox.showinfo("Información", "Por favor, selecciona una clase para eliminar.")

def edit_selected_class():
    selected_index = listbox_classes.curselection()
    if selected_index:
        selected_text = listbox_classes.get(selected_index)
        old_class_name = selected_text.split(" - ")[0]
        new_class_name = simpledialog.askstring("Editar Clase", f"Nuevo nombre para la clase '{old_class_name}':")
        if new_class_name and new_class_name.strip() and new_class_name != old_class_name:
            new_class_name = new_class_name.strip()
            colors[new_class_name] = colors.pop(old_class_name)
            for i in range(len(class_names)):
                if class_names[i] == old_class_name:
                    class_names[i] = new_class_name
            update_class_list()
            # Actualizar las etiquetas en todas las imágenes si es necesario
        elif new_class_name == old_class_name:
            messagebox.showinfo("Información", "El nuevo nombre de la clase es el mismo que el actual.")
        else:
            messagebox.showwarning("Advertencia", "El nombre de la clase no puede estar vacío.")
    else:
        messagebox.showinfo("Información", "Por favor, selecciona una clase para editar.")

# Funciones de Revisión de Etiquetas
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
                    bboxes.append((int(x), int(y), int(w), int(h)))
                    class_names.append(class_name)
        update_class_list_for_image()

def draw_labels_on_frame():
    """Dibuja las etiquetas actuales en el frame."""
    global frame, bboxes, class_names, colors
    if frame is None:
        return
    temp_frame = frame.copy()
    for bbox, class_name in zip(bboxes, class_names):
        x, y, w, h = bbox
        color = colors.get(class_name, (0, 255, 0))
        cv2.rectangle(temp_frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(temp_frame, class_name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    img = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    video_label.imgtk = img
    video_label.config(image=img)

def save_current_labels():
    """Guarda las etiquetas actuales al archivo correspondiente."""
    global image_list, current_image_index, bboxes, class_names, frame
    if 0 <= current_image_index < len(image_list):
        label_path = os.path.join(labels_dir, image_list[current_image_index].replace(".jpg", ".txt"))
        save_labels(label_path, bboxes, class_names, frame.shape[1], frame.shape[0], int(time.time() * 1000))
        update_class_list_for_image()

def update_class_list_for_image():
    """Actualiza la lista de clases asociadas a la imagen actual."""
    listbox_image_classes.delete(0, tk.END)
    for class_name in class_names:
        listbox_image_classes.insert(tk.END, class_name)

def next_image(event=None):
    """Carga la siguiente imagen en la lista."""
    global current_image_index
    if review_mode:
        # Guardar etiquetas antes de cambiar de imagen
        save_current_labels()
    if current_image_index < len(image_list) - 1:
        current_image_index += 1
        load_image_and_labels(current_image_index)
        draw_labels_on_frame()
    else:
        messagebox.showinfo("Información", "Esta es la última imagen.")

def prev_image(event=None):
    """Carga la imagen anterior en la lista."""
    global current_image_index
    if review_mode:
        # Guardar etiquetas antes de cambiar de imagen
        save_current_labels()
    if current_image_index > 0:
        current_image_index -= 1
        load_image_and_labels(current_image_index)
        draw_labels_on_frame()
    else:
        messagebox.showinfo("Información", "Esta es la primera imagen.")

def toggle_review_mode():
    """Activa o desactiva el modo de revisión."""
    global review_mode, image_list, current_image_index, recording, previously_recording
    review_mode = not review_mode
    if review_mode:
        # Si se está grabando, detener la grabación
        if recording:
            previously_recording = True
            toggle_recording()
        else:
            previously_recording = False
        # Cargar lista de imágenes etiquetadas
        image_list = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
        if image_list:
            current_image_index = 0
            load_image_and_labels(current_image_index)
            draw_labels_on_frame()
            label_status.config(text="Modo Revisión Activado")
            # Deshabilitar componentes que no son necesarios en revisión si aplica
            # Por ejemplo, deshabilitar la selección de cámara
            camera_selector.config(state='disabled')
            # Limpiar trackers y listas relacionadas
            trackers.clear()
            kalman_filters.clear()
            optical_flow_points.clear()
            instance_ids.clear()
            initial_histograms.clear()
        else:
            messagebox.showinfo("Información", "No hay imágenes etiquetadas para revisar.")
            review_mode = False
    else:
        # Salir del modo revisión
        label_status.config(text=f"Clase actual: {current_class}")
        if previously_recording:
            toggle_recording()
        # Rehabilitar componentes deshabilitados
        camera_selector.config(state='normal')
        # Re-inicializar trackers basados en las bboxes actuales
        trackers.clear()
        kalman_filters.clear()
        optical_flow_points.clear()
        instance_ids.clear()
        initial_histograms.clear()
        for bbox in bboxes:
            tracker = create_tracker()
            tracker.init(frame, bbox)
            trackers.append(tracker)
            kalman_filters.append(create_kalman_filter())
            initial_hist = calculate_histogram(frame, bbox)
            initial_histograms.append(initial_hist)
            x, y, w, h = bbox
            roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = orb.detectAndCompute(roi_gray, None)
            if keypoints is not None and descriptors is not None:
                for kp in keypoints:
                    kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
                optical_flow_points.append((keypoints, descriptors))
            else:
                optical_flow_points.append((None, None))
        update_class_list()

# Funciones para la lista de clases por imagen en modo revisión
def rename_class_for_image():
    """Renombra una clase seleccionada en la imagen actual."""
    global class_names
    selected_index = listbox_image_classes.curselection()
    if selected_index:
        old_class_name = listbox_image_classes.get(selected_index)
        new_class_name = simpledialog.askstring("Renombrar Clase", f"Nuevo nombre para la clase '{old_class_name}':")
        if new_class_name and new_class_name.strip() and new_class_name != old_class_name:
            new_class_name = new_class_name.strip()
            # Renombrar en la lista de clases
            class_names = [
                new_class_name if name == old_class_name else name
                for name in class_names
            ]
            # Actualizar la lista y guardar los cambios
            update_class_list_for_image()
            save_current_labels()
        elif new_class_name == old_class_name:
            messagebox.showinfo("Información", "El nuevo nombre de la clase es el mismo que el actual.")
        else:
            messagebox.showwarning("Advertencia", "El nombre de la clase no puede estar vacío.")
    else:
        messagebox.showinfo("Información", "Por favor, selecciona una clase para renombrar.")

def delete_class_for_image():
    """Elimina una clase seleccionada en la imagen actual."""
    global bboxes, class_names
    selected_index = listbox_image_classes.curselection()
    if selected_index:
        answer = messagebox.askyesno(
            "Eliminar Clase",
            "¿Estás seguro de que deseas eliminar esta clase y su caja asociada?"
        )
        if answer:
            # Eliminar la clase y su correspondiente bbox
            idx = selected_index[0]
            del bboxes[idx]
            del class_names[idx]
            # Actualizar la lista y guardar los cambios
            update_class_list_for_image()
            save_current_labels()
    else:
        messagebox.showinfo("Información", "Por favor, selecciona una clase para eliminar.")

# Función para limpiar directorios
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

# Función para contar etiquetas
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

# Función para actualizar y dibujar la gráfica
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

# Función para actualizar el video y manejo de trackers
def update_video():
    global frame, frame_count, prev_gray, optical_flow_points, clean_frame
    if review_mode:
        # En modo revisión, mostrar la imagen actual con las etiquetas
        if frame is not None:
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

        if current_bbox and not review_mode:
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

        # Sincronizar las listas solo si todas tienen la misma longitud
        if len(new_bboxes) == len(new_class_names) == len(new_histograms):
            bboxes[:] = new_bboxes
            class_names[:] = new_class_names
            initial_histograms[:] = new_histograms
        else:
            print("Error: Las listas de bounding boxes, nombres de clase y histogramas no están sincronizadas.")

        frame_count += 1
        prev_gray = frame_gray.copy()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        video_label.imgtk = img
        video_label.config(image=img)

    video_label.after(20, update_video)

# Función para cerrar el programa de manera segura
def close_program():
    global recording
    recording = False
    root.quit()
    cap.release()
    cv2.destroyAllWindows()

# Iniciar la aplicación
cap = cv2.VideoCapture(camera_index)
ret, frame = cap.read()
if not ret:
    print(f"No se pudo abrir la cámara {camera_index}")

root = tk.Tk()
root.title("Aplicación de Etiquetado con Gráfica en Tiempo Real")

# Crear la figura de matplotlib
fig = plt.Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

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

# Vincular las teclas A y D para navegación en modo revisión
root.bind("<Key>", on_key_press)

# Selector de cámara
camera_var = StringVar(root)
camera_var.set("0")
camera_selector = OptionMenu(root, camera_var, *map(str, range(10)), command=change_camera)
camera_selector.config(font=("Arial", 12), bg=style_button_bg, fg=style_button_fg, activebackground=style_button_hover_bg)
camera_selector.pack(side=tk.TOP, padx=10, pady=10)

# Panel de configuración
frame_tk = tk.Frame(root, bg=style_frame_bg, highlightbackground=style_frame_highlight, highlightthickness=1, padx=10, pady=10)
frame_tk.pack(side=tk.RIGHT, fill=tk.Y)

# Entrada para nombre de clase
label_class = tk.Label(frame_tk, text="Nombre de la clase:", font=("Arial", 12, "bold"), fg=style_text_fg, bg=style_label_bg)
label_class.pack(pady=(0, 5))
entry_class = tk.Entry(frame_tk, font=("Arial", 12), bg=style_entry_bg, fg=style_entry_fg)
entry_class.pack(pady=(0, 10), fill=tk.X)
button_set_class = tk.Button(frame_tk, text="Set Clase", command=set_class, font=("Arial", 12), bg=style_button_bg, fg=style_button_fg)
button_set_class.pack(pady=(0, 15), fill=tk.X)

# Indicador de clase actual
label_status = tk.Label(frame_tk, text="Clase actual: ", font=("Arial", 10), fg=style_text_fg, bg=style_label_bg)
label_status.pack(pady=(0, 15))

# Lista de clases creadas
label_list = tk.Label(frame_tk, text="Clases creadas:", font=("Arial", 12, "bold"), fg=style_text_fg, bg=style_label_bg)
label_list.pack(pady=(0, 5))
listbox_classes = tk.Listbox(frame_tk, font=("Arial", 10), height=8, bg=style_entry_bg, fg=style_text_fg, selectbackground=style_button_bg)
listbox_classes.pack(pady=(0, 10), fill=tk.BOTH)

# Botones para eliminar y editar clases globales
button_delete_class = tk.Button(frame_tk, text="Eliminar Clase", command=delete_selected_class, font=("Arial", 12), bg="#ff5252", fg="white")
button_delete_class.pack(pady=5, fill=tk.X)
button_edit_class = tk.Button(frame_tk, text="Editar Clase", command=edit_selected_class, font=("Arial", 12), bg="#ffab40", fg="white")
button_edit_class.pack(pady=5, fill=tk.X)

# Lista de clases asociadas a la imagen actual (Modo Revisión)
label_image_classes = tk.Label(frame_tk, text="Clases de la Imagen:", font=("Arial", 12, "bold"), fg=style_text_fg, bg=style_label_bg)
label_image_classes.pack(pady=(20, 5))
listbox_image_classes = tk.Listbox(frame_tk, font=("Arial", 10), height=8, bg=style_entry_bg, fg=style_text_fg, selectbackground=style_button_bg)
listbox_image_classes.pack(pady=(0, 10), fill=tk.BOTH)

# Botones para renombrar y eliminar clases en modo revisión
button_rename_class = tk.Button(frame_tk, text="Renombrar Clase", command=rename_class_for_image, font=("Arial", 12), bg="#ffab40", fg="white")
button_rename_class.pack(pady=5, fill=tk.X)
button_delete_class_image = tk.Button(frame_tk, text="Eliminar Clase", command=delete_class_for_image, font=("Arial", 12), bg="#ff5252", fg="white")
button_delete_class_image.pack(pady=5, fill=tk.X)

# Botones adicionales
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

# Iniciar la actualización del video
# Utilizar after en lugar de threading para evitar conflictos con Tkinter
root.after(20, update_video)

root.mainloop()

cap.release()
cv2.destroyAllWindows()
