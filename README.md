### LabelTracker: Tu Herramienta para el Etiquetado Inteligente en YOLO

**LabelTracker** es una herramienta poderosa y sencilla que facilita el etiquetado de objetos en videos y flujos de trabajo de etiquetado en formato YOLO (You Only Look Once). Este software está diseñado para optimizar la preparación de datos destinados a modelos de aprendizaje profundo, permitiendo una experiencia más eficiente y organizada.

### 🛠️ Características Principales:

- **Interfaz Gráfica Intuitiva:** Permite seleccionar regiones de interés (ROI) directamente sobre el video usando el cursor.
- **Edición Flexible de Etiquetas:** Ajusta y modifica etiquetas en cualquier momento, ya sea durante la grabación o posteriormente en el modo de revisión.
- **Seguimiento Inteligente:** Emplea algoritmos avanzados para seguir el movimiento de los objetos etiquetados a lo largo del video, minimizando el esfuerzo manual.
- **Compatibilidad con YOLO:** Genera archivos de etiquetas directamente en formato YOLO, ideales para entrenar modelos de detección de objetos.
- **Navegación Rápida:** Utiliza las teclas **A** y **D** para navegar entre las imágenes etiquetadas y poder editarlas o revisarlas fácilmente.

### 💻 Cómo Utilizar LabelTracker:

1. **Instala las Dependencias:** Asegúrate de tener instaladas todas las dependencias necesarias. Puedes encontrar las instrucciones en el archivo `requirements.txt`.
   ```sh
   pip install -r requirements.txt
   ```
2. **Ejecuta el Programa:** Inicia LabelTracker mediante el siguiente comando:
   ```sh
   python LabelTracker.py
   ```
3. **Comienza a Etiquetar:** Utiliza la interfaz para seleccionar objetos en el video y comenzar el etiquetado.
4. **Revisión de Etiquetas:** Una vez completada la grabación, puedes revisar, modificar y ajustar las etiquetas según sea necesario.

### 💡 Beneficios de Usar LabelTracker

- **Ahorra Tiempo:** La herramienta está diseñada para hacer el proceso de etiquetado rápido y sencillo, especialmente con el seguimiento automático y la capacidad de editar etiquetas de forma intuitiva.
- **Entrenamiento de Modelos Preciso:** Al generar etiquetas consistentes y precisas, tus modelos de detección de objetos obtendrán un entrenamiento de mayor calidad, lo cual mejora el desempeño del modelo.
- **Flexible y Escalable:** Diseñado para soportar diferentes casos de uso, desde pequeños proyectos hasta grandes volúmenes de datos.

### 🛡️ Requisitos

- **Python 3.6+**
- **OpenCV** 
- **Tkinter** 
- **Matplotlib** 
### 📜 Licencia

Este proyecto está bajo la licencia **GPL-3.0**. Puedes ver más detalles en el archivo `LICENSE`.

### 🌐 Contribuciones

📈 Las contribuciones son siempre bienvenidas. Si deseas agregar nuevas funcionalidades, optimizar el código o corregir errores, no dudes en hacer un fork del repositorio y crear un Pull Request.

### 🚀 ¡Empezar!

Haz un fork del repositorio y comienza a etiquetar de manera más rápida e inteligente con **LabelTracker**. Juntos podemos crear datasets de alta calidad para un futuro de aprendizaje profundo más poderoso.

—--


