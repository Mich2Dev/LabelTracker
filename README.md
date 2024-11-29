LabelTracker es una herramienta diseñada para agilizar y optimizar las tareas de etiquetado en formato YOLO (You Only Look Once), facilitando la preparación de datos para entrenamientos de modelos de aprendizaje profundo.

🚀 Características
Interfaz Intuitiva: Selecciona, edita y ajusta etiquetas fácilmente con solo unos clics.
Navegación Rápida: Cambia entre imágenes etiquetadas usando las teclas A (anterior) y D (siguiente).
Edición Flexible: Ajusta las cajas delimitadoras (ROIs) arrastrando las esquinas y edita las clases directamente.
Modo de Revisión: Inspecciona y corrige etiquetas previamente creadas, garantizando la calidad de los datos.
Integración YOLO: Guarda automáticamente las etiquetas en formato compatible con YOLO para entrenamiento de modelos.
Personalización de Clases: Añade, elimina y edita clases para adaptarte a diferentes proyectos.
🖥️ Requisitos del Sistema
Python: 3.8 o superior
Bibliotecas necesarias:
OpenCV
Tkinter
Pillow
Matplotlib
(Consulta requirements.txt para más detalles.)
⚙️ Instalación
Clona este repositorio:

bash
Copiar código
git clone git@github.com:Mich2Dev/LabelTracker.git
cd LabelTracker
Crea un entorno virtual y activa:

bash
Copiar código
python3 -m venv env
source env/bin/activate
Instala las dependencias:

bash
Copiar código
pip install -r requirements.txt
Ejecuta la aplicación:

bash
Copiar código
python LabelTracker.py
🛠️ Cómo Usarlo
Iniciar Etiquetado:

Abre la aplicación y selecciona tu clase de objeto.
Dibuja las cajas delimitadoras en las regiones de interés (ROIs) usando el mouse.
Haz clic en "Iniciar Grabación" para capturar imágenes etiquetadas.
Modo Revisión:

Haz clic en "Revisar Etiquetas" para inspeccionar las imágenes previamente etiquetadas.
Usa las teclas A y D para navegar entre imágenes.
Ajusta las cajas arrastrando las esquinas o moviéndolas.
Edita las clases asociadas a cada imagen directamente desde la interfaz.
Guardar Cambios:

Los cambios se guardan automáticamente al salir del modo de revisión.
📂 Estructura del Proyecto
plaintext
Copiar código
LabelTracker/
│
├── images/               # Carpeta para imágenes capturadas
├── labels/               # Carpeta para archivos de etiquetas (formato YOLO)
├── LabelTracker.py       # Archivo principal del programa
├── requirements.txt      # Dependencias del proyecto
├── LICENSE               # Licencia del proyecto
└── README.md             # Documentación del proyecto
🎯 Casos de Uso
Preparación de datasets para entrenar modelos YOLO en tareas de detección de objetos.
Corrección de etiquetas y ajustes finos en imágenes ya etiquetadas.
Proyectos de visión por computadora en áreas como vigilancia, agricultura, manufactura, entre otros.
🤝 Contribuciones
¡Las contribuciones son bienvenidas! Si tienes sugerencias, encuentra errores o deseas agregar nuevas características, no dudes en enviar un pull request o abrir un issue.

📝 Licencia
Este proyecto está licenciado bajo la GPL-3.0 License. Consulta el archivo LICENSE para más detalles.

📧 Contacto
Autor: Mich2Dev
