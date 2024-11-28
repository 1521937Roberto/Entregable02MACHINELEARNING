import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Cargar el modelo entrenado
model = tf.keras.models.load_model('modelo_gestos.h5')

# Inicializar MediaPipe Hands y los componentes necesarios
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuración del detector de manos
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Definir las clases de los gestos
class_names = ['ok', 'no', 'L']

# Función para dibujar los landmarks y convertirlos en una imagen
def landmarks_to_image(hand_landmarks, image_size=(128, 128)):
    # Crear una imagen en blanco (tamaño de 128x128)
    image = np.zeros(image_size + (3,), dtype=np.uint8)
    
    # Dibujar puntos y conexiones entre ellos (usando MediaPipe para visualización)
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * image_size[0])
        y = int(landmark.y * image_size[1])
        # Dibujar el punto en la imagen
        cv2.circle(image, (x, y), 3, (255, 255, 255), -1)  # Blanco para los puntos
        
    # Dibujar conexiones entre puntos (para capturar la estructura de la mano)
    for connection in mp_hands.HAND_CONNECTIONS:
        start = hand_landmarks.landmark[connection[0]]
        end = hand_landmarks.landmark[connection[1]]
        start_point = (int(start.x * image_size[0]), int(start.y * image_size[1]))
        end_point = (int(end.x * image_size[0]), int(end.y * image_size[1]))
        cv2.line(image, start_point, end_point, (255, 255, 255), 2)  # Blanco para las líneas
        
    return image

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el video.")
        break

    # Convertir la imagen a RGB porque MediaPipe trabaja con este formato
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen para detectar las manos
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Convertir los landmarks a imagen
            hand_image = landmarks_to_image(hand_landmarks)
            
            # Preprocesar la imagen para que sea compatible con el modelo
            hand_image = hand_image / 255.0  # Normalizar entre 0 y 1
            hand_image = np.expand_dims(hand_image, axis=0)  # Añadir la dimensión del batch
            hand_image = np.resize(hand_image, (1, 128, 128, 3))  # Redimensionar a 128x128x3

            # Hacer la predicción
            predictions = model.predict(hand_image)
            predicted_class = np.argmax(predictions[0])

            # Mostrar la predicción en la imagen
            cv2.putText(frame, f'Predicción: {class_names[predicted_class]}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Dibujar los landmarks de la mano
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar el video con las manos detectadas y la predicción
    cv2.imshow("Detección de Manos y Predicción", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
