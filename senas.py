import cv2
import mediapipe as mp

# Inicializar MediaPipe Hands y los componentes necesarios
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuración del detector de manos
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el video.")
        break

    # Convertir la imagen a RGB porque Mediapipe trabaja con este formato
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen para detectar las manos
    result = hands.process(frame_rgb)

    # Dibujar las manos detectadas con sus landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar el video con las manos detectadas
    cv2.imshow("Detección de Manos", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
