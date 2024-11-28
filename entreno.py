import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Configuración de generador de datos
data_gen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
train_data = data_gen.flow_from_directory('dataset_gestos', target_size=(128, 128), subset='training', class_mode='categorical')
val_data = data_gen.flow_from_directory('dataset_gestos', target_size=(128, 128), subset='validation', class_mode='categorical')

# Obtener las clases automáticamente
GESTOS = train_data.class_indices.keys()

# Crear un modelo sencillo
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(GESTOS), activation='softmax')  # Usar len(GESTOS) aquí
])

# Compilar y entrenar
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=10)

# Guardar el modelo
model.save('modelo_gestos.h5')