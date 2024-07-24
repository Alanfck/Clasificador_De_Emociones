import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Función para limpiar el texto
def limpiar_texto(texto):
    texto = re.sub(r'<.*?>', '', texto)  # Eliminar etiquetas HTML
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)  # Eliminar caracteres especiales
    texto = texto.lower()  # Convertir a minúsculas
    return texto    

# Cargar el dataset
dataset = pd.read_csv("Emotions_Dataset.csv")
dataset.dropna(subset=['text', 'label'], inplace=True)  # Eliminar filas con valores nulos

# Limpiar los textos
dataset['text'] = dataset['text'].apply(limpiar_texto)

# Codificar las etiquetas de texto en números
label_encoder = LabelEncoder()
dataset['label'] = label_encoder.fit_transform(dataset['label'])

# Asignar los datos a las variables X (textos) e y (etiquetas)
X = dataset['text']
y = dataset['label']

# Tokenización y vectorización de texto
max_words = 20000  # Número máximo de palabras a considerar
max_len = 200  # Longitud máxima de las secuencias
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=max_len)

#conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Construir y entrenar el modelo de red neuronal
model = Sequential()
model.add(Embedding(max_words, 64, input_length=max_len))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))  # Dropout para regularización
model.add(LSTM(64))  # Segunda capa LSTM
model.add(Dense(6, activation='softmax'))  #6 unidades para predecir 6 emociones con activación softmax
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 

# Entrenar el modelo y guardar la historia del entrenamiento
historia = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

# Guardar el modelo entrenado
model.save("modelo_emociones3.h5")

# Cargar el modelo entrenado
modelo = load_model("modelo_emociones3.h5")

# Predicciones para el conjunto de prueba
y_pred = modelo.predict(X_test)
y_pred_classes = y_pred.argmax(axis=-1)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Verdadero")
plt.show()

# clasificar la emoción de un texto dado y mostrar un mensaje personalizado
def clasificar_emocion_con_mensaje(texto):
    texto_limpio = limpiar_texto(texto)
    texto_seq = tokenizer.texts_to_sequences([texto_limpio])
    texto_pad = pad_sequences(texto_seq, maxlen=max_len)
    prediccion = modelo.predict(texto_pad)
    emocion_predominante = prediccion.argmax(axis=-1)
    
    # Mensajes personalizados para cada emoción
    mensajes = {
        0: "Tu mensaje parece transmitir tristeza 😔",
        1: "¡Tu mensaje irradia alegría! 😄",
        2: "Parece que tu mensaje está lleno de amor ❤️",
        3: "Tu mensaje parece reflejar enojo 😡",
        4: "Parece que tu mensaje está lleno de miedo 😨",
        5: "¡Qué sorpresa! Tu mensaje transmite asombro 😮"
    }
    
    # Mostrar el mensaje personalizado correspondiente a la emoción predominante
    return mensajes[emocion_predominante[0]]

# Bucle para permitir al usuario ingresar continuamente textos y analizar sus emociones
while True:
    texto_usuario = input("Escribe un texto para analizar su emoción (o escribe 'salir' para terminar): ")
    if texto_usuario.lower() == 'salir':
        print("¡Hasta luego!")
        break
    else:
        mensaje_emocion = clasificar_emocion_con_mensaje(texto_usuario)
        print(mensaje_emocion)

# Graficar la precisión y la pérdida
plt.figure(figsize=(12, 5))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(historia.history['accuracy'], label='Entrenamiento')
plt.plot(historia.history['val_accuracy'], label='Validación')
plt.title('Precisión del modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(historia.history['loss'], label='Entrenamiento')
plt.plot(historia.history['val_loss'], label='Validación')
plt.title('Pérdida del modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()
