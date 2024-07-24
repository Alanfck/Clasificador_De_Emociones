**Este código implementa un sistema de análisis de emociones en texto utilizando un modelo de aprendizaje automático basado en redes neuronales convolucionales (CNN).**

*Componentes principales:*

  ***Preprocesamiento de texto:***

  Elimina etiquetas HTML, caracteres especiales y convierte el texto a minúsculas.
  Tokeniza el texto en palabras individuales.
  Convierte las palabras en números enteros (vectorización).
  Rellena las secuencias de palabras con ceros para que tengan la misma longitud.
  Modelo de red neuronal:

  ***Capa de incrustación:*** Convierte las secuencias de números enteros en vectores de representación.
  ***Capa LSTM:*** Extrae características emocionales del texto.
  ***Capa de abandono:*** Regula el modelo para evitar el sobreajuste.
  ***Capa densa final***: Predice una de las 6 emociones posibles (tristeza, alegría, amor, enojo, miedo, sorpresa).
  
  **Evaluación del modelo:**
  
  Calcula la pérdida y la precisión del modelo en un conjunto de prueba.
  Genera una matriz de confusión para visualizar el rendimiento del modelo en cada emoción.
  Clasificación de emociones:
  
  Permite al usuario ingresar un texto y el modelo predice la emoción predominante.
  Muestra un mensaje personalizado al usuario según la emoción predicha.
  
*Bibliotecas utilizadas:*

  Pandas
  scikit-learn
  TensorFlow
  Keras
  Matplotlib
  Seaborn
