[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11122276&assignment_repo_type=AssignmentRepo)
# XNAP-LANGUAGE TRANSLATION USING SEQ2SEQ LEARNING
El proyecto presentado a continuación consiste en un modelo de RNN que usa la arquitectura de aprendizaje Seq2Seq para traducir una secuencia del inglés a otro idioma. En este documento se describe paso a paso el proceso de creación del modelo (partiendo de un proyecto base, https://github.com/OValery16/Language-Translation-with-deep-learning-), con los inconvenientes encontrados y las soluciones propuestas hasta llegar al resultado final. El código está preparado para generar las gráficas de pérdida del entrenamiento y validación, las gráficas de precisión (accuracy) del entrenamiento y validación, y la gráfica de épocas usando la interfaz de Weights and Biases ([Weights & Biases](https://wandb.ai/site)). Además, el código ha sido ajustado para poderse ejecutar en remoto usando el sistema operativo Linux mediante máquinas virtuales en Azure.

## Objectives
El objetivo principal de este proyecto es la creación y optimización de un modelo de traducción automática basado en técnicas de Deep Learning. Se plantea el desafío de desarrollar un modelo que sea capaz de trabajar con varios idiomas, principalmente castellano e inglés, incluyendo otros idiomas como catalán, finlandés…, es decir, aquellos que utilizan nuestro abecedario occidental; y proporcionar traducciones de una calidad razonable.

Esto implica una búsqueda constante para aumentar la accuracy del modelo, experimentando con diferentes arquitecturas de red, ajustando hiper parámetros, y aplicando técnicas para reducir el overfitting.

También, ponemos especial énfasis en la gestión eficiente de la memoria. Los desafíos técnicos asociados con la memoria durante el entrenamiento del modelo son un tema clave en este proyecto, debido a su magnitud y limitaciones, y se buscarán soluciones para resolver cualquier problema de memoria que pueda surgir, optimizando su uso para garantizar un proceso de entrenamiento fluido y sin demasiadas interrupciones.

## Code structure
Este proyecto contiene toda la información necesaria para ejecutar el código. Las carpetas con nombres que siguen la estructura xxx-eng corresponden a los datasets de cada idioma. La carpeta outputs contiene información de salida de las ejecuciones de los archivos .py y la carpeta wandb contiene la información referente a la gráfica y que produce la interfaz de Weights & Biases al ejecutar el código. Los archivos .h5 (debe haber uno para el codificador y el decodificador) contienen la información del modelo entrenado que se crean una vez se ejecuta el archivo training.py. Estos archivos son usados por el archivo predictionTranslation.py para realizar las predicciones con la información que ha obtenido el modelo del entrenamiento.

El contenido del archivo environment.yml especifica las dependencias del proyecto, es decir, los paquetes y librerías de Python necesarios para ejecutar correctamente el código. No es necesario hacer nada con este archivo, sólo debería modificarse en caso de querer modificar el entorno de desarrollo. Finalmente, los archivos .py (predictionTranslation, util y training) son los archivos Python que contienen el código del proyecto. En el archivo util.py, se encuentran todas las funciones definidas que usan los otros dos archivos, este archivo no es necesario ejecutarlo, pero se deberán modificar las rutas (variable LOG_PATH). En el archivo training.py se encuentran las llamadas a funciones necesarias para realizar el proceso de entrenamiento (que se encuentran, como ya hemos mencionado, en el archivo util.py). De la misma forma, en el archivo predictionTranslation.py se encuentran las llamadas a funciones necesarias para, en este caso, realizar las predicciones.

## How to Run the code?
- Actualice las rutas en util.py, training.py y predictionTranslation.py en función del SO que esté usando (variable LOG_PATH en util.py y filename en los otros dos).
- Ejecute el archivo training.py para entrenar el modelo. Puede utilizar el comando: 'python3 training.py'.
- Para hacer las predicciones, ejecute predictionTranslation.py una vez el modelo se ha entrenado. Puede utilizar el comando: 'python3 predictionTranslation.py'.

## Dataset
Para nuestro proyecto hemos utilizado el dataset Anki. Este se constituye en una recopilación de oraciones y frases en múltiples idiomas utilizado comúnmente para entrenar y evaluar modelos de traducción automática.

Está estructurado en forma de parejas de oraciones, por cada par una oración está en el idioma de origen y la otra es su correspondiente traducción en el idioma objetivo. Cabe destacar que estos pares de oraciones no tienen porqué ser de longitudes y complejidades iguales, es decir que una oración puede ser más corta que la otra. Esto permite explorar una gran variedad de escenarios de traducción ya que la mayoría de las veces una frase en un idioma no será de la misma dimensión que su traducción en otro.

Tiene el siguiente aspecto: 
**Inglés + tabulación + el otro idioma + tabulación + atribución**

![Captura de pantalla 2023-07-09 a las 14 51 44](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/fc9b3ca9-94f6-4f77-82b5-71367496dfae)

En nuestro caso, para mejorar el proyecto inicial y hacer pruebas hemos utilizado el dataset de inglés-español que contiene 139.705 datos.

![Captura de pantalla 2023-07-09 a las 14 40 07](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/3bece356-3fff-4f75-991f-ae4cc4d72257)


## Architecture
Para este proyecto usamos un modelo Seq2Seq (Sequence-to-Sequence), usado en el procesamiento del lenguaje natural para la traducción automática, generación de texto y otras tareas relacionadas con secuencias. Seq2Seq tiene dos componentes principales, un codificador (encoder) y decodificador (decoder).

El codificador toma una secuencia de entrada a través de capas de redes neuronales recurrentes como células LSTM (Long Short-Term Memory) o GRU (Gated Recurrent Unit), por ejemplo una frase en un idioma determinado y la transforma en un vector de estado oculto. Un vector de estado oculto, o hidden state vector en inglés es una representación abstracta y compacta que captura la información relevante y el contexto de una secuencia de entrada en un momento dado. A medida que la secuencia de entrada se pasa por las capas de la red neuronal recurrente, el vector de estado oculto se actualiza. Éste, condensa la información relevante de la secuencia de entrada en una forma que puede ser utilizada por el decodificador (decoder) para generar la secuencia de salida, en nuestro caso, en un idioma diferente.

### Training architecture - forcing teacher method
El decodificador a su vez, produce una palabra o token, teniendo en cuenta tanto la representación actual como las palabras generadas anteriormente. Un token es una unidad básica de texto que se utiliza como punto de referencia durante el análisis y la generación de texto. Puede ser una palabra, un carácter, etc… Por ejemplo, tomando la frase “Hola, ¿cómo estás?”, los tokens podrían ser, “hola”, “¿”, “cómo”… En nuestro proyecto los tokens son palabras. La arquitectura del entrenamiento sigue el método “forcing teacher”. Este consiste en proporcionar al decodificador la secuencia de salida objetivo correcta en cada paso de tiempo durante el entrenamiento, en lugar de utilizar las salidas generadas por el propio modelo como entrada para el siguiente paso. Este método tiene distintas ventajas como estabilidad en el entrenamiento; al proporcionar la secuencia objetivo correcta en cada paso estabilizamos el entrenamiento y reducimos la convergencia del modelo

![image](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/102174790/7c3cb3a6-eb89-424a-a84e-4a8005f0636c)

### LSTM
En cuanto a la de LSTM, se pasa la secuencia de entrada codificada por el encoder. Esta secuencia tiene tamaño 91, que es el número de caracteres distintos de la lengua de entrada, en nuestro caso el inglés. Entonces el encoder procesa esta secuencia y devuelve el estado interno.

Luego, en el decoder tiene como entrada la secuencia de salida, en este caso de longitud 110 que son el número de caracteres de la lengua de salida, en este caso el castellano.

En último lugar, la capa densa, una capa totalmente conectada que coge la información de la capa recurrente y la transforma para  producir una salida final de la red, en este caso pasa de 512 (latent_dim) a 110 (tamaño del lenguaje al que vamos a traducir).

![Captura de pantalla 2023-07-09 a las 14 53 16](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/d8a2cbf9-084d-4007-a9f1-bc939f075be8)

![Captura de pantalla 2023-07-09 a las 14 39 09](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/e2d9719e-95a0-4db0-a950-c9bfc09b42b9)


### GRU
Tiene el mismo proceso que el modelo LSTM. Se diferencian en la estructura de sus capas; las puertas internas por ejemplo, la cantidad de conexiones; dónde los modelos GRU tienen menos conexiones internas que las capas LSTM o la capacidad de almacenamiento a largo plazo; las capas LSTM tienen mejor capacidad para capturar y retener información a largo plazo en comparación con las GRU. 

![Captura de pantalla 2023-07-09 a las 14 54 11](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/a0750a80-7f74-45ba-a8dd-a8bab79b9cf8)

### Choosing the best cell type: GRU vs LSTM
Antes de nada debíamos escoger los tipos de capas que íbamos a usar. Para ello hicimos una ejecución con los mismos parámetros y misma duración en términos de épocas con células GRU y con LSTM para compararlas. 

![Captura de pantalla 2023-07-09 a las 14 55 13](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/74969fff-6dd1-4a8f-87b6-5a45b8014940)

En las gráficas comparativas se puede observar que de primeras, las células GRU tienen un mayor accuracy y menor loss. No obstante podemos observar que parecen converger antes mientras que las células LSTM tenían margen para seguir aprendiendo.

Teniendo esta observación en cuenta, y a sabiendas que las capas LSTM tienen un mayor  número de conexiones internas y una mayor capacidad de almacenamiento a largo plazo, las escogimos en vez de las GRU ya que pensamos que serían más óptimas para nuestro modelo a largo plazo.

### Model's complexity 
Nos encontramos con un starting point muy simple, con un modelo sin muchos hiper parámetros y con una sola capa de codificación y decodificación. Empezamos haciendo las pruebas de hiper parámetros con una única capa de encoder y decoder. Fuimos aumentando la complejidad con hiper parámetros que se explicarán más adelante y nos estancamos en un punto donde conseguimos arreglar gran parte del overfitting donde teníamos loss de entrenamiento mayor que la de validación y el mismo caso con las accuracies pero inverso (mayor accuracy de validación con diferencia) pero el modelo nos convergía muy pronto, atascándonos en una accuracy de aproximadamente el 20%.

Para solucionar esto decidimos aumentar la complejidad de nuestro modelo añadiendo capas. Hicimos pruebas con una, dos y tres capas para encoder y decoder. Pudimos observar como se ve en la gráficas que parecía que para nuestro proyecto lo mejor serían 3 capas de encoder y decoder. Cabe destacar que los hiper parámetros que probamos con una única capa (se verán a continuación) funcionaban también los mejores para múltiples capas, lo único que hubo que cambiar fue quitar el dropout ya que como teníamos regularización de tipo L2 ya teníamos un exceso de regularización.

![Captura de pantalla 2023-07-09 a las 14 57 00](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/9b369930-bccf-4ff2-b3e0-4b9e07295b10)

![Captura de pantalla 2023-07-09 a las 14 57 15](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/393c678b-1f7b-462d-9f42-084e5ca355e9)

Cómo se puede observar en la gráfica, con dos capas se iba a obtener un peor resultado, pues empezaba a converger sobre el 24% de accuracy, menos que las demás, pero más rápido que nuestro modelo simple. Por otro lado, aunque con 3 capas el modelo converge antes, con 4 capas acaba convergiendo en el mismo punto, dando los mismos resultados. Con esto en mente y viendo que con 4 capas, el ritmo de ejecución era considerablemente más lento (cuantas más capas, más lenta la ejecución), decidimos que la mejor opción era usar 3 capas LSTM para el encoder y el decoder respectivamente.

## Memory - Data loader
Partiendo del proyecto usado como punto inicial, se tuvo que resolver un problema de memoria. Con 139.705 muestras en el entrenamiento, las máquinas alcanzaban la capacidad máxima de memoria en la GPU, abortando el proceso y haciendo que no se pudiera entrenar el conjunto de datos completo de una vez. Para solucionar este problema decidimos seguir un enfoque llamado “batch-training”. Este consiste en entrenar nuestro modelo usando lotes (batches) de datos en vez de procesar el conjunto de datos entero de una tirada. En este proceso, los datos de entrenamiento se dividen en subconjuntos más pequeños de datos llamados lotes, y los parámetros del modelo se actualizan en los gradientes calculados en cada lote.
Los objetivos que teníamos con este enfoque eran:
  - **Ganar eficiencia en memoria**: en vez de procesar el dataset entero de una vez solo debíamos   procesar cada lote cada vez, lo que significaba que no ejecutábamos tantos       datos.
  - **Generalización**: un enfoque basado en lotes ayuda con la generalización del modelo, esto se   debe a que prevenimos que el modelo memorice ejemplos específicos                introduciendo variabilidad   en cada época.

Por tanto, dividimos los datos en lotes de 30.000 muestras. Teníamos opciones menores, pero queríamos mejorar la eficiencia del entrenamiento tomando ventaja de la capacidad de procesado paralelo en la GPU. Por otro lado también queríamos una mayor generalización y estimaciones de gradiente más suaves, debido a que los lotes más grandes son más representativos de todo el dataset en cuanto a variedad de muestras, comparado con lotes más pequeños o ejemplos individuales. Con esta implementación queríamos actualizaciones más estables y solucionar el problema de memoria.

Optamos también por crear un data loader para generar lotes de datos para el entrenamiento e ir agregándolos de poco a poco a memoria. 

Para poder cuantificar la magnitud de la complejidad del problema, utilizamos varios comandos en terminal. Con el ‘model summary’, pudimos ver que nuestro modelo, con nuestro latent dim base (usando las capas LSTM), tiene prácticamente 10 millones de parámetros. Esto implica que todos los cálculos que tengan que ver con parámetros y actualizaciones ocupan mucha memoria, demostrando así que se trata de un modelo bastante grande, a nuestro parecer. 

## Training process
Inicialmente, tuvimos que pensar qué estrategia en cuanto a lotes y épocas podíamos seguir para realizar nuestro training. Al principio, erróneamente, decidimos entrenar para cada lote, durante varias épocas. El modelo veía los datos muchas veces de un determinado lote en vez de ir viendo a menudo los otros datos del siguiente; entonces, cuando saltaba al otro lote, el modelo se había sobre ajustado demasiado a los datos del anterior lote. Esto lo íbamos viendo en las gráficas que iban saliendo en el WandB, con unos picos bastante pronunciados de bajada de accuracy e incremento de la loss.

Viendo que no generalizaba bien con esta estrategia de varias épocas por lotes, decidimos probar con una alternativa opuesta, la correcta. En este caso se trataría de leer una vez un lote, y entrenar el modelo con este; de esta forma, ir pasando de lote en lote, y cuando ya se hayan leído todos los lotes, eso representaría una época entera. De esta forma, se entrenaba el modelo adoptando una estrategia BFS (se exploran primero todos los lotes) y aprendía progresivamente de cada lote, sin sobre ajustarse a ninguno, ya que no tiene la ‘oportunidad’ de memorizar tanto los datos de cada lote en cambio, ve una variedad más amplia de datos en una época.

Para visualizar las diferencias que hemos comentado entre nuestro primer intento y el que lo acabó sustituyendo, adjuntamos unas gráficas en las que se ve claramente un patrón en las líneas de 5 épocas por lote, con picos de bajada en las accuracies y picos de subida en las loss, mientras que la alternativa sigue un camino más lineal, más favorable.

![Captura de pantalla 2023-07-09 a las 14 46 50](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/04fcfc23-5b32-4eeb-9f55-532951bc30bc)


## Hyperparameters & Overfitting
A partir de nuestras estimaciones iniciales de los mejores hiper parámetros basándonos en lo que le podría convenir a nuestro modelo, con la posterior comprobación de las gráficas de validación y training, de la accuracy y la loss, vamos a enseñar el proceso seguido para encontrar los mejores hiper parámetros.

Al principio, al ajustarse demasiado el modelo a los datos de nuestro conjunto de entrenamiento, barajamos diferentes opciones para de alguna manera, simplificar nuestro modelo y estos son los cambios que hicimos.

#### - Dropout:
El dropout es una técnica de regularización para reducir el overfitting. Esta consiste en que durante el entrenamiento, de manera aleatoria, se apagan un porcentaje de neuronas en una capa. Esto ayuda a prevenir la dependencia excesiva de ciertas unidades y fuerza a la red a aprender con una fracción aleatoria de unidades en cada paso.

![Captura de pantalla 2023-07-09 a las 14 14 29](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/c899360e-c9f3-46b2-a708-b4f1c460cb07)

Con una sola capa en nuestra red neuronal vimos que lo mejor era añadir dropout, debido al mal rendimiento que tenía sin aplicarlo. En este caso el mejor era de 0.2, es decir que se apagaban de manera aleatoria el 20% de las neuronas.

Pero a pesar de esto, descubrimos mirando en teoria y articulos de towards data science / medium que el dropout nos estaba causando uno de los problemas extraños que comentamos en la reunión con el profesor, donde teníamos una training loss demasiado alta en comparación de la validation. Pudimos comprender que un dropout como el que teniamos penalizaba demasiado al training, que es donde se aplica esta 'penalización', y por ello nos surgía esa gran diferencia. 

Entonces añadiendo más capas prescindimos del dropout, arreglando el exceso de regularización, por lo que ahora ya no teníamos esos resultados pésimos que teníamos con una capa sin aplicar el dropout.

![Captura de pantalla 2023-07-09 a las 14 15 22](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/274045a7-02cc-4323-b403-7a7a733e8be9)

![Captura de pantalla 2023-07-09 a las 14 15 47](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/0b5085db-56cb-42e1-b919-c409cff96ab4)

Cómo podemos observar con la ejecución de arriba de 3 capas, en nuestro modelo comparando ejecución con dropout y sin, al añadir dropout teníamos que las losses no iban a la par (teniendo una loss de training mayor) y que las accuracies tampoco (mayor validation accuracy).

Se puede observar que quitando el dropout arreglamos este problema.

#### - Latent dim:
También probamos en hacer cambios en el tamaño de la dimensión latente. Esta es un conjunto de características ocultas en un modelo de aprendizaje automático, se utiliza para capturar y representar información relevante significativa.

![Captura de pantalla 2023-07-09 a las 14 16 49](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/ee7de096-2d54-429e-adb9-5f0d13c9bcc4)

Hicimos las primeras pruebas con una sola capa teniendo así un modelo más simple. Viendo los gráficos no había una diferencia considerable entre ellas, igualmente, siendo la peor latent dim con tamaño 256, la que nos parecía mejor era de 512 pues converge más lentamente.

![Captura de pantalla 2023-07-09 a las 14 17 17](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/b5c3b404-fe04-453e-9f29-9e4e7d4eccbe)

Luego, cuando añadimos capas al modelo volvimos a probar las latent dim de 512 y 1024 para ver si se mantenían los resultados. Aquí observamos dos cosas, primero que la ejecución con latent dim 1024 era extremadamente lenta y por ello se ve que se corta la gráfica porque estábamos perdiendo mucho tiempo. Y la otra es que converge mucho antes la latent dim de 1024 que la de 512 con múltiples capas en el modelo. Es por ello que escogimos la de 512 finalmente.

#### - Mini-batch:
El tamaño del mini-batch se refiere al número de muestras de entrenamiento que se utilizan en cada iteración del proceso de entrenamiento. Es un hiperparámetro que puede ajustarse y determina cuántas muestras se procesan simultáneamente antes de actualizar los pesos del modelo. Un tamaño de mini-batch más grande puede acelerar el entrenamiento, mientras que un tamaño más pequeño puede proporcionar estimaciones de gradiente más precisas. Es por ello que quisimos probar varios tamaños para determinar cuál era el más acertado.

![Captura de pantalla 2023-07-09 a las 14 18 17](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/f7ce9be6-19c8-4eb2-844e-8f616fedae78)

Pudimos observar fácilmente que, aunque tuviéramos un entrenamiento más lento, nos convenía utilizar el tamaño de mini-batch original de 256 pues nos permitiría conseguir una menor loss y una mayor accuracy. No valía la pena probar mini batches más pequeños porque sino el entrenamiento sería demasiado lento.

#### - Data loader randomization:
Para mejorar la generalización del modelo y evitar el overfitting pensamos en aleatorizar el orden de los datos de los lotes en el data loader. La aleatorización en un data loader implica cambiar el orden de las muestras de entrenamiento una vez se ha cargado el lote de datos y antes de entregar ese lote de datos para el entrenamiento.

Esto lo hacemos ya que sino los datos se le presentaban al modelo en un orden específico, el original, haciendo que este se vea influenciado por patrones o tendencias de los datos.

![Captura de pantalla 2023-07-09 a las 14 19 31](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/cc804024-778e-4507-9268-9633a311a7e2)

Como podemos observar, con la aleatorización aplicada, podemos hacer que el modelo aprenda más y vemos que converge más lentamente tanto en la accuracy como la loss de validación, que es lo que buscábamos.

#### - Regularization:
La regularización consiste en agregar un término de penalización a la función objetivo, proporcional a una fórmula matemática, distinta para cada tipo.
Nosotros probamos la regularización L1, L2 y Elasticnet (que consiste en una combinación de las dos anteriores).

![Captura de pantalla 2023-07-09 a las 14 20 40](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/f8089ffd-5987-44d0-a331-30e12d2e2161)

Pudimos observar claramente como la mejor era la regularización L2, ayudando más a generalizar el modelo. Mientras que la peor, con gran diferencia, era la L1. En esta gráfica parece una línea recta, aunque no lo era; lo único  es que tenía unas accuracies y losses tan bajas que al compararlas con las otras no se llegan a apreciar.

#### - Batch normalization:
El batch normalization es una técnica utilizada en redes neuronales para normalizar las salidas de cada capa durante el entrenamiento. Consiste en calcular la media y la desviación estándar de un mini-batch y luego normalizar los datos utilizando esas estadísticas.

Primero hicimos pruebas con nuestro modelo simple de una sola capa. En estas podemos ver que casi no se aprecia diferencia y que no vale la pena implementarlo.

![Captura de pantalla 2023-07-09 a las 14 21 30](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/a6b52457-999c-4789-b623-90413fee6726)

Cuando hicimos más complejo nuestro modelo, añadiendo capas, hicimos más pruebas. En la imagen de abajo en lila es batch normalization en todas las capas y en gris en la primera capa solamente. Llegamos a la conclusión de que lo mejor era como se ve en el gráfico, añadir el batch normalization a todas las capas y solo en el encoder.

![Captura de pantalla 2023-07-09 a las 14 21 53](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/c6bef2df-e50a-42cf-97ab-384489d84dd4)

#### - Optimizer:
El Optimizador determina cómo se actualizan los pesos en función de los gradientes calculados en cada iteración. El objetivo es encontrar los valores óptimos de los pesos que permitan al modelo converger hacia una solución que minimice la pérdida en los datos de entrenamiento.

![Captura de pantalla 2023-07-09 a las 14 22 40](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/8ce5e9c0-3a28-4963-8eb8-4b7ffb48ea95)

En nuestro caso, el que nos ha dado mejor resultado en las ejecuciones ha sido el RMSProp, especialmente en las ejecuciones más largas de varias capas, donde conseguía reducir mejor que el Adam los picos de bajada de accuracy y subida de loss. Creemos que puede deberse a ciertas características de este optimizador como la estabilidad de los gradientes, calculando los promedios móviles de los cuadrados de los gradientes anteriores, reduciendo la variabilidad y magnitud de los gradientes en comparación a Adam y así evitar más oscilaciones bruscas.

#### - Learning rate:
El learning rate es un hiperparámetro que controla la cantidad de ajuste que se realiza en los pesos de las neuronas durante el proceso de entrenamiento. Es decir, determina qué tan rápido o lento se actualizan los pesos en función del error calculado en el backpropagation.
En nuestro proyecto hemos acabado utilizando un learning rate scheduler con la técnica de decay. Esto es, cada X épocas, reducimos el learning rate multiplicándose por un factor. Para realizar esto primero hicimos una ejecución larga para ver donde convergía nuestro modelo fijándonos en el accuracy, desde ahí, sacamos cada cuantas épocas debía reducirse el learning rate para evitar que converja el modelo. 

![Captura de pantalla 2023-07-09 a las 14 23 32](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/4a0ab70e-addb-4d99-b455-d8e0d5dd8766)

En la gráfica podemos observar como en la época 80 empezaba a convergir el modelo y de golpe incrementa la accuracy debido a que reducimos el learning rate. Pusimos que hubiera un decay de este cada 80 steps (20 épocas), consiguiendo así una mayor accuracy y menor loss que en las demás ejecuciones.

## Results

### Final execution
Finalmente, hicimos una ejecución larga, de varias horas con 135 epochs, con todos los parámetros que hemos ido comentando. Como se puede ver, hemos conseguido reducir bastante el overfitting que nos íbamos encontrando constantemente a la hora de realizar las ejecuciones. Hemos llegado a una loss bastante baja tanto en training como en validation (0.037 y 0.038), y también hemos conseguido subir la accuracy tanto en training como validation (0.37 y 0.37). A pesar de esto, pensamos que con ciertos ajustes y ejecuciones mucho más largas, podríamos llegar a mejorarlo todavía más.

![Captura de pantalla 2023-07-09 a las 14 27 49](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/c6270bcd-8b3b-4cfa-8f26-bbfc4bdce42b)

#### Prediction
Los resultados de las traducciones no son especialmente muy buenos, ya que dependiendo de la complejidad puede fallar en bastantes cosas, pero vemos que ya hace algunas traducciones con bastante sentido.

![Captura de pantalla 2023-07-09 a las 14 28 57](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/f6718e80-bda7-42bf-863b-60cc7c418eda)
![Captura de pantalla 2023-07-09 a las 14 29 49](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/ddeff323-0d9f-45e8-84d5-20e5a52e620c)

### Language comparison
Finalmente quisimos también probar nuestro modelo con distintos idiomas. Empezamos usando el dataset de catalán, pero al tener demasiadas pocas frases, decidimos cambiar al dataset en castellano. Con este conjunto de datos sí pudimos entrenar de forma exitosa nuestro modelo, ya que poseía más de 100.000 ejemplares.

Para añadir más valor al proyecto, escogimos otro idioma, esta vez uno que no se pareciese en nada a estos dos para ver qué impacto tiene la arquitectura Seq2Seq y si realmente no hace falta que se parezcan los idiomas. Escogimos, entonces, el finlandés ya que tenía suficientes datos (unos 70.000) para entrenar el modelo. Decidimos hacer ejecuciones con las mismas epochs en ambas, sin utilizar la ejecución grande de la del castellano, para así poder comparar bien el rendimiento con el mismo tamaño. 

Como se puede observar en las gráficas a continuación, ambos idiomas (español y finlandés) mostraron resultados parecidos:

![Captura de pantalla 2023-07-09 a las 14 31 06](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/a32fcb86-d445-4988-a357-2fa8f9874bf5)

Aunque uno tenga mejores resultados que el otro, vemos que las tendencias de ambos son prácticamente iguales, la convergencia es la misma. Es un buen ejemplo para observar que la arquitectura usada (seq2seq) aprende a capturar las relaciones y patrones entre las palabras y las estructuras de las oraciones en ambos idiomas, y así puede generalizar para traducir entre idiomas diferentes. Es decir, el modelo es capaz de aprender representaciones de alto nivel y realizar una traducción basada en la información contextual aprendida durante el entrenamiento. Además, cabe destacar que, en el proceso de entrenamiento, los parámetros que recibe el modelo están tokenizados, por tanto, al estar el texto codificado, el modelo no entiende de palabras, simplemente trata con números (indistintamente del idioma).

#### Prediction
También hicimos alguna traducción al finlandés. No podíamos saber si tenían cierto sentido las traducciones, porque no sabemos nada del idioma, pero fuimos traduciendo las frases a otro traductor (el de google) y a pesar de no ser excelentes, tienen cierto sentido ambas, como ocurre con las de castellano.

![Captura de pantalla 2023-07-09 a las 14 32 51](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/e179fb45-1407-41af-948b-fd6a822f08f4)
![Captura de pantalla 2023-07-09 a las 14 33 07](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133142194/2bb5d23e-4d9f-4591-b3bd-663d7c9290a5)

## Conclusions
Como conclusiones del proyecto, creemos haber realizado un bastante buen trabajo, mejorando significativamente respecto a cuando lo empezamos inicialmente, pese a no haber obtenido excelentes accuracies y predicciones. Consideramos haber cumplido los objetivos que nos propusimos al inicio del proyecto y hemos gestionado los problemas que nos han ido surgiendo a lo largo del proyecto de la mejor forma que hemos podido. Creemos que la memoria ha sido un limitante a la hora de hacer el proyecto, muchas ejecuciones largas se interrumpían o se “morían” a mitad por no tener suficiente memoria, especialmente durante el ecuador del proyecto.

En la documentación del proyecto inicial, se aclara que entrenar el modelo por completo tarda semanas y que se empiezan a obtener buenos resultados cuando se entrena durante decenas de horas; es por ello, que cómo posible mejora nos hubiese gustado poder ejecutar durante mucho más tiempo el modelo.

Finalmente, pese a las dificultades que hemos tenido, creemos que la experiencia de usar máquinas remotas de Azure para tener mayor capacidad de ejecución ha sido una experiencia agradable. Los profesores han presentado herramientas muy útiles que nos han facilitado faena, cómo la interfaz de Weights & Biases (que auto gestiona las gráficas) o el uso de Github (que ha permitido trabajar en grupo más cómodamente) y las cuales planeamos usar en un futuro para próximos proyectos.



## Contributors
Miguel J Garrido - 1605542@uab.cat
Carlos Leta - 1599255@uab.cat
Albert Pumar - 1597973@uab.cat

Xarxes Neuronals i Aprenentatge Profund
Grau de Data Engineering, 
UAB, 2023
