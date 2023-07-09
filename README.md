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

![image](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/102174790/5220fba8-c8bf-4229-a3f4-ed069ba0a2d4)

En nuestro caso, para mejorar el proyecto inicial y hacer pruebas hemos utilizado el dataset de inglés-español que contiene 139.705 datos.

![image](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/102174790/2e6e6a7b-c67b-4ab8-a598-05b355b1f943)

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

![image](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/102174790/737c2e8c-f1f3-4c2c-bdc8-e48e9397adff)


![image](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/102174790/8af6efdf-f22d-4012-a92f-9253d425d231)

### GRU
Tiene el mismo proceso que el modelo LSTM. Se diferencian en la estructura de sus capas; las puertas internas por ejemplo, la cantidad de conexiones; dónde los modelos GRU tienen menos conexiones internas que las capas LSTM o la capacidad de almacenamiento a largo plazo; las capas LSTM tienen mejor capacidad para capturar y retener información a largo plazo en comparación con las GRU. 

![image](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/102174790/971c573a-ad8a-4d80-9498-62617112cb36)

### Choosing the best cell type: GRU vs LSTM
Antes de nada debíamos escoger los tipos de capas que íbamos a usar. Para ello hicimos una ejecución con los mismos parámetros y misma duración en términos de épocas con células GRU y con LSTM para compararlas. 

![image](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/102174790/edb1b419-396f-4d69-8ed4-7c02261fbbf2)

En las gráficas comparativas se puede observar que de primeras, las células GRU tienen un mayor accuracy y menor loss. No obstante podemos observar que parecen converger antes mientras que las células LSTM tenían margen para seguir aprendiendo.

Teniendo esta observación en cuenta, y a sabiendas que las capas LSTM tienen un mayor  número de conexiones internas y una mayor capacidad de almacenamiento a largo plazo, las escogimos en vez de las GRU ya que pensamos que serían más óptimas para nuestro modelo a largo plazo.

### Model's complexity 
Nos encontramos con un starting point muy simple, con un modelo sin muchos hiper parámetros y con una sola capa de codificación y decodificación. Empezamos haciendo las pruebas de hiper parámetros con una única capa de encoder y decoder. Fuimos aumentando la complejidad con hiper parámetros que se explicarán más adelante y nos estancamos en un punto donde conseguimos arreglar gran parte del overfitting y empezamos a ver un overfitting donde se separaban las losses, teniendo por encima la loss de entrenamiento que de validación.

Para solucionar esto decidimos aumentar la complejidad de nuestro modelo añadiendo capas. Hicimos pruebas con una, dos y tres capas para encoder y decoder. Pudimos observar como se ve en la gráficas que parecía que para nuestro proyecto lo mejor serían 3 capas de encoder y decoder. Cabe destacar que los hiper parámetros que probamos con una única capa (se verán a continuación) funcionaban también los mejores para múltiples capas, lo único que hubo que cambiar fue quitar el dropout ya que como teníamos regularización de tipo L2 ya teníamos un exceso de regularización.

![image](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/102174790/44fac945-a87b-400b-a750-06020c1a0854)

![image](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/102174790/529cf55a-9646-4842-afd2-64c30e70b7a8)

Cómo se puede observar en la gráfica, con dos capas se iba a obtener un peor resultado, pues empezaba a converger sobre el 24% de accuracy, menos que las demás. Por otro lado, aunque con 3 capas el modelo converge antes, con 4 capas acaba convergiendo en el mismo punto, dando los mismos resultados. Con esto en mente y viendo que con 4 capas, el ritmo de ejecución era considerablemente más lento (cuantas más capas, más lenta la ejecución), decidimos que la mejor opción era usar 3 capas LSTM para el encoder y el decoder respectivamente.

## Memory - Data loader
Partiendo del proyecto usado como punto inicial, se tuvo que resolver un problema de memoria. Con 139.705 muestras en el entrenamiento, las máquinas alcanzaban la capacidad máxima de memoria en la GPU, abortando el proceso y haciendo que no se pudiera entrenar el conjunto de datos completo de una vez. Para solucionar este problema decidimos seguir un enfoque llamado “batch-training”. Este consiste en entrenar nuestro modelo usando lotes (batches) de datos en vez de procesar el conjunto de datos entero de una tirada. En este proceso, los datos de entrenamiento se dividen en subconjuntos más pequeños de datos llamados lotes, y los parámetros del modelo se actualizan en los gradientes calculados en cada lote.
Los objetivos que teníamos con este enfoque eran:
  - Ganar eficiencia en memoria: en vez de procesar el dataset entero de una vez solo debíamos   procesar cada lote cada vez, lo que significaba que no ejecutábamos tantos datos.
  - Generalización: un enfoque basado en lotes ayuda con la generalización del modelo, esto se   debe a que prevenimos que el modelo memorice ejemplos específicos introduciendo variabilidad   en cada época.
Por tanto, dividimos los datos en lotes de 30.000 muestras. Teníamos opciones menores, pero queríamos mejorar la eficiencia del entrenamiento tomando ventaja de la capacidad de procesado paralelo en la GPU. Por otro lado también queríamos una mayor generalización y estimaciones de gradiente más suaves, debido a que los lotes más grandes son más representativos de todo el dataset en cuanto a variedad de muestras, comparado con lotes más pequeños o ejemplos individuales. Con esta implementación queríamos actualizaciones más estables y solucionar el problema de memoria.

Optamos también por crear un data loader para generar lotes de datos para el entrenamiento e ir agregándolos de poco a poco a memoria. 

Para poder cuantificar la magnitud de la complejidad del problema, utilizamos varios comandos en terminal. Con el ‘model summary’, pudimos ver que nuestro modelo, con nuestro latent dim base (usando las capas LSTM), tiene prácticamente 10 millones de parámetros. Esto implica que todos los cálculos que tengan que ver con parámetros y actualizaciones ocupan mucha memoria, demostrando así que se trata de un modelo bastante grande, a nuestro parecer. 

## Training process
Inicialmente, tuvimos que pensar qué estrategia en cuanto a lotes y épocas podíamos seguir para realizar nuestro training. Al principio, erróneamente, decidimos entrenar para cada lote, durante varias épocas. El modelo veía los datos muchas veces de un determinado lote en vez de ir viendo a menudo los otros datos del siguiente; entonces, cuando saltaba al otro lote, el modelo se había sobre ajustado demasiado a los datos del anterior lote. Esto lo íbamos viendo en las gráficas que iban saliendo en el WandB, con unos picos bastante pronunciados de bajada de accuracy e incremento de la loss.

Viendo que no generalizaba bien con esta estrategia de varias épocas por lotes, decidimos probar con una alternativa opuesta, la correcta. En este caso se trataría de leer una vez un lote, y entrenar el modelo con este; de esta forma, ir pasando de lote en lote, y cuando ya se hayan leído todos los lotes, eso representaría una época entera. De esta forma, se entrenaba el modelo adoptando una estrategia BFS (se exploran primero todos los lotes) y aprendía progresivamente de cada lote, sin sobre ajustarse a ninguno, ya que no tiene la ‘oportunidad’ de memorizar tanto los datos de cada lote en cambio, ve una variedad más amplia de datos en una época.
Para visualizar las diferencias que hemos comentado entre nuestro primer intento y el que lo acabó sustituyendo, adjuntamos unas gráficas en las que se ve claramente un patrón en las líneas de 5 épocas por lote, con picos de bajada en las accuracies y picos de subida en las loss, mientras que la alternativa sigue un camino más lineal, más favorable.

![image](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/102174790/60251aff-13e1-4511-b84a-095d87c32e9a)

## Hyperparameters & Overfitting
**Hyperparameters**
A partir de nuestras estimaciones iniciales de los mejores hiper parámetros basándonos en lo que le podría convenir a nuestro modelo, con la posterior comprobación de las gráficas de validación y training, de la accuracy y la loss, vamos a enseñar el proceso seguido para encontrar los mejores hiper parámetros.

**Overfitting**
Al principio, al ajustarse demasiado el modelo a los datos de nuestro conjunto de entrenamiento, barajamos diferentes opciones para de alguna manera, simplificar nuestro modelo y estos son los cambios que hicimos.


## Contributors
Miguel J Garrido - 1605542@uab.cat
Carlos Leta - 1599255@uab.cat

Xarxes Neuronals i Aprenentatge Profund
Grau de Data Engineering, 
UAB, 2023
