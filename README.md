[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11122276&assignment_repo_type=AssignmentRepo)
# XNAP-LANGUAGE TRANSLATION USING SEQ2SEQ LEARNING
El proyecto presentado a continuación consiste en un modelo de RNN que usa la arquitectura de aprendizaje Seq2Seq para traducir una secuencia del inglés a otro idioma. En este documento se describe paso a paso el proceso de creación del modelo (partiendo de un proyecto base, https://github.com/OValery16/Language-Translation-with-deep-learning-), con los inconvenientes encontrados y las soluciones propuestas hasta llegar al resultado final. El código está preparado para generar las gráficas de pérdida del entrenamiento y validación, las gráficas de precisión (accuracy) del entrenamiento y validación, y la gráfica de épocas usando la interfaz de Weights and Biases ([Weights & Biases](https://wandb.ai/site)). Además, el código ha sido ajustado para poderse ejecutar en remoto usando el sistema operativo Linux mediante máquinas virtuales en Azure.

## Code structure
You must create as many folders as you consider. You can use the proposed structure or replace it by the one in the base code that you use as starting point. Do not forget to add Markdown files as needed to explain well the code and how to use it.

## Example Code
The given code is a simple CNN example training on the MNIST dataset. It shows how to set up the [Weights & Biases](https://wandb.ai/site)  package to monitor how your network is learning, or not.

Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:
```
conda activate xnap-example
```

To run the example code:
```
python main.py
```

## Resultados (Gráficas y predicciones)
Una vez aplicados los cambios relativos al overfitting, decidimos también aplicar cambios en la arquitectura e hiperparámetros para mejorar el código de starting point. La primera decisión que tomamos fue probar una ejecución tanto con las capas de LSTM como con las de GRU para determinar cuáles son las que mejor se ajustaban a nuestro modelo y con las que obteníamos mejores resultados.

![image](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133141608/54df1617-5b12-4ead-821b-95adb818bf25)

Como se puede observar en estas gráficas, las funciones de GRU convergen antes. Esto se debe a que tiene una arquitectura más simple que las LSTM y menor número de parámetros, esto es bueno ya que puede resultar en una mayor eficiencia computacional y velocidad durante el entrenamiento y la inferencia. Sin embargo, como lo que buscamos es capturar y retener información a largo plazo para tener una comprensión profunda de la estructura y la semántica del lenguaje, decidimos usar capas LSTM, ya que tienen una capacidad de aprendizaje ligeramente superior que las GRUs.

Establecida la arquitectura y las celdas que componen nuestro modelo, nos pusimos a modificar hiperparámetros en función de los resultados que teníamos hasta ese momento (de la misma forma que hicimos cuando gestionamos overfitting, como se ha explicado anteriormente). El primer parámetro que modificamos fue el learning rate, al cual optamos por añadirle un schedule y hacer que variara a lo largo de las épocas. Esto puede permitir solucionar problemas de estancamiento en mínimos locales y sobreajuste. Aunque en nuestro caso, no nos encontrábamos con ningún tipo de estancamiento, el modelo sí tenía sobreajuste a los datos de entrenamiento, cómo se puede observar en las gráficas a continuación:

![image](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133141608/ef6481c6-727f-4356-8bb0-3dd42181fabb)

La función de pérdida en la validación se puede ver cómo augmenta pasadas las 30 épocas y como la función de precisión de validación converge antes que la función de precisión del conjunto de entrenamiento cuando el learning rate es fijo. Cuando le añadimos un schedule, solucionamos estos problemas, como se observa en la línea gris.

Finalmente quisimos también probar nuestro modelo con distintos idiomas. Empezamos usando el dataset de catalán, pero al tener demasiadas pocas frases, decidimos cambiar al dataset en castellano. Con este conjunto de datos si pudimos entrenar de forma exitosa nuestro modelo, ya que poseía más de 100000 ejemplares (de hecho, como ya hemos mencionado al inicio, incluso tuvimos que entrenar el modelo por bloques ya que los datos no cabían en memoria).

Para añadir más valor al proyecto, escogimos otro idioma, esta vez uno que no se pareciese en nada a estos dos para ver qué impacto tiene la arquitectura Seq2Seq y si realmente no hace falta que se parezcan los idiomas. Escogimos, entonces, el finlandés ya que tenía suficientes datos para entrenar el modelo. Como se puede observar en las gráficas a continuación, ambos idiomas (español y finlandés) mostraron resultados parecidos:

![image](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133141608/999d3d84-9676-4ce5-9801-2ae5e7733c9f)

Aunque uno tenga mayor precisión y otro tenga una función de pérdida menor, vemos que las tendencias de ambos son prácticamente iguales, la convergencia es la misma. Es un buen ejemplo para observar que la arquitectura usada (seq2seq) aprende a capturar las relaciones y patrones entre las palabras y las estructuras de las oraciones en ambos idiomas, y así puede generalizar para traducir entre idiomas diferentes. Es decir, el modelo es capaz de aprender representaciones de alto nivel y realizar una traducción basada en la información contextual aprendida durante el entrenamiento. Además, cabe destacar que, en el proceso de entrenamiento, los parámetros que recibe el modelo están tokenizados, por tanto, al estar el texto codificado, el modelo no entiende de palabras, simplemente trata con números (indistintamente del idioma).

Antes de pasar a mostrar los resultados finales, queremos comentar que la única métrica usada para determinar qué tan buenos son los resultados y las predicciones ha sido la accuracy por el motivo que se ha comentado justo arriba. Cuando se compila el modelo y se determina la métrica, esta se usará para comparar los valores obtenidos con los reales. En el caso de la accuracy, calcula la proporción de elementos en las secuencias que coinciden exactamente entre las predicciones del modelo y las salidas reales. Es decir, cuenta la cantidad de elementos que son iguales en ambas secuencias y luego divide este número por el total de elementos en las secuencias para obtener un valor de exactitud entre 0 y 1. El problema es que estos elementos están codificados (son tokens), es por ello que, si se desease usar métricas específicas de traducción como pueden ser Bleu, Meteor o Rouge, se debería hacer un proceso de destokenización ya que estas métricas trabajan con texto. Esto implicaría que para cada bloque de muestras se debería destokenizar los elementos codificados y obtener el texto correspondiente, aplicarle la métrica al resultado en texto y que el modelo aprenda con este valor de la métrica. Aunque estas métricas ofreciesen un mejor feedback al modelo, consideramos que el coste computacional extra que esto supone (por cómo está diseñado el modelo usado) no nos iba a favorecer, teniendo en cuenta los problemas de memoria que teníamos, como ya se ha comentado.

Ahora sí, queremos mostrar una comparativa gráfica entre una de nuestras últimas ejecuciones y una de las primeras que hicimos con el código tal como estaba en el starting point.

![image](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133141608/af070fa9-0a6b-4e9d-b457-e15aa574cb75)

Como se puede ver en el gráfico de más a la derecha y como ya se ha comentado anteriormente, al inicio se sobreentrenaba cada bloque provocando caídas en las funciones de precisión y aumentos en las funciones de pérdida (se sobreajustaba a los datos de ese bloque). Aplicando también todos los cambios mencionados: dropout, batch normalization, learning rate schedule y optimizer (simplemente cambiamos RMSprop por Adam) se ha obtenido la gráfica morada, que cómo se puede observar, es mucho mejor que la primera gráfica.

Los resultados de predicciones no son muy buenos, ya que, si se le pide una frase un poco elaborada, ya no acierta la predicción.

![image](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133141608/855fc3bc-d1cd-45f2-a66c-ede86b366b64)

![image](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133141608/cf2a29c6-9b1e-45e9-8f8b-7c1a36fff076)

Vemos que cuando se usa el dataset en castellano, si queremos predecir una palabra sencilla el modelo es capaz de traducirla correctamente. Si queremos añadir dificultad, vemos entonces como la predicción ya no es la traducción correcta.

![image](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133141608/902a4f79-688a-4c32-80a4-4ff9b0ae9c9a)

![image](https://github.com/DCC-UAB/xnap-project-ed_group_03/assets/133141608/b47101ed-32d5-4548-8fca-3d798dffb1cc)

Vemos que sucede lo mismo para el caso del finlandés, una palabra sencilla la predice bien, una más elaborada ya no (la traducción de la segunda al castellano es me duele la pierna, que obviamente no es la traducción de My name is). Cabe añadir, que estos resultados al menos nos dan en el idioma correcto, y que estas tienen sentido, ya que las primeras predicciones ni formaban parte de ningún idioma ni tenían significado.

## Conclusiones

Como conclusiones del proyecto, creemos haber realizado un buen trabajo, mejorando significativamente el punto inicial, pese a no haber obtenido ni mucha precisión ni grandes predicciones. Consideramos haber cumplido los objetivos que nos propusimos al inicio del proyecto y hemos gestionado los problemas que nos han ido surgiendo a lo largo del proyecto de la mejor forma que hemos podido. Creemos que la memoria ha sido un limitante a la hora de hacer el proyecto, muchas ejecuciones largas se interrumpían o se “morían” a mitad por no tener suficiente memoria. El no haber podido usar la cantidad total de datos creemos que puede haber influido también a que el modelo no haya aprendido patrones complejos que mejorarían las predicciones.

En la documentación del proyecto usado como punto inicial, se aclara que entrenar el modelo por completo tarda semanas y que se empiezan a obtener buenos resultados cuando se entrena más de 20 horas, es por ello, que cómo posible mejora nos hubiese gustado poder ejecutar durante más tiempo el modelo. También creemos que para mejorar resultados se deberían añadir más capas de LSTM (o de GRU, si se usarán en su lugar) para aumentar la complejidad del modelo y que este pueda aprender patrones más complejos de ambos lenguajes.

Finalmente, pese a las dificultades que hemos tenido, creemos que la experiencia de usar máquinas remotas de Azure para tener maypr capacidad de ejecución ha sido una experiencia agradable. Los profesores han presentado herramientas muy útiles que nos han facilitado faena, cómo la interfaz de Weights & Biases (que auto gestiona las gráficas) o el uso de Github, que ha permitido trabajar en grupo más cómodamente y las cuales planeamos usar en un futuro para próximos proyectos.

## Contributors
Albert Pumar - 1597973@uab.cat
Miguel J Garrido - 1605542@uab.cat
Carlos Leta A. - 1599255@uab.cat

Xarxes Neuronals i Aprenentatge Profund
Grau de Data Engineering, 
UAB, 2023
