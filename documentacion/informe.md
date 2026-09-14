# Informe Técnico – Telco Customer Churn

**Integrantes:**Kimberly Bobadilla, Luna Cortés y Millaray Paillafil.
**Asignatura:**Machine Learning  
**Caso:**Predicción de abandono de clientes (Telco Customer Churn)  
**Año:** 2026

---

## 1. Resumen ejecutivo

El presente proyecto desarrolla una solución de Machine Learning orientada a la identificación de clientes con riesgo de abandono (*Churn*) en una empresa de telecomunicaciones.

Se trabajó con el dataset **Telco Customer Churn**, incluido en el repositorio del proyecto. El conjunto contiene **7.043 registros y 21 variables**, donde la variable objetivo `Churn` indica si el cliente abandonó o no el servicio.

El proyecto contempla las etapas de comprensión del problema, exploración y calidad de los datos, preparación, modelamiento y evaluación. Se compararon tres modelos de clasificación: **Regresión Logística, Random Forest y Gradient Boosting**.

Como criterio de salida del proceso de prueba y error se definieron los siguientes valores mínimos:

- Recall >= 0,75
- F1-score >= 0,60
- ROC-AUC >= 0,80

La **Regresión Logística** fue el modelo seleccionado porque cumple simultáneamente los tres criterios definidos.

---

## 2. Descripción del problema de negocio

Una empresa de telecomunicaciones necesita identificar clientes que presentan mayor probabilidad de abandonar el servicio.

El abandono de clientes puede representar una pérdida de ingresos y la necesidad de realizar esfuerzos de captación de nuevos clientes. Por este motivo, una herramienta predictiva puede apoyar la identificación temprana de clientes que podrían abandonar.

El problema se aborda como una tarea de **clasificación binaria**, donde:

- `0 = No Churn`: el cliente permanece.
- `1 = Churn`: el cliente abandona.

La solución propuesta no busca reemplazar la decisión humana, sino entregar una herramienta de apoyo para priorizar posibles acciones de retención.

---

## 3. Objetivos del proyecto

### Objetivo general

Desarrollar y evaluar modelos de Machine Learning capaces de identificar clientes con mayor probabilidad de abandono del servicio.

### Objetivos específicos

1. Explorar y comprender las características del dataset.
2. Identificar problemas de calidad y valores faltantes.
3. Preparar las variables para el modelamiento.
4. Dividir los datos en entrenamiento y prueba manteniendo la proporción de la variable objetivo.
5. Entrenar y comparar tres modelos de clasificación.
6. Evaluar los modelos mediante Accuracy, Precision, Recall, F1-score y ROC-AUC.
7. Definir un criterio de salida que permita detener el proceso de prueba y error.
8. Revisar posibles sesgos, aspectos éticos y de privacidad.

---

## 4. Indicadores y criterio de salida

La pauta solicita la definición de indicadores asociados al problema. Sin embargo, para la evaluación del proceso de modelamiento se sigue la aclaración docente que prioriza los **criterios de salida**, entendidos como rangos mínimos que permiten decidir cuándo detener el proceso de prueba y error.

### Criterio de salida

| Métrica | Valor mínimo | Justificación |
|---|---:|---|
| Recall | >= 0,75 | Prioriza la detección de clientes que efectivamente abandonan |
| F1-score | >= 0,60 | Busca un equilibrio entre Precision y Recall |
| ROC-AUC | >= 0,80 | Evalúa la capacidad general de discriminación del modelo |

El modelo debe cumplir **simultáneamente** los tres criterios.

El Recall se prioriza porque un falso negativo corresponde a un cliente que abandona y que el modelo no logró identificar.

---

## 5. Fuente de datos y herramientas

### Fuente de datos

El dataset utilizado se encuentra en:

`data/raw/Telco_Customer_Churn_Dataset.csv`

El archivo contiene información relacionada con características demográficas, servicios contratados, tipo de contrato, método de pago, cargos y abandono del cliente.

### Herramientas utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Git/GitHub para versionamiento y organización del proyecto

La estructura del repositorio permite separar los datos originales, datos procesados y notebooks.

---

## 6. Descripción de las variables

| Variable | Tipo | Valores únicos | Uso | Descripción |
|---|---|---:|---|---|
| customerID | Categórica/ID | 7043 | Se elimina | Identificador único |
| gender | Categórica | 2 | Predictora | Género del cliente |
| SeniorCitizen | Numérica | 2 | Predictora | Indicador de adulto mayor |
| Partner | Categórica | 2 | Predictora | Tiene pareja |
| Dependents | Categórica | 2 | Predictora | Tiene personas dependientes |
| tenure | Numérica | — | Predictora | Meses de permanencia |
| PhoneService | Categórica | 2 | Predictora | Servicio telefónico |
| MultipleLines | Categórica | 3 | Predictora | Múltiples líneas |
| InternetService | Categórica | 3 | Predictora | Tipo de Internet |
| OnlineSecurity | Categórica | 3 | Predictora | Seguridad en línea |
| OnlineBackup | Categórica | 3 | Predictora | Respaldo en línea |
| DeviceProtection | Categórica | 3 | Predictora | Protección del dispositivo |
| TechSupport | Categórica | 3 | Predictora | Soporte técnico |
| StreamingTV | Categórica | 3 | Predictora | Streaming de televisión |
| StreamingMovies | Categórica | 3 | Predictora | Streaming de películas |
| Contract | Categórica | 3 | Predictora | Tipo de contrato |
| PaperlessBilling | Categórica | 2 | Predictora | Facturación electrónica |
| PaymentMethod | Categórica | 4 | Predictora | Método de pago |
| MonthlyCharges | Numérica | — | Predictora | Cargo mensual |
| TotalCharges | Numérica | — | Predictora | Cargos acumulados |
| Churn | Categórica/binaria | 2 | Objetivo | Abandono del cliente |

---

## 7. Análisis exploratorio y calidad de los datos

### 7.1 Dimensiones

El dataset contiene:

- **7.043 filas**
- **21 columnas**

### 7.2 Variable objetivo

La distribución observada fue:

| Churn | Cantidad | Porcentaje |
|---|---:|---:|
| No | 5.174 | 73,5% |
| Yes | 1.869 | 26,5% |

Existe un desbalance moderado de clases. Por este motivo, no se considera conveniente evaluar el modelo únicamente mediante Accuracy.

### 7.3 Duplicados

No se encontraron registros duplicados.

### 7.4 Valores vacíos

Se identificaron **11 valores vacíos en `TotalCharges`**.

Al revisar estos registros se observó que corresponden a clientes con `tenure = 0`. Por esta razón, después de convertir la variable a formato numérico, los valores faltantes fueron reemplazados por **0**.

Esta decisión se documenta y se realiza antes del modelamiento.

### 7.5 Eliminación de `customerID`

La variable `customerID` fue eliminada porque corresponde a un identificador y no representa una característica útil del comportamiento del cliente.

---

## 8. Principales hallazgos del EDA

### Tipo de contrato

La proporción de abandono por tipo de contrato fue aproximadamente:

| Contrato | Churn |
|---|---:|
| Month-to-month | 42,7% |
| One year | 11,3% |
| Two year | 2,8% |

Se observa una mayor proporción de abandono en los clientes con contratos mensuales.

### Servicio de Internet

| Servicio | Churn |
|---|---:|
| DSL | 19,0% |
| Fiber optic | 41,9% |
| No Internet | 7,4% |

La categoría Fiber optic presenta una proporción de abandono superior a las otras categorías.

### Método de pago

| Método de pago | Churn |
|---|---:|
| Bank transfer (automatic) | 16,7% |
| Credit card (automatic) | 15,2% |
| Electronic check | 45,3% |
| Mailed check | 19,1% |

Electronic check presenta la mayor proporción de abandono dentro de esta variable.

### Antigüedad

La antigüedad promedio fue aproximadamente:

- Clientes que permanecen: **37,57 meses**
- Clientes que abandonan: **17,98 meses**

Esto muestra una diferencia relevante entre ambos grupos.

### Cargo mensual

El cargo mensual promedio fue aproximadamente:

- Clientes que permanecen: **61,27**
- Clientes que abandonan: **74,44**

Los clientes que abandonan presentan, en promedio, cargos mensuales superiores.

### Conclusión del EDA

El análisis exploratorio permitió identificar diferencias entre clientes que abandonan y clientes que permanecen. Destacan especialmente el tipo de contrato, el servicio de Internet, el método de pago, la antigüedad y los cargos mensuales.

Estos resultados representan asociaciones observadas en los datos y no permiten afirmar que una variable sea, por sí sola, la causa del abandono.

---

## 9. Preparación y transformación de los datos

El proceso de preparación fue realizado en Python mediante las siguientes etapas:

1. Carga del dataset original.
2. Revisión de dimensiones y tipos de datos.
3. Identificación de valores nulos y vacíos.
4. Conversión de `TotalCharges` a formato numérico.
5. Imputación de los valores faltantes de `TotalCharges` con 0, debido a que corresponden a `tenure = 0`.
6. Eliminación de `customerID`.
7. Conversión de `Churn` a valores binarios.
8. Aplicación de One-Hot Encoding a las variables categóricas.
9. División de datos en entrenamiento y prueba.
10. Uso de división estratificada.
11. Escalamiento de las variables numéricas.

La división utilizada fue:

- 80% entrenamiento
- 20% prueba
- `random_state = 42`
- `stratify = y`

El escalamiento se ajustó solamente utilizando el conjunto de entrenamiento y posteriormente se aplicó al conjunto de prueba. De esta manera se evita utilizar información del conjunto de prueba durante el entrenamiento.

---

## 10. Metodología CRISP-DM

### 1. Comprensión del negocio

El problema corresponde a la identificación de clientes con riesgo de abandono.

### 2. Comprensión de los datos

Se revisaron las dimensiones, tipos de datos, distribución de Churn, valores vacíos, duplicados y principales relaciones entre variables.

### 3. Preparación de los datos

Se trataron los valores vacíos, se eliminó el identificador, se transformaron las variables categóricas y se escalaron las variables numéricas.

### 4. Modelamiento

Se utilizaron tres modelos:

- Regresión Logística
- Random Forest
- Gradient Boosting

### 5. Evaluación

Los modelos fueron evaluados mediante Accuracy, Precision, Recall, F1-score y ROC-AUC.

Además, se aplicó el criterio de salida definido previamente.

### 6. Conclusión

La Regresión Logística alcanzó los criterios mínimos definidos y fue seleccionada como modelo final bajo las condiciones establecidas.

---

## 11. Modelamiento y resultados

Los resultados obtenidos con la configuración actual del notebook final son:

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Regresión Logística | 0,738 | 0,504 | **0,783** | **0,614** | **0,842** |
| Random Forest | 0,790 | 0,634 | 0,492 | 0,554 | 0,822 |
| Gradient Boosting | **0,798** | **0,654** | 0,505 | 0,570 | **0,842** |

### Evaluación del criterio de salida

| Modelo | Recall >= 0,75 | F1 >= 0,60 | ROC-AUC >= 0,80 | Resultado |
|---|---|---|---|---|
| Regresión Logística | Sí | Sí | Sí | **Cumple** |
| Random Forest | No | No | Sí | No cumple |
| Gradient Boosting | No | No | Sí | No cumple |

### Modelo seleccionado

La **Regresión Logística** fue seleccionada porque cumple simultáneamente los tres criterios de salida establecidos.

Aunque Gradient Boosting presenta una Accuracy superior, no alcanza el Recall mínimo definido. Por esta razón no se selecciona como modelo final bajo el criterio establecido.

La selección tampoco se basa únicamente en Accuracy, ya que la variable objetivo presenta un desbalance moderado y el objetivo principal es detectar clientes que podrían abandonar.

---

## 12. Sesgos identificados

El uso de datos históricos puede generar diferentes tipos de sesgo:

### Sesgo de representación

Los datos disponibles pueden no representar completamente a todos los clientes actuales o futuros.

### Sesgo histórico

Las relaciones presentes en los datos históricos pueden reflejar comportamientos o decisiones del pasado que no necesariamente se mantienen en el tiempo.

### Sesgo por grupos

Variables como `gender` y `SeniorCitizen` pueden presentar diferencias entre grupos. Por este motivo, se deben revisar los resultados del modelo por segmento antes de utilizarlo para decisiones reales.

### Desbalance de clases

El 26,5% de los registros corresponde a Churn, mientras que el 73,5% corresponde a No Churn. Esto justifica complementar Accuracy con Recall, F1-score y ROC-AUC.

---

## 13. Ética y privacidad

El modelo debe utilizarse como una herramienta de apoyo y no como una decisión automática sobre los clientes.

Se deben considerar los siguientes aspectos:

- Proteger la información personal de los clientes.
- Evitar utilizar datos que no sean necesarios para el objetivo.
- Controlar el acceso a los datos.
- Revisar posibles diferencias de rendimiento entre grupos.
- Evitar decisiones discriminatorias basadas únicamente en la predicción.
- Informar y supervisar el uso del modelo cuando sea aplicado en un contexto real.
- Revisar periódicamente el rendimiento del modelo debido a posibles cambios en el comportamiento de los clientes.

La variable `customerID` no se utiliza en el modelamiento, ya que funciona solamente como identificador.

---

## 14. Limitaciones

El dataset utilizado corresponde a información histórica y puede no representar completamente las condiciones actuales de una empresa de telecomunicaciones.

Además, el modelo identifica asociaciones presentes en los datos, pero no demuestra relaciones causales.

El desempeño obtenido también depende de la distribución de los datos y de las variables disponibles. Por lo tanto, antes de una utilización real sería necesario validar el modelo con datos nuevos y realizar controles adicionales de desempeño y sesgo.

---

## 15. Conclusiones

El proyecto permitió desarrollar un flujo completo de Machine Learning para el problema de abandono de clientes.

Durante el análisis se identificaron diferencias relevantes entre clientes que abandonan y clientes que permanecen, especialmente relacionadas con el tipo de contrato, servicio de Internet, método de pago, antigüedad y cargos mensuales.

La preparación de los datos permitió tratar los valores vacíos de `TotalCharges`, eliminar el identificador `customerID`, transformar las variables categóricas y preparar los conjuntos de entrenamiento y prueba.

Se compararon tres modelos de clasificación. La Regresión Logística fue seleccionada porque cumplió simultáneamente los criterios de salida establecidos: Recall igual o superior a 0,75, F1-score igual o superior a 0,60 y ROC-AUC igual o superior a 0,80.

Por lo tanto, bajo las condiciones definidas para este proyecto, se alcanzó un nivel de desempeño aceptable y se puede detener el proceso de prueba y error.

El resultado debe interpretarse como una herramienta de apoyo para identificar clientes de riesgo y no como una decisión automática. Para una implementación real sería necesario continuar validando el modelo con nuevos datos y monitorear su comportamiento, sesgos y privacidad.

---

## 16. Estructura del proyecto

```text
telco-ml-e1/
│
├── data/
│   ├── raw/
│   │   └── Telco_Customer_Churn_Dataset.csv
│   │
│   └── processed/
│       ├── Telco_Customer_Churn_Clean.csv
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/
│   ├── machine_telco_limpio.ipynb
│   ├── machine_telco_procesado.ipynb
│   ├── telco_modelamiento.ipynb
│   └── telco_pfc.ipynb
│
└── README.md
```

---

## 17. Archivos principales

El notebook `telco_pfc.ipynb` integra el flujo principal del proyecto y permite reproducir la limpieza, preparación, modelamiento y evaluación.

Los notebooks restantes representan las etapas de desarrollo del proyecto.

