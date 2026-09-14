# Telco Customer Churn - Machine Learning

## 1. Descripción del proyecto

Este proyecto busca analizar y predecir el **abandono de clientes (Churn)** en una empresa de telecomunicaciones utilizando técnicas de Machine Learning.

El objetivo es identificar patrones presentes en los datos históricos de los clientes y desarrollar un modelo de clasificación capaz de distinguir entre clientes que abandonan y clientes que permanecen en la compañía.

El proyecto se desarrolla siguiendo las principales etapas de la metodología **CRISP-DM**, desde la comprensión del problema hasta la evaluación y selección del modelo.

---

## 2. Problema

Las empresas de telecomunicaciones pueden enfrentar una pérdida importante de clientes debido al abandono del servicio.

Por este motivo, se busca desarrollar un modelo que permita identificar clientes con características asociadas al abandono, de manera que las predicciones puedan utilizarse como apoyo para futuras estrategias de retención.

---

## 3. Objetivos

### Objetivo general

Desarrollar un modelo de Machine Learning que permita clasificar a los clientes según su probabilidad de abandono.

### Objetivos específicos

- Explorar y comprender las características del dataset.
- Identificar problemas de calidad en los datos.
- Realizar la limpieza y preparación de las variables.
- Analizar patrones asociados al abandono de clientes.
- Entrenar diferentes modelos de clasificación.
- Comparar el rendimiento de los modelos utilizando distintas métricas.
- Seleccionar el modelo de acuerdo con criterios de salida definidos previamente.
- Identificar posibles sesgos y aspectos éticos relacionados con el uso del modelo.

---

## 4. Dataset

Se utiliza el dataset **Telco Customer Churn**, compuesto por:

- **7.043 registros**
- **21 variables**
- Variable objetivo: `Churn`

La variable `Churn` indica si el cliente abandonó o permaneció en la compañía.

Entre las variables disponibles se encuentran:

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`

El identificador `customerID` se utiliza únicamente como identificador y se elimina antes del modelamiento.

---

## 5. Fuente de datos

El proyecto utiliza el dataset:

**Telco Customer Churn Dataset – IBM / Kaggle**

El dataset corresponde a información histórica de clientes de telecomunicaciones y contiene variables relacionadas con sus servicios, contratos, pagos, antigüedad y abandono.

---

## 6. Estructura del proyecto

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
