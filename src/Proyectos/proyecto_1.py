import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def cargar_datos():
    # Cargar el archivo Excel
    try:
        df = pd.read_excel(r'C:\Users\omaroalvaradoc\Documents\Personal\Extracto_Hipotecario.xlsx')
        return df
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None

def preparar_datos(df):
    # Variables independientes (X) y dependiente (y)
    X = df[['capital', 'intereses', 'seguros']]
    y = df['total_mensual']  # Asumiendo que existe esta columna
    
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def crear_modelo(X_train, X_test, y_train, y_test):
    # Crear y entrenar el modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred = modelo.predict(X_test)
    
    # Evaluar el modelo
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print("\nResultados del Modelo:")
    print(f"R² Score: {r2:.4f}")
    print(f"Error Cuadrático Medio: {mse:.4f}")
    print("\nCoeficientes:")
    for nombre, coef in zip(['Capital', 'Intereses', 'Seguros'], modelo.coef_):
        print(f"{nombre}: {coef:.4f}")
    
    return modelo

def visualizar_resultados(df, modelo):
    # Gráfico de correlación
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[['capital', 'intereses', 'seguros', 'total_mensual']].corr(), 
                annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.show()
    
    # Gráfico de dispersión: Capital vs Total Mensual
    plt.figure(figsize=(10, 6))
    plt.scatter(df['capital'], df['total_mensual'])
    plt.xlabel('Capital')
    plt.ylabel('Total Mensual')
    plt.title('Relación entre Capital y Total Mensual')
    plt.show()

def main():
    # Cargar datos
    df = cargar_datos()
    if df is None:
        return
    
    # Preparar datos
    X_train, X_test, y_train, y_test = preparar_datos(df)
    
    # Crear y evaluar modelo
    modelo = crear_modelo(X_train, X_test, y_train, y_test)
    
    # Visualizar resultados
    visualizar_resultados(df, modelo)
    
    # Ejemplo de predicción
    print("\nEjemplo de Predicción:")
    ejemplo = [[1000000, 50000, 10000]]  # Valores de ejemplo
    prediccion = modelo.predict(ejemplo)
    print(f"Para un préstamo con:")
    print(f"Capital: $1,000,000")
    print(f"Intereses: $50,000")
    print(f"Seguros: $10,000")
    print(f"El pago total mensual predicho es: ${prediccion[0]:,.2f}")

if __name__ == "__main__":
    main()
