import numpy as np

def ejercicio_1():
    print("=== Ejercicio 1: Cadena con números y texto ===")
    # Crear cadena con 2 números enteros, uno decimal y texto
    cadena = "En el año 2024 compré 5 casas por 1.5 millones"
    print(f"Cadena creada: {cadena}")
    
    # Extraer números usando string splitting y conversión
    palabras = cadena.split()
    anio = np.int32(palabras.index("2024"))  # Corregido para extraer el número 2024
    cantidad = np.int32(palabras.index("5"))  # Corregido para extraer el número 5
    precio = np.float32(palabras.index("1.5"))  # Corregido para extraer el número 1.5
    
    # Mostrar los números extraídos
    print("\nNúmeros extraídos como variables numpy:")
    print(f"- Año (int32): {anio}")
    print(f"- Cantidad (int32): {cantidad}")
    print(f"- Precio (float32): {precio}")
    
    # Crear un array numpy con los números
    numeros = np.array([anio, cantidad, precio])
    print(f"\nArray numpy con todos los números: {numeros}")

def ejercicio_2():
    print("\n=== Ejercicio 2: Lista con números y texto ===")
    # Crear array numpy para los números y lista para textos
    numeros = np.array([42, 17, 89, 3.14])
    textos = ["python", "programación"]
    
    print(f"Array numpy de números: {numeros}")
    print(f"Lista de textos: {textos}")
    
    # Realizar 3 extracciones diferentes
    print("\nExtracciones:")
    print(f"1. Primeros tres números: {numeros[:3]}")
    print(f"2. Último texto: {textos[-1]}")
    print(f"3. Último número: {numeros[-1]}")

def main():
    ejercicio_1()
    ejercicio_2()

if __name__ == "__main__":
    main()
