def practica_listas():
    # Lista 1: Frutas
    frutas = ["manzana", "pera", "uva", "fresa", "mango", "piña"]
    print("=== Lista de Frutas ===")
    print(f"Lista completa: {frutas}")
    
    # Extracciones de la lista de frutas
    print(f"Las primeras tres frutas: {frutas[0:3]}")  # Slice de los primeros 3 elementos
    print(f"Las últimas dos frutas: {frutas[-2:]}")    # Slice de los últimos 2 elementos
    
    # Lista 2: Números de lotería
    numeros_loteria = [23, 45, 12, 78, 34, 90, 55]
    print("\n=== Lista de Números de Lotería ===")
    print(f"Lista completa: {numeros_loteria}")
    
    # Extracciones de la lista de números
    print(f"Números pares: {[num for num in numeros_loteria if num % 2 == 0]}")  # Extracción con comprensión de lista
    print(f"Números mayores a 50: {[num for num in numeros_loteria if num > 50]}")  # Otra extracción con filtro

if __name__ == "__main__":
    practica_listas()
