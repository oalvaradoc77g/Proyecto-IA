import numpy as np

def ejemplos_variables():
    # Tipos numéricos
    entero = 10  # int: números enteros
    flotante = 3.14  # float: números decimales
    complejo = 3 + 4j  # complex: números complejos
    
    print("=== Números ===")
    print(f"Entero: {entero}, tipo: {type(entero)}")
    print(f"Flotante: {flotante}, tipo: {type(flotante)}")
    print(f"Complejo: {complejo}, tipo: {type(complejo)}")

    # Texto
    texto = "Hola Mundo"  # str: cadenas de texto
    texto_multilinea = """
    Este es un texto
    que ocupa varias
    líneas
    """
    
    print("\n=== Texto ===")
    print(f"Texto simple: {texto}")
    print(f"Texto multilinea: {texto_multilinea}")

    # Booleanos
    verdadero = True  # bool: valores True/False
    falso = False
    
    print("\n=== Booleanos ===")
    print(f"Verdadero: {verdadero}")
    print(f"Falso: {falso}")

    # Listas (arrays mutables)
    lista = [1, 2, 3, "texto", True]  # Pueden contener diferentes tipos
    print("\n=== Listas ===")
    print(f"Lista completa: {lista}")
    print(f"Primer elemento: {lista[0]}")
    print(f"Longitud de la lista: {len(lista)}")
    print(f"Elemento 2:  {lista[1]} + Elemento 3: {lista[2]}")
    
    lista.append("nuevo")  # Agregar elementos
    print(f"Lista después de append: {lista}")

    # Tuplas (arrays inmutables)
    tupla = (1, 2, 3, "texto")  # No se pueden modificar después de crear
    print("\n=== Tuplas ===")
    print(f"Tupla: {tupla}")
    print(f"Segundo elemento: {tupla[1]}")

    # Diccionarios (key-value pairs)
    diccionario = {
        "nombre": "Python",
        "version": 3.9,
        "es_genial": True
    }
    print("\n=== Diccionarios ===")
    print(f"Diccionario: {diccionario}")
    print(f"Valor de 'nombre': {diccionario['nombre']}")

    # Sets (conjuntos únicos)
    conjunto = {1, 2, 3, 3, 2, 1}  # Elementos duplicados se eliminan
    print("\n=== Sets ===")
    print(f"Conjunto: {conjunto}")

    # Arrays multidimensionales (usando NumPy)
    matriz = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    print("\n=== Matrices ===")
    print(f"Matriz completa: {matriz}")
    print(f"Elemento [1][1]: {matriz[1,1]}")  # Acceder al número 5
    print(f"Fila 1 elemento 1: {matriz.item(1,1)}")  # Ahora sí podemos usar item() con NumPy
    print(f"Forma de la matriz: {matriz.shape}")
    
    
    # Otro Ejemplo
    t_mixta=({},[],(),set(),True,3.14,2+3j,"Hola",42)
    print("\n=== Tupla Mixta ===")
    print(f"Tupla mixta: {t_mixta}")
    print(f"Tipo de cada elemento en la tupla mixta:")  
    print([type(elem) for elem in t_mixta])
    print(f"Longitud de la tupla mixta: {len(t_mixta)}")
    print(f"Accediendo a elementos específicos:")
    print(f"Elemento 0 (diccionario): {t_mixta[0]}")
    print(f"Elemento 1 (lista): {t_mixta[1]}")  
    print(f"Elemento 2 (tupla): {t_mixta[2]}")
    print(f"Elemento 3 (set): {t_mixta[3]}")
    print(f"Elemento 4 (booleano): {t_mixta[4]}")
    print(f"Elemento 5 (float): {t_mixta[5]}")
    print(f"Elemento 6 (complejo): {t_mixta[6]}")
    print(f"Elemento 7 (string): {t_mixta[7]}")
    print(f"Elemento 8 (entero): {t_mixta[8]}")
    print(f"Ocurrencias: {t_mixta.count(42)}")  # Contar ocurrencias de 42
    

if __name__ == "__main__":
    ejemplos_variables()
