# Ejemplos de uso de strings en Python

# Métodos básicos de strings
texto = "Hola Mundo"
print(texto.upper())  # Convierte a mayúsculas
print(texto.lower())  # Convierte a minúsculas
print(texto.split())  # Divide el string en una lista
print(texto.find("Mundo"))  # Encuentra la posición de una subcadena
print(texto.startswith("Hola"))  # Verifica si empieza con una subcadena
print(texto.endswith("Mundo"))  # Verifica si termina con una subcadena
print(len(texto))  # Longitud del string
print(texto.index("Mundo"))  # Índice de la subcadena (error si no se encuentra)
print(texto.isalpha())  # Verifica si todos los caracteres son alfabéticos
print(texto.isdigit())  # Verifica si todos los caracteres son dígitos

# Concatenación
nombre = "Python"
version = "3.9"
print("Programando en " + nombre + " versión " + version)
print(f"Programando en {nombre} versión {version}")  # f-strings

# Métodos útiles
frase = "   python es genial   "
print(frase.strip())  # Elimina espacios en blanco
print(frase.replace("python", "Python"))  # Reemplaza texto
print(frase.count("n"))  # Cuenta ocurrencias

# Slicing (rebanado)
palabra = "Python"
print(palabra[0:2])    # Primeros dos caracteres
print(palabra[-2:])    # Últimos dos caracteres
print(palabra[::-1])   # Invierte el string

