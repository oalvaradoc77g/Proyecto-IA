# Practica No. 3

# 1. Tres tipos de diccionarios

# Diccionario simple
diccionario_simple = {
    "marca": "Toyota",
    "modelo": "Corolla",
    "año": 2023,
    "color": "azul",
    "precio": 25000,
}

# Diccionario anidado
diccionario_anidado = {
    "videojuego": {
        "titulo": "The Legend of Zelda",
        "genero": "Aventura",
        "plataforma": "Nintendo Switch",
        "calificacion": 9.5,
    },
    "desarrollador": {"nombre": "Nintendo", "pais": "Japon", "fundacion": 1889},
}

# Diccionario con listas
diccionario_con_listas = {
    "frutas": ["manzana", "banana", "naranja", "uva"],
    "colores": ["rojo", "verde", "amarillo", "morado"],
    "numeros": [1, 3, 5, 7, 9],
    "ciudades": ["Madrid", "Barcelona", "Valencia", "Sevilla"],
}

print("=== DICCIONARIOS ===")
print("Diccionario simple:", diccionario_simple)
print("Diccionario anidado:", diccionario_anidado)
print("Diccionario con listas:", diccionario_con_listas)
print()

# 2. Conjuntos - creación, eliminaciones, adiciones y operaciones

# Crear 4 conjuntos
conjunto_animales = {"perro", "gato", "elefante", "leon", "tigre"}
conjunto_colores = {"rojo", "azul", "verde", "amarillo", "morado"}
conjunto_numeros = {10, 20, 30, 40, 50}
conjunto_frutas = {"manzana", "banana", "naranja", "kiwi", "mango"}

print("=== CONJUNTOS INICIALES ===")
print("Animales:", conjunto_animales)
print("Colores:", conjunto_colores)
print("Numeros:", conjunto_numeros)
print("Frutas:", conjunto_frutas)
print()

# Dos eliminaciones
conjunto_animales.remove("tigre")
conjunto_colores.discard("morado")

print("=== DESPUÉS DE ELIMINACIONES ===")
print("Animales (eliminado tigre):", conjunto_animales)
print("Colores (eliminado morado):", conjunto_colores)
print()

# Dos adiciones
conjunto_numeros.add(60)
conjunto_frutas.add("piña")

print("=== DESPUÉS DE ADICIONES ===")
print("Numeros (agregado 60):", conjunto_numeros)
print("Frutas (agregado piña):", conjunto_frutas)
print()

# Crear conjuntos adicionales para operaciones
conjunto_a = {1, 2, 3, 4, 5}
conjunto_b = {4, 5, 6, 7, 8}
conjunto_c = {1, 3, 5, 7, 9}

print("=== OPERACIONES ENTRE CONJUNTOS ===")
print("Conjunto A:", conjunto_a)
print("Conjunto B:", conjunto_b)
print("Conjunto C:", conjunto_c)
print()

# Tres operaciones entre conjuntos
union_ab = conjunto_a.union(conjunto_b)
interseccion_ac = conjunto_a.intersection(conjunto_c)
diferencia_bc = conjunto_b.difference(conjunto_c)

print("Unión A ∪ B:", union_ab)
print("Intersección A ∩ C:", interseccion_ac)
print("Diferencia B - C:", diferencia_bc)
