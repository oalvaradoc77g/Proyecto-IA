def print_separator(title):
    """Función auxiliar para imprimir separadores y títulos"""
    print("\n" + "="*50)
    print(f"{title:^50}")
    print("="*50)

# 1. Creación y manipulación básica de diccionarios
print_separator("DICCIONARIOS EN PYTHON")

# Diccionario de estudiante
estudiante = {
    "nombre": "Juan",
    "edad": 20,
    "cursos": ["Python", "JavaScript", "SQL"],
    "activo": True
}

print("Diccionario original:", estudiante)
print("Accediendo al nombre:", estudiante["nombre"])
print("Lista de cursos:", estudiante["cursos"])

# 2. Métodos de diccionarios
print_separator("MÉTODOS DE DICCIONARIOS")

# Agregando nuevos elementos
estudiante["promedio"] = 9.5
print("Después de agregar promedio:", estudiante)

# Obteniendo todas las claves
print("Claves del diccionario:", estudiante.keys())

# Obteniendo todos los valores
print("Valores del diccionario:", estudiante.values())

# 3. Diccionario anidado
print_separator("DICCIONARIO ANIDADO")

universidad = {
    "facultad_ingenieria": {
        "nombre": "Facultad de Ingeniería",
        "carreras": ["Sistemas", "Civil", "Mecánica"],
        "estudiantes": 1000
    },
    "facultad_medicina": {
        "nombre": "Facultad de Medicina",
        "carreras": ["Medicina General", "Enfermería"],
        "estudiantes": 800
    }
}

print("Facultades:", universidad.keys())
print("Carreras en Ingeniería:", universidad["facultad_ingenieria"]["carreras"])

# 4. Ejemplo práctico
print_separator("EJEMPLO PRÁCTICO - INVENTARIO")

inventario = {
    "manzanas": {"cantidad": 50, "precio": 1.0},
    "peras": {"cantidad": 30, "precio": 1.5},
    "naranjas": {"cantidad": 40, "precio": 0.75}
}

# Calcular valor total del inventario
total = sum(item["cantidad"] * item["precio"] for item in inventario.values())
print("Valor total del inventario: ${:.2f}".format(total))

# Mostrar inventario detallado
print("\nInventario detallado:")
for fruta, datos in inventario.items():
    print(f"{fruta.capitalize()}: {datos['cantidad']} unidades a ${datos['precio']} c/u")
    
    
print_separator("TIENDA")
tienda = {
        "productos": {
            "leche": {"precio": 2.50, "stock": 100},
            "pan": {"precio": 1.00, "stock": 50},
            "huevos": {"precio": 3.00, "stock": 200},
            "arroz": {"precio": 1.50, "stock": 150}
        },
        "empleados": {
            "cajero": ["Ana", "Pedro"],
            "reponedor": ["Juan"],
            "gerente": ["Maria"]
        },
        "horario": {
            "apertura": "08:00",
            "cierre": "20:00"
        }
    }
    
print("Productos disponibles:", tienda["productos"].keys())
# Desglose de empleados por rol
for rol, empleados in tienda["empleados"].items():
    print(f"{rol.capitalize()}: {len(empleados)} empleados - {', '.join(empleados)}")
print("Total empleados:", sum(len(v) for v in tienda["empleados"].values()))
print(f"Horario: {tienda['horario']['apertura']} - {tienda['horario']['cierre']}")