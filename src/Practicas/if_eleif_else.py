# ======================================================
#  Ejemplos de estructuras if-elif-else en Python
#  Autor: GitHub Copilot
# ======================================================

def print_separator():
    print("\n" + "=" * 50 + "\n")

# Ejemplo 1: Calificaciones
def evaluar_calificacion(nota):
    print(f"Evaluando la calificación: {nota}")
    if nota >= 90:
        return "🌟 Excelente - A"
    elif nota >= 80:
        return "✨ Muy Bien - B"
    elif nota >= 70:
        return "👍 Bien - C"
    elif nota >= 60:
        return "⚠️ Suficiente - D"
    else:
        return "❌ Reprobado - F"

# Ejemplo 2: Clima
def recomendar_actividad(temperatura):
    print(f"Temperatura actual: {temperatura}°C")
    if temperatura > 30:
        return "🌞 Mejor quédate en casa con el aire acondicionado"
    elif temperatura > 25:
        return "🏊 Es buen momento para ir a la piscina"
    elif temperatura > 15:
        return "🚶 El clima es perfecto para dar un paseo"
    elif temperatura > 5:
        return "🧥 Abrígate bien antes de salir"
    else:
        return "⛄ Mejor quédate en casa, está muy frío"

def main():
    # Prueba de calificaciones
    print("\n📚 SISTEMA DE CALIFICACIONES")
    calificaciones = [95, 83, 75, 62, 45]
    for calif in calificaciones:
        resultado = evaluar_calificacion(calif)
        print(f"Calificación {calif}: {resultado}")

    print_separator()

    # Prueba de clima
    print("🌡️  RECOMENDACIONES SEGÚN EL CLIMA")
    temperaturas = [35, 28, 20, 10, 0]
    for temp in temperaturas:
        recomendacion = recomendar_actividad(temp)
        print(f"Para {temp}°C: {recomendacion}")

if __name__ == "__main__":
    main()