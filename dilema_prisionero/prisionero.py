import random

class Jugador:
    """
    Clase que representa un jugador en el dilema del prisionero.
    Cada jugador tiene un nombre, una estrategia, años de condena y un historial de decisiones.
    """
    def __init__(self, nombre, estrategia):
        self.nombre = nombre
        self.estrategia = estrategia  # Puede ser: "cooperar", "traicionar", "aleatorio", "tit_for_tat"
        self.anios = 0  # Contador de años de condena acumulados
        self.historial = []  # Lista para guardar todas las decisiones tomadas

    def decidir(self, historial_oponente):
        """
        Determina si el jugador coopera (True) o traiciona (False) según su estrategia
        - cooperar: siempre coopera
        - traicionar: siempre traiciona
        - aleatorio: decide al azar
        - ojo_por_ojo: imita la última jugada del oponente
        """
        if self.estrategia == "cooperar":
            return True
        elif self.estrategia == "traicionar":
            return False
        elif self.estrategia == "aleatorio":
            return random.choice([True, False])
        elif self.estrategia == "ojo_por_ojo":
            if not historial_oponente:
                return True
            return historial_oponente[-1]

class DilemaDelPrisionero:
    """
    Clase principal que implementa la lógica del juego del dilema del prisionero.
    Matriz de pagos (en años de cárcel):
    - Si ambos cooperan: 3 años cada uno
    - Si ambos traicionan: 1 año cada uno
    - Si uno coopera y otro traiciona: 0 años para el cooperador, 5 años para el traidor
    """
    def __init__(self, jugador1, jugador2):
        self.jugador1 = jugador1
        self.jugador2 = jugador2
        
    def jugar_ronda(self):
        """
        Ejecuta una ronda del juego donde:
        1. Ambos jugadores toman una decisión
        2. Se registran las decisiones en el historial
        3. Se calculan los años de condena según la matriz de pagos
        Retorna: tupla con las decisiones de ambos jugadores
        """
        # Obtener decisiones de los jugadores
        decision1 = self.jugador1.decidir(self.jugador2.historial)
        decision2 = self.jugador2.decidir(self.jugador1.historial)
        
        # Actualizar historiales
        self.jugador1.historial.append(decision1)
        self.jugador2.historial.append(decision2)
        
        # Asignar años según la matriz de pagos
        if decision1 and decision2:  # Ambos cooperan
            self.jugador1.anios += 3
            self.jugador2.anios += 3
        elif decision1 and not decision2:  # 1 coopera, 2 traiciona
            self.jugador1.anios += 0
            self.jugador2.anios += 5
        elif not decision1 and decision2:  # 1 traiciona, 2 coopera
            self.jugador1.anios += 5
            self.jugador2.anios += 0
        else:  # Ambos traicionan
            self.jugador1.anios += 1
            self.jugador2.anios += 1
        
        return decision1, decision2

def main():
    """
    Función principal que ejecuta una simulación del dilema del prisionero:
    - Crea dos jugadores con diferentes estrategias
    - Ejecuta 10 rondas del juego
    - Muestra los resultados de cada ronda
    """
    # Crear jugadores con diferentes estrategias
    jugador1 = Jugador("Tomas", "cooperar")  # Este jugador siempre cooperará
    jugador2 = Jugador("Manuel", "ojo_por_ojo")  # Este jugador imitará la última jugada del oponente
    jugador3 = Jugador("Ricardo", "aleatorio")  # Este jugador decidirá al azar 
    
    # Crear juego
    #juego = DilemaDelPrisionero(jugador1, jugador2)
    juego = DilemaDelPrisionero(jugador1, jugador3)
    #juego = DilemaDelPrisionero(jugador2, jugador3)

    # Jugar 10 rondas
    for ronda in range(10):
        decision1, decision2 = juego.jugar_ronda()
        print(f"\nRonda {ronda + 1}:")
        print(f"{jugador1.nombre}: {'Cooperó' if decision1 else 'Traicionó'}")
        print(f"{jugador2.nombre}: {'Cooperó' if decision2 else 'Traicionó'}")
        print(f"Años de condena - {jugador1.nombre}: {jugador1.anios}, {jugador2.nombre}: {jugador2.anios}")

if __name__ == "__main__":
    main()
