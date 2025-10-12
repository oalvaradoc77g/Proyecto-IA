import random

class Jugador:
    #    Clase que representa un jugador en el dilema del prisionero.
    #Cada jugador tiene un nombre, una estrategia, años de condena y un historial de decisiones.
    
    def __init__(self, nombre, estrategia):
        self.nombre = nombre
        self.estrategia = estrategia  # "cooperar","traicionar","aleatorio","ojo_por_ojo"
        self.anios = 0
        self.historial = []

    def decidir(self, historial_oponente):
        # Determina si el jugador coopera (True) o traiciona (False) según su estrategia
        #- cooperar: siempre coopera
        #- traicionar: siempre traiciona
        #- aleatorio: decide al azar
        #- ojo_por_ojo: imita la última jugada del oponente
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
        return True

class DilemaDelPrisionero:
    #    Clase principal que implementa la lógica del juego del dilema del prisionero.
    #Matriz de pagos (en años de cárcel):
    #- Si ambos cooperan: 3 años cada uno
    #- Si ambos traicionan: 1 año cada uno
    #- Si uno coopera y otro traiciona: 0 años para el cooperador, 5 años para el traidor
    
    def __init__(self, jugador1, jugador2):
        self.jugador1 = jugador1
        self.jugador2 = jugador2
        
    def jugar_ronda(self):
        #        Ejecuta una ronda del juego donde:
        #1. Ambos jugadores toman una decisión
        #2. Se registran las decisiones en el historial
        #3. Se calculan los años de condena según la matriz de pagos
        #Retorna: tupla con las decisiones de ambos jugadores
        decision1 = self.jugador1.decidir(self.jugador2.historial)
        decision2 = self.jugador2.decidir(self.jugador1.historial)
        
        self.jugador1.historial.append(decision1)
        self.jugador2.historial.append(decision2)
        
        if decision1 and decision2:       # ambos cooperan
            self.jugador1.anios += 3
            self.jugador2.anios += 3
        elif decision1 and not decision2: # j1 coopera, j2 traiciona
            self.jugador1.anios += 0
            self.jugador2.anios += 5
        elif not decision1 and decision2: # j1 traiciona, j2 coopera
            self.jugador1.anios += 5
            self.jugador2.anios += 0
        else:                              # ambos traicionan
            self.jugador1.anios += 1
            self.jugador2.anios += 1
        
        return decision1, decision2

def main():
    #  Función principal que ejecuta una simulación del dilema del prisionero:
    #- Crea dos jugadores con diferentes estrategias
    #- Ejecuta 10 rondas del juego
    #- Muestra los resultados de cada ronda
    jugador1 = Jugador("Tomas", "cooperar")
    jugador2 = Jugador("Manuel", "ojo_por_ojo")
    jugador3 = Jugador("Ricardo", "aleatorio")
    
    # Cambia la pareja aquí según quieras simular
    juego = DilemaDelPrisionero(jugador1, jugador3)  
    #juego = DilemaDelPrisionero(jugador2, jugador3)
    #juego = DilemaDelPrisionero(jugador1, jugador2)

    rondas = 10
    for ronda in range(rondas):
        d1, d2 = juego.jugar_ronda()
        print(f"\nRonda {ronda + 1}:")
        print(f"{juego.jugador1.nombre}: {'Cooperó' if d1 else 'Traicionó'}")
        print(f"{juego.jugador2.nombre}: {'Cooperó' if d2 else 'Traicionó'}")
        print(f"Años acumulados - {juego.jugador1.nombre}: {juego.jugador1.anios}, {juego.jugador2.nombre}: {juego.jugador2.anios}")

    # Salida final: años totales y quién va a la cárcel (simple: años>0)
    print("\n=== RESULTADO FINAL ===")
    for j in (juego.jugador1, juego.jugador2):
        estado = "VA A LA CÁRCEL" if j.anios > 0 else "NO VA A LA CÁRCEL"
        print(f"{j.nombre}: {j.anios} años → {estado}")

    # Quién recibió más condena
    if juego.jugador1.anios > juego.jugador2.anios:
        print(f"\n{juego.jugador1.nombre} recibe más condena que {juego.jugador2.nombre}.")
    elif juego.jugador1.anios < juego.jugador2.anios:
        print(f"\n{juego.jugador2.nombre} recibe más condena que {juego.jugador1.nombre}.")
    else:
        print("\nAmbos recibieron la misma condena.")

if __name__ == "__main__":
    main()
