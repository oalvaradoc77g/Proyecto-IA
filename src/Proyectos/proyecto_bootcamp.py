"""
Asistente Financiero con Lógica Difusa
Archivo único: proyecto_bootcamp.py
Objetivo: cargar Datos Movimientos Financieros.csv, procesar, calcular métricas y obtener
recomendación financiera mediante lógica difusa. Modo consola por defecto, GUI opcional.
"""

import os
import sys
import random
from datetime import datetime, timedelta
import signal

import numpy as np
import pandas as pd

# import opcionales
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except Exception as _import_exc:
    # Guardar los objetos como None para manejar más adelante, pero
    # imprimir la excepción real para diagnóstico (habitual conflicto con numpy)
    fuzz = None
    ctrl = None
    import traceback

    print(
        "Warning: fallo importando 'skfuzzy' (scikit-fuzzy). Detalle:", file=sys.stderr
    )
    traceback.print_exception(type(_import_exc), _import_exc, _import_exc.__traceback__)

# tkinter y matplotlib solo si hay GUI
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    tk = None
    plt = None
    FigureCanvasTkAgg = None


# -------------------------
# Util: cargar y normalizar CSV de movimientos bancarios
# -------------------------
def cargar_movimientos_csv(ruta_csv):
    """
    Lee ruta_csv (CSV del extracto), limpia columnas 'Débitos' y 'Créditos',
    parsea 'Fecha' y agrega por mes retornando DataFrame con columnas:
    ['fecha','ingreso','gasto_essencial','gasto_no_essencial','ahorro']
    """
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(ruta_csv)

    df = pd.read_csv(ruta_csv, dtype=str, encoding="utf-8", low_memory=False)
    # Normalizar nombre de fecha
    if "Fecha" in df.columns:
        fechas = df["Fecha"].astype(str).str.strip()
        # Intentar parseo flexible sin usar el argumento obsoleto
        # Primero intento el parseo general (pandas/dateutil)
        df["Fecha_parsed"] = pd.to_datetime(fechas, errors="coerce")
        # Si muchas fechas nulas, intentar reemplazar meses en español por inglés breve
        if df["Fecha_parsed"].isna().mean() > 0.2:
            mes_map = {
                "ENE": "JAN",
                "FEB": "FEB",
                "MAR": "MAR",
                "ABR": "APR",
                "MAY": "MAY",
                "JUN": "JUN",
                "JUL": "JUL",
                "AGO": "AUG",
                "SEP": "SEP",
                "OCT": "OCT",
                "NOV": "NOV",
                "DIC": "DEC",
            }
            fechas2 = fechas.replace(mes_map, regex=True)
            # Intentar un formato estricto común: 'YYYY MON DD' (ej: '2024 OCT 02')
            df["Fecha_parsed"] = pd.to_datetime(
                fechas2, errors="coerce", format="%Y %b %d"
            )
            # Si sigue fallando, volver al parser flexible sobre la versión traducida
            if df["Fecha_parsed"].isna().mean() > 0.2:
                df["Fecha_parsed"] = pd.to_datetime(fechas2, errors="coerce")
        df["Fecha"] = df["Fecha_parsed"]
        df.drop(columns=["Fecha_parsed"], inplace=True, errors="ignore")
    else:
        df["Fecha"] = pd.NaT

    # Función limpieza numérica
    def limpiar_num(col):
        if col not in df.columns:
            return pd.Series(0, index=df.index)
        s = df[col].astype(str).str.replace(r"[^0-9\.\-]", "", regex=True)
        return pd.to_numeric(s.replace("", "0"), errors="coerce").fillna(0)

    df["Débitos"] = limpiar_num("Débitos")
    df["Créditos"] = limpiar_num("Créditos")

    # Agregar por mes (usar Fecha; si nula, agrupar por índice)
    if df["Fecha"].notna().any():
        df = df.sort_values("Fecha")
        # usar 'ME' (month end) para evitar futuros deprecations
        grp = df.groupby(pd.Grouper(key="Fecha", freq="ME"))
    else:
        grp = df.groupby(df.index // 30)  # fallback

    ingresos = grp["Créditos"].sum()
    gastos = grp["Débitos"].sum()
    # Distribuir gastos en esencial y no esencial 60% esencial, 40% no esencial
    # porque esa distribución es común en finanzas personales
    gasto_essencial = gastos * 0.6
    gasto_no_essencial = gastos * 0.4
    # El ahorro es la diferencia entre ingresos y gastos
    ahorro = ingresos - gastos

    datos = pd.DataFrame(
        {
            "fecha": (
                ingresos.index.to_timestamp()
                if hasattr(ingresos.index, "to_timestamp")
                else ingresos.index
            ),
            "ingreso": ingresos.values.astype(float),
            "gasto_essencial": gasto_essencial.values.astype(float),
            "gasto_no_essencial": gasto_no_essencial.values.astype(float),
            "ahorro": ahorro.values.astype(float),
        }
    ).reset_index(drop=True)

    return datos


# -------------------------
# Clase: AsistenteFinancieroDifuso (modo consola + opcional GUI)
# -------------------------
class AsistenteFinancieroDifuso:
    def __init__(self, datos_df=None, root=None):
        if fuzz is None or ctrl is None:
            raise ImportError(
                "scikit-fuzzy no está instalado. Ejecuta: pip install scikit-fuzzy"
            )
        self.root = root
        self.datos_financieros = (
            datos_df.copy()
            # Se encarga de asegurar que los datos sean un DataFrame
            if isinstance(datos_df, pd.DataFrame)
            # Genera datos simulados si no se pasan datos
            else self._generar_simulados()
        )
        self._asegurar_tipos()
        self._crear_sistema_difuso()
        if self.root is not None:
            if tk is None:
                raise RuntimeError("Tkinter no disponible en este entorno")
            self._crear_interfaz()

    def _generar_simulados(self, dias=90):
        fechas = [datetime.now() - timedelta(days=x) for x in range(dias)]
        datos = []
        for f in reversed(fechas):
            g_e = random.uniform(800, 1200)
            g_ne = random.uniform(200, 500)
            a = random.uniform(100, 800)
            ingreso = g_e + g_ne + a
            datos.append(
                {
                    "fecha": f,
                    "ingreso": ingreso,
                    "gasto_essencial": g_e,
                    "gasto_no_essencial": g_ne,
                    "ahorro": a,
                }
            )
        return pd.DataFrame(datos)

    def _asegurar_tipos(self):
        if "fecha" in self.datos_financieros.columns:
            self.datos_financieros["fecha"] = pd.to_datetime(
                self.datos_financieros["fecha"], errors="coerce"
            )
        for c in ["ingreso", "gasto_essencial", "gasto_no_essencial", "ahorro"]:
            if c in self.datos_financieros.columns:
                self.datos_financieros[c] = pd.to_numeric(
                    self.datos_financieros[c], errors="coerce"
                ).fillna(0.0)
            else:
                self.datos_financieros[c] = 0.0

    def _crear_sistema_difuso(self):
        # entradas
        self.ratio_ahorro = ctrl.Antecedent(np.arange(0, 101, 1), "ratio_ahorro")
        self.ratio_gasto_essencial = ctrl.Antecedent(
            np.arange(0, 101, 1), "ratio_gasto_essencial"
        )
        self.estabilidad_ingresos = ctrl.Antecedent(
            np.arange(0, 101, 1), "estabilidad_ingresos"
        )
        # salida
        self.recomendacion = ctrl.Consequent(np.arange(0, 101, 1), "recomendacion")

        self.ratio_ahorro["bajo"] = fuzz.trimf(self.ratio_ahorro.universe, [0, 0, 20])
        self.ratio_ahorro["medio"] = fuzz.trimf(
            self.ratio_ahorro.universe, [10, 30, 50]
        )
        self.ratio_ahorro["alto"] = fuzz.trimf(
            self.ratio_ahorro.universe, [40, 100, 100]
        )

        self.ratio_gasto_essencial["bajo"] = fuzz.trimf(
            self.ratio_gasto_essencial.universe, [0, 0, 35]
        )
        self.ratio_gasto_essencial["medio"] = fuzz.trimf(
            self.ratio_gasto_essencial.universe, [25, 45, 65]
        )
        self.ratio_gasto_essencial["alto"] = fuzz.trimf(
            self.ratio_gasto_essencial.universe, [55, 100, 100]
        )

        self.estabilidad_ingresos["baja"] = fuzz.trimf(
            self.estabilidad_ingresos.universe, [0, 0, 50]
        )
        self.estabilidad_ingresos["media"] = fuzz.trimf(
            self.estabilidad_ingresos.universe, [30, 60, 90]
        )
        self.estabilidad_ingresos["alta"] = fuzz.trimf(
            self.estabilidad_ingresos.universe, [70, 100, 100]
        )

        self.recomendacion["emergencia"] = fuzz.trimf(
            self.recomendacion.universe, [0, 0, 30]
        )
        self.recomendacion["conservador"] = fuzz.trimf(
            self.recomendacion.universe, [20, 40, 60]
        )
        self.recomendacion["moderado"] = fuzz.trimf(
            self.recomendacion.universe, [50, 65, 80]
        )
        self.recomendacion["agresivo"] = fuzz.trimf(
            self.recomendacion.universe, [70, 100, 100]
        )

        reglas = [
            ctrl.Rule(
                self.ratio_ahorro["bajo"] & self.ratio_gasto_essencial["alto"],
                self.recomendacion["emergencia"],
            ),
            ctrl.Rule(
                self.ratio_ahorro["bajo"] & self.ratio_gasto_essencial["medio"],
                self.recomendacion["conservador"],
            ),
            ctrl.Rule(
                self.ratio_ahorro["medio"] & self.estabilidad_ingresos["alta"],
                self.recomendacion["moderado"],
            ),
            ctrl.Rule(
                self.ratio_ahorro["alto"]
                & self.estabilidad_ingresos["alta"]
                & self.ratio_gasto_essencial["bajo"],
                self.recomendacion["agresivo"],
            ),
            ctrl.Rule(
                self.ratio_ahorro["medio"] & self.estabilidad_ingresos["baja"],
                self.recomendacion["conservador"],
            ),
            ctrl.Rule(
                self.ratio_ahorro["alto"] & self.estabilidad_ingresos["media"],
                self.recomendacion["moderado"],
            ),
        ]
        self.ctrl_system = ctrl.ControlSystem(reglas)
        self.sim = ctrl.ControlSystemSimulation(self.ctrl_system)

    def calcular_metricas(self):
        datos = self.datos_financieros
        # usar últimos 90 días si hay fecha
        if "fecha" in datos.columns and datos["fecha"].notna().any():
            maxf = datos["fecha"].max()
            ventana = maxf - pd.Timedelta(days=90)
            sub = datos[datos["fecha"] >= ventana]
            if sub.empty:
                sub = datos
        else:
            sub = datos
        ingreso_prom = sub["ingreso"].mean()
        ahorro_prom = sub["ahorro"].mean()
        gasto_ess_prom = sub["gasto_essencial"].mean()
        estabilidad = 100 - (sub["ingreso"].std() / (ingreso_prom + 1e-9)) * 100
        ratio_ahorro = (ahorro_prom / (ingreso_prom + 1e-9)) * 100
        ratio_gasto_ess = (gasto_ess_prom / (ingreso_prom + 1e-9)) * 100
        return {
            "ratio_ahorro": float(np.clip(ratio_ahorro, 0, 100)),
            "ratio_gasto_essencial": float(np.clip(ratio_gasto_ess, 0, 100)),
            "estabilidad_ingresos": float(np.clip(estabilidad, 0, 100)),
            "ingreso_promedio": float(ingreso_prom),
            "ahorro_promedio": float(ahorro_prom),
        }

    def obtener_recomendacion(self):
        metricas = self.calcular_metricas()
        self.sim.input["ratio_ahorro"] = metricas["ratio_ahorro"]
        self.sim.input["ratio_gasto_essencial"] = metricas["ratio_gasto_essencial"]
        self.sim.input["estabilidad_ingresos"] = metricas["estabilidad_ingresos"]
        self.sim.compute()
        val = float(self.sim.output["recomendacion"])
        if val <= 30:
            cat = "EMERGENCIA"
            exp = "Recortar gastos no esenciales, crear fondo de emergencia, revisar deudas."
        elif val <= 60:
            cat = "CONSERVADOR"
            exp = "Mantener ahorro, instrumentos de bajo riesgo, construir fondo 3-6 meses."
        elif val <= 80:
            cat = "MODERADO"
            exp = "Incrementar inversiones, considerar riesgo medio, plan largo plazo."
        else:
            cat = "AGRESIVO"
            exp = "Diversificar agresivamente, aprovechar alta capacidad de ahorro."
        return {
            "categoria": cat,
            "valor": val,
            "explicacion": exp,
            "metricas": metricas,
        }

    # modo consola
    def correr_en_consola(self):
        r = self.obtener_recomendacion()
        m = r["metricas"]
        print("=" * 70)
        print("INFORME FINANCIERO - Asistente Difuso")
        print("=" * 70)
        print(f"Fecha análisis: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"Ratio Ahorro: {m['ratio_ahorro']:.1f}%")
        print(f"Ratio Gasto Esencial: {m['ratio_gasto_essencial']:.1f}%")
        print(f"Estabilidad Ingresos: {m['estabilidad_ingresos']:.1f}%")
        print(f"Ingreso Promedio: ${m['ingreso_promedio']:.2f}")
        print(f"Ahorro Promedio: ${m['ahorro_promedio']:.2f}")
        print(f"Recomendación: {r['categoria']} ({r['valor']:.1f})")
        print(r["explicacion"])
        print("=" * 70)

    # GUI mínima (opcional)
    def _crear_interfaz(self):
        self.root.title("Asistente Financiero - Lógica Difusa")
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)
        btn = ttk.Button(frame, text="Analizar situación", command=self._accion_gui)
        btn.pack(fill="x", pady=5)
        self.txt = scrolledtext.ScrolledText(frame, width=80, height=20)
        self.txt.pack(fill="both", expand=True, pady=5)
        if plt is not None:
            btn_g = ttk.Button(frame, text="Ver gráficas", command=self._graficas_gui)
            btn_g.pack(fill="x", pady=5)

    def _accion_gui(self):
        r = self.obtener_recomendacion()
        m = r["metricas"]
        reporte = (
            f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Ratio Ahorro: {m['ratio_ahorro']:.1f}%\n"
            f"Ratio Gasto Esencial: {m['ratio_gasto_essencial']:.1f}%\n"
            f"Estabilidad: {m['estabilidad_ingresos']:.1f}%\n"
            f"Ingreso Promedio: ${m['ingreso_promedio']:.2f}\n"
            f"Ahorro Promedio: ${m['ahorro_promedio']:.2f}\n\n"
            f"Recomendación: {r['categoria']} ({r['valor']:.1f})\n"
            f"{r['explicacion']}\n"
        )
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", reporte)

    def _graficas_gui(self):
        if plt is None or FigureCanvasTkAgg is None:
            messagebox.showwarning(
                "Dependencia", "matplotlib no disponible para gráficas"
            )
            return
        win = tk.Toplevel(self.root)
        fig, ax = plt.subplots(2, 1, figsize=(8, 6))
        d = self.datos_financieros
        ax[0].plot(d["fecha"], d["ingreso"], label="Ingreso")
        ax[0].plot(d["fecha"], d["gasto_essencial"], label="Gasto esencial")
        ax[0].legend()
        ax[0].tick_params(axis="x", rotation=45)
        ratios = (d["ahorro"] / (d["ingreso"] + 1e-9)) * 100
        ax[1].plot(d["fecha"], ratios, color="green", label="Ratio ahorro %")
        ax[1].legend()
        ax[1].tick_params(axis="x", rotation=45)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


# -------------------------
# main
# -------------------------
def main():
    # ruta_csv = r"c:\Users\omaroalvaradoc\Documents\Personal\Proyectos\CURSO IA\data\raw\Datos Movimientos Financieros.csv"
    ruta_csv = ""

    # Modo: podemos forzar GUI/console con argumentos o elegir interactivamente si está en un TTY.
    mode = None
    if "--gui" in sys.argv or "-g" in sys.argv:
        mode = "gui"
    elif "--console" in sys.argv or "-c" in sys.argv:
        mode = "console"
    elif sys.stdout.isatty():
        # Pedir modo al usuario si se ejecuta interactivamente
        try:
            choice = (
                input("Elige modo: (g)ui / (c)onsole / Enter = auto ").strip().lower()
            )
            if choice in ("g", "gui"):
                mode = "gui"
            elif choice in ("c", "console"):
                mode = "console"
        except Exception:
            pass

    try:
        datos = cargar_movimientos_csv(ruta_csv)
    except Exception as e:
        print("No se pudo cargar CSV, se usarán datos simulados. Error:", e)
        datos = None

    # Ejecutar en consola por defecto
    try:
        asistente = AsistenteFinancieroDifuso(datos_df=datos, root=None)
    except ImportError as ie:
        print(ie)
        print("Instala dependencias: pip install scikit-fuzzy pandas numpy")
        sys.exit(1)
    except Exception as e:
        print("Error inicializando asistente:", e)
        sys.exit(1)

    # Si el usuario selecciona GUI, intentar iniciar la interfaz (si Tkinter está disponible)
    if mode == "gui":
        if tk is None:
            print(
                "Tkinter no disponible en este entorno. Se ejecutará en modo consola."
            )
            asistente.correr_en_consola()
            return
        root = tk.Tk()

        # Bind a clean exit when the window closes
        def _on_close_and_exit():
            try:
                root.destroy()
            finally:
                # Ensure the process terminates
                sys.exit(0)

        root.protocol("WM_DELETE_WINDOW", _on_close_and_exit)

        # Ensure Ctrl+C or SIGTERM also shuts down the GUI and process gracefully
        def _signal_exit(signum, frame):
            try:
                root.destroy()
            except Exception:
                pass
            sys.exit(0)

        signal.signal(signal.SIGINT, _signal_exit)
        signal.signal(signal.SIGTERM, _signal_exit)

        try:
            asistente = AsistenteFinancieroDifuso(datos_df=datos, root=root)
        except Exception as e:
            print("Error inicializando GUI:", e)
            print("Cayendo a modo consola...")
            asistente.correr_en_consola()
            return

        # ejecutar mainloop con manejo de KeyboardInterrupt para no imprimir tracebacks
        try:
            root.mainloop()
        except KeyboardInterrupt:
            # usuario interrumpió con Ctrl+C en la terminal
            try:
                root.destroy()
            except Exception:
                pass
            print("Interrupción recibida: cerrando aplicación.")
            sys.exit(0)
    else:
        # Modo consola por defecto
        asistente.correr_en_consola()


if __name__ == "__main__":
    main()
