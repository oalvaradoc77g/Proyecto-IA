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
import shutil
import warnings

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
    from tkinter import ttk, scrolledtext, messagebox, filedialog
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
        df["Fecha_parsed"] = pd.NaT
        patron = fechas.str.match(r"^\d{4}\s[A-Za-z]{3}\s\d{2}$")
        if patron.any():
            df.loc[patron, "Fecha_parsed"] = pd.to_datetime(
                fechas.loc[patron], format="%Y %b %d", errors="coerce"
            )
        faltantes = df["Fecha_parsed"].isna()
        if faltantes.any():
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
            fechas_norm = fechas.loc[faltantes].str.upper().replace(mes_map, regex=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                df.loc[faltantes, "Fecha_parsed"] = pd.to_datetime(
                    fechas_norm, errors="coerce"
                )
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
# Clase: MiniMLP (Micro red neuronal para patrones financieros)
# Mini MLP simple para componente neuronal
# MLP es Multi-Layer Perceptron (Perceptrón Multicapa)
# -------------------------
class MiniMLP:
    # Inicializa pesos y biases
    # biases son vectores añadidos a cada capa
    # hidden es 6 debido a que es un tamaño común para una capa oculta pequeña
    # seed es 42 porque es un valor común para reproducibilidad
    def __init__(self, input_dim, hidden_dim=6, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(scale=0.1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(scale=0.1, size=(hidden_dim, 1))
        self.b2 = np.zeros(1)

    # Funciones de activación y derivadas
    def _relu(self, x):
        return np.maximum(0, x)

    # Derivada de ReLU se encarga de retornar 1 donde x>0, 0 en otro caso, x es la entrada pre-activación
    def _relu_deriv(self, x):
        return (x > 0).astype(float)

    # Función sigmoide para salida entre 0 y 1
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    # Entrenamiento con descenso de gradiente
    # epochs es la cantidad de iteraciones para entrenar se deja 400 porque es un valor común para un entrenamiento rápido
    # lr es la tasa de aprendizaje se deja 0.04 porque es un valor común para un entrenamiento rápido
    # X es la matriz de características de entrada
    # y es el vector de etiquetas o valores objetivo
    def entrenar(self, X, y, epochs=500, lr=0.05):
        if len(X) == 0:
            return
        for _ in range(epochs):
            z1 = X @ self.W1 + self.b1
            a1 = self._relu(z1)
            z2 = a1 @ self.W2 + self.b2
            y_pred = self._sigmoid(z2)
            error = y_pred - y

            grad_W2 = a1.T @ error / len(X)
            grad_b2 = error.mean(axis=0)
            da1 = error @ self.W2.T
            dz1 = da1 * self._relu_deriv(z1)
            grad_W1 = X.T @ dz1 / len(X)
            grad_b1 = dz1.mean(axis=0)

            self.W2 -= lr * grad_W2
            self.b2 -= lr * grad_b2
            self.W1 -= lr * grad_W1
            self.b1 -= lr * grad_b1

    def predecir(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        return self._sigmoid(z2)


# -------------------------
# Clase: AsistenteFinancieroDifuso (modo consola + opcional GUI)
# -------------------------
class AsistenteFinancieroDifuso:
    # Inicializa el asistente financiero difuso
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
        self.origen_datos = "CSV" if isinstance(datos_df, pd.DataFrame) else "Simulado"
        self.neuro_modelo = None
        self.ultimo_resultado = None
        self._asegurar_tipos()
        self._crear_sistema_difuso()
        self._entrenar_neuro_modelo()
        if self.root is not None:
            if tk is None:
                raise RuntimeError("Tkinter no disponible en este entorno")
            self._crear_interfaz()

    # Entrena el modelo neuronal con datos históricos simulados
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

    # Asegura que los tipos de datos en el DataFrame sean correctos
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

    # Crea el sistema difuso para la toma de decisiones financieras
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

    # Entrena el modelo neuronal con datos históricos simulados
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

    # Inferencia con el modelo neuronal
    def obtener_recomendacion(self):
        metricas = self.calcular_metricas()
        self.sim.input["ratio_ahorro"] = metricas["ratio_ahorro"]
        self.sim.input["ratio_gasto_essencial"] = metricas["ratio_gasto_essencial"]
        self.sim.input["estabilidad_ingresos"] = metricas["estabilidad_ingresos"]
        self.sim.compute()
        valor_difuso = float(self.sim.output["recomendacion"])
        valor_neuronal = self._inferir_neuro(metricas)
        valor_final = (valor_difuso * 0.6) + (valor_neuronal * 0.4)
        if valor_final <= 30:
            cat = "EMERGENCIA"
            exp = "Recortar gastos no esenciales, crear fondo de emergencia, revisar deudas."
        elif valor_final <= 60:
            cat = "CONSERVADOR"
            exp = "Mantener ahorro, instrumentos de bajo riesgo, construir fondo 3-6 meses."
        elif valor_final <= 80:
            cat = "MODERADO"
            exp = "Incrementar inversiones, considerar riesgo medio, plan largo plazo."
        else:
            cat = "AGRESIVO"
            exp = "Diversificar agresivamente, aprovechar alta capacidad de ahorro."
        return {
            "categoria": cat,
            "valor": valor_final,
            "valor_difuso": valor_difuso,
            "valor_neuronal": valor_neuronal,
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
        print(
            f"  · Difuso: {r['valor_difuso']:.1f} | Neuronal: {r['valor_neuronal']:.1f}"
        )
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
        ttk.Label(frame, text=f"Fuente de datos: {self.origen_datos}").pack(fill="x")
        tablero = ttk.LabelFrame(frame, text="Panel Neuro-Difuso", padding=10)
        tablero.pack(fill="x", pady=5)
        self.metricas_widgets = {}
        info_map = {
            "ratio_ahorro": self._mostrar_info_ratio,
            "ratio_gasto_essencial": self._mostrar_info_gasto,
            "estabilidad_ingresos": self._mostrar_info_estabilidad,
        }
        for idx, (clave, texto) in enumerate(
            [
                ("ratio_ahorro", "Ratio de Ahorro"),
                ("ratio_gasto_essencial", "Gasto Esencial"),
                ("estabilidad_ingresos", "Estabilidad Ingresos"),
            ]
        ):
            fila = ttk.Frame(tablero)
            fila.grid(row=idx, column=0, sticky="ew", pady=2)
            lbl = ttk.Label(fila, text=f"{texto}: --")
            lbl.pack(side="left")
            if clave in info_map:
                ttk.Button(fila, text="?", width=2, command=info_map[clave]).pack(
                    side="left", padx=(6, 0)
                )
            pb = ttk.Progressbar(fila, maximum=100, length=220)
            pb.pack(side="right", padx=5)
            self.metricas_widgets[clave] = (lbl, pb)
        cont_categoria = ttk.Frame(tablero)
        cont_categoria.grid(row=3, column=0, sticky="w", pady=(8, 2))
        self.lbl_categoria = ttk.Label(
            cont_categoria, text="Perfil pendiente", font=("Arial", 12, "bold")
        )
        self.lbl_categoria.pack(side="left")
        ttk.Button(
            cont_categoria, text="?", width=2, command=self._mostrar_info_perfil
        ).pack(side="left", padx=(6, 0))
        self.pb_recomendacion = ttk.Progressbar(tablero, maximum=100, length=320)
        self.pb_recomendacion.grid(row=4, column=0, sticky="ew")
        if plt is not None:
            btn_g = ttk.Button(frame, text="Ver gráficas", command=self._graficas_gui)
            btn_g.pack(fill="x", pady=5)

    # Acción al presionar el botón en la GUI
    def _accion_gui(self):
        r = self.obtener_recomendacion()
        self.ultimo_resultado = r
        m = r["metricas"]
        reporte = (
            f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Ratio Ahorro: {m['ratio_ahorro']:.1f}%\n"
            f"Ratio Gasto Esencial: {m['ratio_gasto_essencial']:.1f}%\n"
            f"Estabilidad: {m['estabilidad_ingresos']:.1f}%\n"
            f"Ingreso Promedio: ${m['ingreso_promedio']:.2f}\n"
            f"Ahorro Promedio: ${m['ahorro_promedio']:.2f}\n\n"
            f"Recomendación: {r['categoria']} ({r['valor']:.1f})\n"
            f"  · Difuso: {r['valor_difuso']:.1f} | Neuronal: {r['valor_neuronal']:.1f}\n"
            f"{r['explicacion']}\n"
            f"{self._explicar_componentes(r)}\n"
        )
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", reporte)
        self._actualizar_panel_metricas(m, r)

    # Explicación detallada de los componentes
    def _actualizar_panel_metricas(self, metricas, resultado):
        if not hasattr(self, "metricas_widgets"):
            return
        for clave, (lbl, pb) in self.metricas_widgets.items():
            valor = metricas.get(clave, 0.0)
            titulo = lbl.cget("text").split(":")[0]
            lbl.config(text=f"{titulo}: {valor:.1f}%")
            pb["value"] = valor
        self.lbl_categoria.config(
            text=f"Perfil {resultado['categoria']} · {resultado['valor']:.1f}"
        )
        self.pb_recomendacion["value"] = resultado["valor"]

    # Explicación de los componentes difuso y neuronal
    def _mostrar_info_ratio(self):
        mensaje = (
            "Ratio de Ahorro = (Ahorro promedio / Ingreso promedio) * 100.\n"
            "Un porcentaje alto implica holgura financiera y resiliencia ante emergencias."
        )
        (
            messagebox.showinfo("¿Qué es el Ratio de Ahorro?", mensaje)
            if messagebox
            else print(mensaje)
        )

    # Explicación de los componentes difuso y neuronal
    def _mostrar_info_gasto(self):
        mensaje = (
            "Gasto Esencial % = (Gasto esencial / Ingreso promedio) * 100.\n"
            "Describe cuánto de tus ingresos se dirige a necesidades básicas; "
            "superar 60% suele limitar el ahorro."
        )
        (
            messagebox.showinfo("¿Qué es el Gasto Esencial?", mensaje)
            if messagebox
            else print(mensaje)
        )

    # Explicación de los componentes difuso y neuronal
    def _mostrar_info_estabilidad(self):
        mensaje = (
            "Estabilidad de Ingresos calcula la variación de tus ingresos recientes.\n"
            "Valores cercanos a 100% significan ingresos consistentes; bajos implican volatilidad."
        )
        (
            messagebox.showinfo("¿Qué es la Estabilidad de Ingresos?", mensaje)
            if messagebox
            else print(mensaje)
        )

    # Explicación del perfil financiero
    def _mostrar_info_perfil(self):
        if self.ultimo_resultado:
            res = self.ultimo_resultado
            mensaje = (
                f"Perfil {res['categoria']} ({res['valor']:.1f}).\n"
                f"Componente difuso: {res['valor_difuso']:.1f}\n"
                f"Componente neuronal: {res['valor_neuronal']:.1f}\n\n"
                f"{res['explicacion']}"
            )
        else:
            mensaje = "Realiza un análisis primero para obtener tu perfil financiero."
        (
            messagebox.showinfo("Detalle del Perfil", mensaje)
            if messagebox
            else print(mensaje)
        )

    # Explicación detallada de los componentes
    def obtener_recomendacion(self):
        metricas = self.calcular_metricas()
        self.sim.input["ratio_ahorro"] = metricas["ratio_ahorro"]
        self.sim.input["ratio_gasto_essencial"] = metricas["ratio_gasto_essencial"]
        self.sim.input["estabilidad_ingresos"] = metricas["estabilidad_ingresos"]
        self.sim.compute()
        valor_difuso = float(self.sim.output["recomendacion"])
        valor_neuronal = self._inferir_neuro(metricas)
        valor_final = (valor_difuso * 0.6) + (valor_neuronal * 0.4)
        if valor_final <= 30:
            cat = "EMERGENCIA"
            exp = "Recortar gastos no esenciales, crear fondo de emergencia, revisar deudas."
        elif valor_final <= 60:
            cat = "CONSERVADOR"
            exp = "Mantener ahorro, instrumentos de bajo riesgo, construir fondo 3-6 meses."
        elif valor_final <= 80:
            cat = "MODERADO"
            exp = "Incrementar inversiones, considerar riesgo medio, plan largo plazo."
        else:
            cat = "AGRESIVO"
            exp = "Diversificar agresivamente, aprovechar alta capacidad de ahorro."
        return {
            "categoria": cat,
            "valor": valor_final,
            "valor_difuso": valor_difuso,
            "valor_neuronal": valor_neuronal,
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
        print(
            f"  · Difuso: {r['valor_difuso']:.1f} | Neuronal: {r['valor_neuronal']:.1f}"
        )
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
        ttk.Label(frame, text=f"Fuente de datos: {self.origen_datos}").pack(fill="x")
        tablero = ttk.LabelFrame(frame, text="Panel Neuro-Difuso", padding=10)
        tablero.pack(fill="x", pady=5)
        self.metricas_widgets = {}
        info_map = {
            "ratio_ahorro": self._mostrar_info_ratio,
            "ratio_gasto_essencial": self._mostrar_info_gasto,
            "estabilidad_ingresos": self._mostrar_info_estabilidad,
        }
        for idx, (clave, texto) in enumerate(
            [
                ("ratio_ahorro", "Ratio de Ahorro"),
                ("ratio_gasto_essencial", "Gasto Esencial"),
                ("estabilidad_ingresos", "Estabilidad Ingresos"),
            ]
        ):
            fila = ttk.Frame(tablero)
            fila.grid(row=idx, column=0, sticky="ew", pady=2)
            lbl = ttk.Label(fila, text=f"{texto}: --")
            lbl.pack(side="left")
            if clave in info_map:
                ttk.Button(fila, text="?", width=2, command=info_map[clave]).pack(
                    side="left", padx=(6, 0)
                )
            pb = ttk.Progressbar(fila, maximum=100, length=220)
            pb.pack(side="right", padx=5)
            self.metricas_widgets[clave] = (lbl, pb)
        cont_categoria = ttk.Frame(tablero)
        cont_categoria.grid(row=3, column=0, sticky="w", pady=(8, 2))
        self.lbl_categoria = ttk.Label(
            cont_categoria, text="Perfil pendiente", font=("Arial", 12, "bold")
        )
        self.lbl_categoria.pack(side="left")
        ttk.Button(
            cont_categoria, text="?", width=2, command=self._mostrar_info_perfil
        ).pack(side="left", padx=(6, 0))
        self.pb_recomendacion = ttk.Progressbar(tablero, maximum=100, length=320)
        self.pb_recomendacion.grid(row=4, column=0, sticky="ew")
        if plt is not None:
            btn_g = ttk.Button(frame, text="Ver gráficas", command=self._graficas_gui)
            btn_g.pack(fill="x", pady=5)

    # Acción al presionar el botón en la GUI
    def _accion_gui(self):
        r = self.obtener_recomendacion()
        self.ultimo_resultado = r
        m = r["metricas"]
        reporte = (
            f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Ratio Ahorro: {m['ratio_ahorro']:.1f}%\n"
            f"Ratio Gasto Esencial: {m['ratio_gasto_essencial']:.1f}%\n"
            f"Estabilidad: {m['estabilidad_ingresos']:.1f}%\n"
            f"Ingreso Promedio: ${m['ingreso_promedio']:.2f}\n"
            f"Ahorro Promedio: ${m['ahorro_promedio']:.2f}\n\n"
            f"Recomendación: {r['categoria']} ({r['valor']:.1f})\n"
            f"  · Difuso: {r['valor_difuso']:.1f} | Neuronal: {r['valor_neuronal']:.1f}\n"
            f"{r['explicacion']}\n"
            f"{self._explicar_componentes(r)}\n"
        )
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", reporte)
        self._actualizar_panel_metricas(m, r)

    # Actualiza el panel de métricas en la GUI
    def _actualizar_panel_metricas(self, metricas, resultado):
        if not hasattr(self, "metricas_widgets"):
            return
        for clave, (lbl, pb) in self.metricas_widgets.items():
            valor = metricas.get(clave, 0.0)
            titulo = lbl.cget("text").split(":")[0]
            lbl.config(text=f"{titulo}: {valor:.1f}%")
            pb["value"] = valor
        self.lbl_categoria.config(
            text=f"Perfil {resultado['categoria']} · {resultado['valor']:.1f}"
        )
        self.pb_recomendacion["value"] = resultado["valor"]

    # Explicación detallada de los componentes
    def _mostrar_info_ratio(self):
        mensaje = (
            "Ratio de Ahorro = (Ahorro promedio / Ingreso promedio) * 100.\n"
            "Un porcentaje alto implica holgura financiera y resiliencia ante emergencias."
        )
        (
            messagebox.showinfo("¿Qué es el Ratio de Ahorro?", mensaje)
            if messagebox
            else print(mensaje)
        )

    # Explicación detallada de los componentes
    def _mostrar_info_gasto(self):
        mensaje = (
            "Gasto Esencial % = (Gasto esencial / Ingreso promedio) * 100.\n"
            "Describe cuánto de tus ingresos se dirige a necesidades básicas; "
            "superar 60% suele limitar el ahorro."
        )
        (
            messagebox.showinfo("¿Qué es el Gasto Esencial?", mensaje)
            if messagebox
            else print(mensaje)
        )

    # Explicación detallada de los componentes
    def _mostrar_info_estabilidad(self):
        mensaje = (
            "Estabilidad de Ingresos calcula la variación de tus ingresos recientes.\n"
            "Valores cercanos a 100% significan ingresos consistentes; bajos implican volatilidad."
        )
        (
            messagebox.showinfo("¿Qué es la Estabilidad de Ingresos?", mensaje)
            if messagebox
            else print(mensaje)
        )

    # Explicación del perfil financiero
    def _mostrar_info_perfil(self):
        if self.ultimo_resultado:
            res = self.ultimo_resultado
            mensaje = (
                f"Perfil {res['categoria']} ({res['valor']:.1f}).\n"
                f"Componente difuso: {res['valor_difuso']:.1f}\n"
                f"Componente neuronal: {res['valor_neuronal']:.1f}\n\n"
                f"{res['explicacion']}"
            )
        else:
            mensaje = "Realiza un análisis primero para obtener tu perfil financiero."
        (
            messagebox.showinfo("Detalle del Perfil", mensaje)
            if messagebox
            else print(mensaje)
        )

    # Explicación detallada de los componentes
    def _explicar_componentes(self, resultado):
        difuso = resultado["valor_difuso"]
        neuronal = resultado["valor_neuronal"]
        if abs(difuso - neuronal) <= 3:
            motivo = "Ambos módulos coinciden: las reglas difusas y la red neuronal ven un perfil similar."
        elif difuso > neuronal:
            motivo = (
                "El componente difuso domina porque las métricas actuales encajan mejor con las reglas "
                "de ahorro, gasto esencial y estabilidad predefinidas."
            )
        else:
            motivo = (
                "El componente neuronal domina al detectar patrones recientes que sugieren "
                "un comportamiento distinto al esperado por las reglas difusas."
            )
        return f"Interpretación del score -> Difuso: {difuso:.1f} vs Neuronal: {neuronal:.1f}. {motivo}"

    # Acción al presionar el botón en la GUI
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

    # Preparar dataset para el modelo neuronal
    def _preparar_dataset_neuro(self):
        df = self.datos_financieros.copy()
        if df.empty:
            return None, None
        if "fecha" in df.columns:
            df = df.sort_values("fecha")
        divisiones = max(len(df) // 15, 1)
        bloques_idx = np.array_split(np.arange(len(df)), divisiones)
        filas = []
        for idx in bloques_idx:
            bloque = df.iloc[idx]
            if bloque.empty:
                continue
            ingreso = bloque["ingreso"].mean()
            ahorro = bloque["ahorro"].mean()
            gasto = bloque["gasto_essencial"].mean()
            estabilidad = 100 - (bloque["ingreso"].std() / (ingreso + 1e-9)) * 100
            feat = [
                np.clip((ahorro / (ingreso + 1e-9)) * 100, 0, 100),
                np.clip((gasto / (ingreso + 1e-9)) * 100, 0, 100),
                np.clip(estabilidad, 0, 100),
            ]
            objetivo = (feat[0] * 0.5 + (100 - feat[1]) * 0.3 + feat[2] * 0.2) / 100
            filas.append((feat, objetivo))
        if not filas:
            return None, None
        X = np.array([f[0] for f in filas]) / 100.0
        y = np.array([[f[1]] for f in filas])
        return X, y

    # Entrenar el modelo neuronal
    def _entrenar_neuro_modelo(self):
        X, y = self._preparar_dataset_neuro()
        if X is None:
            self.neuro_modelo = None
            return
        self.neuro_modelo = MiniMLP(input_dim=X.shape[1], hidden_dim=6)
        self.neuro_modelo.entrenar(X, y, epochs=400, lr=0.04)

    # Inferencia con el modelo neuronal
    def _inferir_neuro(self, metricas):
        if self.neuro_modelo is None:
            return metricas["ratio_ahorro"]
        vector = np.array(
            [
                [
                    metricas["ratio_ahorro"],
                    metricas["ratio_gasto_essencial"],
                    metricas["estabilidad_ingresos"],
                ]
            ]
        )
        pred = float(self.neuro_modelo.predecir(vector / 100.0)[0][0]) * 100
        return float(np.clip(pred, 0, 100))

    # Mostrar gráficas en una ventana nueva
    def _graficas_gui(self):
        if plt is None or FigureCanvasTkAgg is None:
            messagebox.showwarning(
                "Dependencia", "matplotlib no disponible para gráficas"
            )
            return
        win = tk.Toplevel(self.root)
        win.title("Análisis gráfico neuro-difuso")
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        datos = self.datos_financieros
        axes[0, 0].plot(datos["fecha"], datos["ingreso"], label="Ingreso")
        axes[0, 0].plot(
            datos["fecha"], datos["gasto_essencial"], label="Gasto esencial"
        )
        axes[0, 0].set_title("Ingresos vs gasto esencial")
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis="x", rotation=45)

        axes[0, 1].plot(datos["fecha"], datos["ahorro"], color="green", label="Ahorro")
        axes[0, 1].set_title("Ahorro acumulado")
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis="x", rotation=45)

        ratios = (datos["ahorro"] / (datos["ingreso"] + 1e-9)) * 100
        axes[1, 0].plot(datos["fecha"], ratios, color="purple", label="Ratio ahorro %")
        axes[1, 0].set_title("Ratio de ahorro (%)")
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis="x", rotation=45)

        axes[1, 1].hist(datos["ingreso"], bins=15, color="orange", alpha=0.7)
        axes[1, 1].set_title("Distribución ingresos")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        btns_frame = ttk.Frame(win)
        btns_frame.pack(fill="x", pady=4)
        ttk.Button(
            btns_frame,
            text="ℹ Ingresos vs gasto",
            command=lambda: self._explicar_ingresos_gasto(datos),
        ).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(
            btns_frame,
            text="ℹ Ahorro acumulado",
            command=lambda: self._explicar_ahorro(datos),
        ).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(
            btns_frame,
            text="ℹ Ratio ahorro",
            command=lambda: self._explicar_ratio(datos),
        ).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(
            btns_frame,
            text="ℹ Distribución ingresos",
            command=lambda: self._explicar_histograma_dyn(datos),
        ).pack(side="left", expand=True, fill="x", padx=2)

    # Explicación del perfil financiero
    def _explicar_ingresos_gasto(self, datos):
        total_ing = float(datos["ingreso"].sum())
        ess = float(datos["gasto_essencial"].sum())
        pct_ess = (ess / (total_ing + 1e-9)) * 100
        mensaje = (
            "Ingresos vs gasto esencial:\n\n"
            f"Ingreso total observado: ${total_ing:,.0f}\n"
            f"Gasto esencial total: ${ess:,.0f} ({pct_ess:.1f}% del ingreso)\n\n"
            "Interpretación: un porcentaje elevado del ingreso destinado a gastos básicos reduce margen de ahorro."
        )
        (
            messagebox.showinfo("Ingresos vs gasto esencial", mensaje)
            if messagebox
            else print(mensaje)
        )

    # Explicación del perfil financiero
    def _explicar_ahorro(self, datos):
        ahorros = datos["ahorro"].astype(float)
        prom = ahorros.mean()
        med = ahorros.median()
        tendencia = "estable" if ahorros.std() < prom * 0.25 else "variable"
        mensaje = (
            "Ahorro acumulado:\n\n"
            f"Ahorro medio diario: ${prom:,.0f}\n"
            f"Ahorro mediano diario: ${med:,.0f}\n"
            f"Variabilidad: {tendencia}\n\n"
            "Interpretación: revisar consistencia; si es variable, conviene calendarizar aportes."
        )
        (
            messagebox.showinfo("Ahorro acumulado", mensaje)
            if messagebox
            else print(mensaje)
        )

    # Explicación del perfil financiero
    def _explicar_ratio(self, datos):
        ratios = (datos["ahorro"] / (datos["ingreso"] + 1e-9)) * 100
        prom = ratios.mean()
        med = np.median(ratios)
        maxv = ratios.max()
        mensaje = (
            "Ratio de ahorro (% sobre ingreso):\n\n"
            f"Promedio: {prom:.1f}%\n"
            f"Mediana: {med:.1f}%\n"
            f"Máximo observado: {maxv:.1f}%\n\n"
            "Interpretación: valores crecientes indican mejora; bajo promedio sugiere optimizar gastos no esenciales."
        )
        (
            messagebox.showinfo("Ratio de ahorro", mensaje)
            if messagebox
            else print(mensaje)
        )

    # Explicación del perfil financiero
    def _generar_explicacion_histograma(self, datos):
        ingresos = datos.get("ingreso", pd.Series(dtype=float)).astype(float).dropna()
        if ingresos.empty:
            return "No hay datos suficientes para analizar la distribución."
        rangos = [
            (0, 10_000_000, "0-10M"),
            (10_000_000, 20_000_000, "10-20M"),
            (20_000_000, 30_000_000, "20-30M"),
            (30_000_000, 40_000_000, "30-40M"),
            (40_000_000, None, "40M+"),
        ]
        lineas = []
        for low, high, etiqueta in rangos:
            if high is None:
                mask = ingresos >= low
            else:
                mask = (ingresos >= low) & (ingresos < high)
            count = int(mask.sum())
            if count == 0:
                cuali = "rango ausente"
            elif count <= 2:
                cuali = "ocasional"
            elif count <= 5:
                cuali = "frecuencia media"
            else:
                cuali = "muy frecuente"
            lineas.append(f"• {etiqueta}: {count} registros ({cuali})")
        pico = ingresos.median()
        resumen = (
            f"Mediana aproximada: ${pico:,.0f}. Concentración principal cerca de "
            f"{self._etiqueta_rango(pico)}."
        )
        return (
            "📊 Distribución de ingresos (histograma)\n\n"
            "Eje X: montos en pesos.\nEje Y: frecuencia de aparición en cada rango.\n\n"
            + "\n".join(lineas)
            + "\n\n"
            + resumen
        )

    # Explicación del perfil financiero
    def _etiqueta_rango(self, valor):
        if valor < 10_000_000:
            return "0-10M"
        if valor < 20_000_000:
            return "10-20M"
        if valor < 30_000_000:
            return "20-30M"
        if valor < 40_000_000:
            return "30-40M"
        return "40M+"

    # Explicación del perfil financiero
    def _explicar_histograma_dyn(self, datos):
        mensaje = self._generar_explicacion_histograma(datos)
        (
            messagebox.showinfo("Distribución de ingresos", mensaje)
            if messagebox
            else print(mensaje)
        )

    # Guardar script como archivo de texto


# -------------------------
# main
# -------------------------
def main():
    ruta_csv = r"c:\Users\omaroalvaradoc\Documents\Personal\Proyectos\CURSO IA\data\raw\Datos Movimientos Financieros.csv"
    # ruta_csv = ""

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
