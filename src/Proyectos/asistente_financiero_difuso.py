"""
ASISTENTE FINANCIERO CON LÓGICA DIFUSA
Bootcamp Project - Inteligencia Artificial

Notas:
 - Contiene una interfaz gráfica mínima con Tkinter (usando 'root' si se pasa).
 - Implementa IA simbólica mediante Lógica Difusa (scikit-fuzzy).
 - Si no hay CSV válido, usa datos simulados; también puede procesar extractos simples.
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Import opcionales para GUI y difuso; reportar si no están instalados
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    tk = None  # GUI no disponible en entorno sin tkinter

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except Exception as e:
    fuzz = None
    ctrl = None


# Mini MLP simple para componente neuronal
# MLP es Multi-Layer Perceptron (Perceptrón Multicapa)
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
        return 1 / (1 + np.exp(-x))

    # Entrenamiento con descenso de gradiente
    # epochs es la cantidad de iteraciones para entrenar se deja 400 porque es un valor común para un entrenamiento rápido
    # lr es la tasa de aprendizaje se deja 0.04 porque es un valor común para un entrenamiento rápido
    # X es la matriz de características de entrada
    # y es el vector de etiquetas o valores objetivo
    def entrenar(self, X, y, epochs=400, lr=0.04):
        if len(X) == 0:
            return
        for _ in range(epochs):
            z1 = X @ self.W1 + self.b1
            a1 = self._relu(z1)
            z2 = a1 @ self.W2 + self.b2
            y_pred = self._sigmoid(z2)
            err = y_pred - y
            grad_W2 = a1.T @ err / len(X)
            grad_b2 = err.mean(axis=0)
            da1 = err @ self.W2.T
            dz1 = da1 * self._relu_deriv(z1)
            grad_W1 = X.T @ dz1 / len(X)
            grad_b1 = dz1.mean(axis=0)
            self.W2 -= lr * grad_W2
            self.b2 -= lr * grad_b2
            self.W1 -= lr * grad_W1
            self.b1 -= lr * grad_b1

    # Predicción usando la red neuronal entrenada
    def predecir(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        return self._sigmoid(z2)


class AsistenteFinancieroDifuso:
    # Clase principal para el asistente financiero con lógica difusa y componente neuronal
    def __init__(self, root=None, datos_externos=None):
        """
        root: objeto Tk (si se desea GUI). Si es None, el asistente funciona en modo consola.
        datos_externos: pd.DataFrame o ruta a CSV con columnas mínimas:
           ['fecha'|'Fecha', 'ingreso'|'Créditos', 'Débitos'|'Débitos', 'ahorro' opcional]
           Si se pasa un CSV bruto del extracto, se espera que la función externa lo preprocese.
        """
        self.root = root
        self.datos_financieros = None
        self.origen_datos = "Simulado"
        if isinstance(datos_externos, pd.DataFrame):
            self.origen_datos = "DataFrame externo"
        elif isinstance(datos_externos, str) and os.path.exists(datos_externos):
            self.origen_datos = "CSV"
        self.neuro_modelo = None
        self.ultimo_resultado = None

        # Validar disponibilidad de skfuzzy
        if fuzz is None or ctrl is None:
            raise ImportError(
                "skfuzzy no está instalado. Ejecuta: pip install scikit-fuzzy"
            )

        # Cargar datos externos si se proporcionan
        if isinstance(datos_externos, pd.DataFrame):
            self.datos_financieros = datos_externos.copy()
        elif isinstance(datos_externos, str) and os.path.exists(datos_externos):
            # Intentar leer CSV simple; el usuario ya tiene una función de carga en prroyecto_bootcamp.py
            try:
                df = pd.read_csv(
                    datos_externos,
                    parse_dates=["Fecha"],
                    dayfirst=True,
                    encoding="utf-8",
                    dtype=str,
                )
                # Normalizar columnas si vienen en formato del extracto
                if (
                    "Débitos" in df.columns
                    and "Créditos" in df.columns
                    and "Fecha" in df.columns
                ):
                    df["Débitos"] = (
                        df["Débitos"].str.replace(",", "").astype(float).fillna(0)
                    )
                    df["Créditos"] = (
                        df["Créditos"].str.replace(",", "").astype(float).fillna(0)
                    )
                    df["fecha"] = pd.to_datetime(
                        df["Fecha"], dayfirst=True, errors="coerce"
                    )
                    df["ingreso"] = df["Créditos"]
                    df["gasto_essencial"] = (
                        df["Débitos"] * 0.6
                    )  # heurística si no hay etiqueta
                    df["gasto_no_essencial"] = df["Débitos"] * 0.4
                    df["ahorro"] = (
                        df["ingreso"] - df["gasto_essencial"] - df["gasto_no_essencial"]
                    )
                    self.datos_financieros = df[
                        [
                            "fecha",
                            "ingreso",
                            "gasto_essencial",
                            "gasto_no_essencial",
                            "ahorro",
                        ]
                    ]
                else:
                    # si no corresponde intentar usar como tabla ya preprocesada
                    self.datos_financieros = pd.read_csv(
                        datos_externos, parse_dates=["fecha"], dayfirst=True
                    )
            except Exception:
                # fallback a simulados
                self.datos_financieros = self.generar_datos_simulados()
        else:
            # Si no hay datos externos, generar datos simulados (90 días)
            self.datos_financieros = self.generar_datos_simulados()

        # Asegurar tipos numéricos y columna fecha
        if "fecha" in self.datos_financieros.columns:
            self.datos_financieros["fecha"] = pd.to_datetime(
                self.datos_financieros["fecha"], errors="coerce"
            )
        for col in ["ingreso", "gasto_essencial", "gasto_no_essencial", "ahorro"]:
            if col in self.datos_financieros.columns:
                self.datos_financieros[col] = pd.to_numeric(
                    self.datos_financieros[col], errors="coerce"
                ).fillna(0)

        # Sistema difuso
        self.sistema_control = None
        self.crear_sistema_difuso()
        self.crear_modelo_neuronal()

        # Crear GUI si se pidió
        if self.root is not None:
            if tk is None:
                raise RuntimeError("Tkinter no está disponible en este entorno")
            self.crear_interfaz()

    # Generar datos simulados
    def generar_datos_simulados(self, dias=90):
        """Genera datos financieros simulados (diarios) y devuelve DataFrame."""
        fechas = [datetime.now() - timedelta(days=x) for x in range(dias)]
        datos = []
        for fecha in reversed(fechas):
            gasto_essencial = random.uniform(800, 1200)
            gasto_no_essencial = random.uniform(200, 500)
            ahorro = random.uniform(100, 800)
            ingreso = gasto_essencial + gasto_no_essencial + ahorro
            datos.append(
                {
                    "fecha": fecha,
                    "ingreso": ingreso,
                    "gasto_essencial": gasto_essencial,
                    "gasto_no_essencial": gasto_no_essencial,
                    "ahorro": ahorro,
                }
            )
        return pd.DataFrame(datos)

    # Crear sistema difuso
    def crear_sistema_difuso(self):
        """Construye el sistema de lógica difusa (mismas reglas y membresías mejoradas)."""
        # Variables de entrada
        self.ratio_ahorro = ctrl.Antecedent(np.arange(0, 101, 1), "ratio_ahorro")
        self.ratio_gasto_essencial = ctrl.Antecedent(
            np.arange(0, 101, 1), "ratio_gasto_essencial"
        )
        self.estabilidad_ingresos = ctrl.Antecedent(
            np.arange(0, 101, 1), "estabilidad_ingresos"
        )
        # Salida
        self.recomendacion = ctrl.Consequent(np.arange(0, 101, 1), "recomendacion")

        # Membresías (ligeramente ajustadas)
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

        # Reglas
        regla1 = ctrl.Rule(
            self.ratio_ahorro["bajo"] & self.ratio_gasto_essencial["alto"],
            self.recomendacion["emergencia"],
        )
        regla2 = ctrl.Rule(
            self.ratio_ahorro["bajo"] & self.ratio_gasto_essencial["medio"],
            self.recomendacion["conservador"],
        )
        regla3 = ctrl.Rule(
            self.ratio_ahorro["medio"] & self.estabilidad_ingresos["alta"],
            self.recomendacion["moderado"],
        )
        regla4 = ctrl.Rule(
            self.ratio_ahorro["alto"]
            & self.estabilidad_ingresos["alta"]
            & self.ratio_gasto_essencial["bajo"],
            self.recomendacion["agresivo"],
        )
        regla5 = ctrl.Rule(
            self.ratio_ahorro["medio"] & self.estabilidad_ingresos["baja"],
            self.recomendacion["conservador"],
        )
        regla6 = ctrl.Rule(
            self.ratio_ahorro["alto"] & self.estabilidad_ingresos["media"],
            self.recomendacion["moderado"],
        )

        sistema_recomendacion = ctrl.ControlSystem(
            [regla1, regla2, regla3, regla4, regla5, regla6]
        )
        self.sistema_control = ctrl.ControlSystemSimulation(sistema_recomendacion)

    # Calcular métricas financieras
    def calcular_metricas(self):
        """Calcula métricas a partir del DataFrame interno. Retorna dict."""
        datos = self.datos_financieros.copy()
        # Si hay datos diarios, agregamos por mes para métricas más estables
        if "fecha" in datos.columns and pd.api.types.is_datetime64_any_dtype(
            datos["fecha"]
        ):
            # tomar últimos 90 días y agrupar por mes
            fecha_corte = datos["fecha"].max()
            ventana = fecha_corte - pd.Timedelta(days=90)
            datos_vent = datos[datos["fecha"] >= ventana]
            if datos_vent.empty:
                datos_vent = datos
            # promedio simple
            ingreso_promedio = datos_vent["ingreso"].mean()
            ahorro_promedio = (
                datos_vent["ahorro"].mean() if "ahorro" in datos_vent.columns else 0.0
            )
            gasto_essencial_promedio = (
                datos_vent["gasto_essencial"].mean()
                if "gasto_essencial" in datos_vent.columns
                else 0.0
            )
            # estabilidad: coef de variación transformado a índice (100 es muy estable)
            estabilidad_ingresos = (
                100
                - (datos_vent["ingreso"].std() / (datos_vent["ingreso"].mean() + 1e-9))
                * 100
            )
        else:
            ingreso_promedio = datos["ingreso"].mean()
            ahorro_promedio = datos.get("ahorro", pd.Series([0])).mean()
            gasto_essencial_promedio = datos.get(
                "gasto_essencial", pd.Series([0])
            ).mean()
            estabilidad_ingresos = (
                100 - (datos["ingreso"].std() / (datos["ingreso"].mean() + 1e-9)) * 100
            )

        ratio_ahorro = (ahorro_promedio / (ingreso_promedio + 1e-9)) * 100
        ratio_gasto_essencial = (
            gasto_essencial_promedio / (ingreso_promedio + 1e-9)
        ) * 100

        return {
            "ratio_ahorro": float(np.clip(ratio_ahorro, 0, 100)),
            "ratio_gasto_essencial": float(np.clip(ratio_gasto_essencial, 0, 100)),
            "estabilidad_ingresos": float(np.clip(estabilidad_ingresos, 0, 100)),
            "ingreso_promedio": float(ingreso_promedio),
            "ahorro_promedio": float(ahorro_promedio),
        }

    # Crear modelo neuronal
    def crear_modelo_neuronal(self):
        X, y = self._preparar_dataset_neuro()
        if X is None:
            self.neuro_modelo = None
            return
        self.neuro_modelo = MiniMLP(input_dim=X.shape[1], hidden_dim=6)
        self.neuro_modelo.entrenar(X, y, epochs=450, lr=0.03)

    # Preparar dataset para componente neuronal
    def _preparar_dataset_neuro(self):
        df = self.datos_financieros.copy()
        if df.empty:
            return None, None
        if "fecha" in df.columns:
            df = df.sort_values("fecha")
        divisiones = max(len(df) // 15, 1)
        bloques_idx = np.array_split(np.arange(len(df)), divisiones)
        payload = []
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
            payload.append((feat, objetivo))
        if not payload:
            return None, None
        X = np.array([p[0] for p in payload]) / 100.0
        y = np.array([[p[1]] for p in payload])
        return X, y

    # Inferir recomendación con componente neuronal
    def _inferir_neuro(self, metricas):
        if self.neuro_modelo is None:
            return metricas["ratio_ahorro"]
        v = np.array(
            [
                [
                    metricas["ratio_ahorro"],
                    metricas["ratio_gasto_essencial"],
                    metricas["estabilidad_ingresos"],
                ]
            ]
        )
        pred = float(self.neuro_modelo.predecir(v / 100.0)[0][0]) * 100
        return float(np.clip(pred, 0, 100))

    # Obtener recomendación final
    def obtener_recomendacion(self):
        metricas = self.calcular_metricas()
        self.sistema_control.input["ratio_ahorro"] = metricas["ratio_ahorro"]
        self.sistema_control.input["ratio_gasto_essencial"] = metricas[
            "ratio_gasto_essencial"
        ]
        self.sistema_control.input["estabilidad_ingresos"] = metricas[
            "estabilidad_ingresos"
        ]
        self.sistema_control.compute()
        valor_difuso = float(self.sistema_control.output["recomendacion"])
        valor_neuronal = self._inferir_neuro(metricas)
        valor_final = (valor_difuso * 0.6) + (valor_neuronal * 0.4)
        # Categoría y explicación (mismos textos, retornan en string)
        if valor_final <= 30:
            categoria = "EMERGENCIA"
            explicacion = "Acciones: recortar gastos no esenciales, crear fondo de emergencia, revisar deudas."
        elif valor_final <= 60:
            categoria = "CONSERVADOR"
            explicacion = "Acciones: mantener ahorro, instrumentos de bajo riesgo, construir fondo 3-6 meses."
        elif valor_final <= 80:
            categoria = "MODERADO"
            explicacion = "Acciones: incrementar inversiones, considerar riesgo medio, plan largo plazo."
        else:
            categoria = "AGRESIVO"
            explicacion = "Acciones: diversificar agresivamente, aprovechar alta capacidad de ahorro."
        return {
            "categoria": categoria,
            "valor": valor_final,
            "valor_difuso": valor_difuso,
            "valor_neuronal": valor_neuronal,
            "explicacion": explicacion,
            "metricas": metricas,
        }

    # -----------------------------------------------------------------
    # Métodos opcionales de GUI (solo se activan si root no es None)
    # -----------------------------------------------------------------
    def crear_interfaz(self):
        """Crea interfaz Tk (si root existe)."""
        self.root.title("🤖 Asistente Financiero con Lógica Difusa")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f8ff")

        # Al cerrar la ventana, terminar el proceso para no dejar la terminal esperando
        def _on_close_gui():
            try:
                self.root.destroy()
            finally:
                # Forzar la terminación del proceso
                import sys

                sys.exit(0)

        self.root.protocol("WM_DELETE_WINDOW", _on_close_gui)

        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        titulo = ttk.Label(
            main_frame,
            text="🤖 Asistente Financiero con Lógica Difusa",
            font=("Arial", 16, "bold"),
            foreground="#2c3e50",
        )
        titulo.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        btn_analizar = ttk.Button(
            main_frame,
            text="🔍 Analizar Mi Situación Financiera",
            command=self._mostrar_analisis_gui,
            style="Accent.TButton",
        )
        btn_analizar.grid(row=1, column=0, columnspan=3, pady=10, sticky="ew")

        ttk.Label(
            main_frame, text=f"Fuente de datos: {self.origen_datos}", font=("Arial", 10)
        ).grid(row=1, column=0, columnspan=3, sticky="w")
        info_map = {
            "ratio_ahorro": self._mostrar_info_ratio,
            "ratio_gasto_essencial": self._mostrar_info_gasto,
            "estabilidad_ingresos": self._mostrar_info_estabilidad,
        }
        frame_metricas = ttk.LabelFrame(
            main_frame, text="📊 Métricas Financieras", padding="10"
        )
        frame_metricas.grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)
        frame_metricas.columnconfigure(0, weight=1)

        for idx, (clave, texto) in enumerate(
            [
                ("ratio_ahorro", "Ratio de Ahorro"),
                ("ratio_gasto_essencial", "Gasto Esencial"),
                ("estabilidad_ingresos", "Estabilidad Ingresos"),
            ]
        ):
            fila = ttk.Frame(frame_metricas)
            fila.grid(row=idx, column=0, sticky="ew", pady=3)
            lbl = ttk.Label(fila, text=f"{texto}: --")
            lbl.pack(side="left")
            if clave in info_map:
                ttk.Button(fila, text="?", width=2, command=info_map[clave]).pack(
                    side="left", padx=(6, 0)
                )
            pb = ttk.Progressbar(fila, maximum=100, length=260)
            pb.pack(side="right")
            frame_metricas.columnconfigure(0, weight=1)
        cont_estado = ttk.Frame(main_frame)
        cont_estado.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        self.lbl_estado = ttk.Label(
            cont_estado,
            text="Score neuro-difuso pendiente",
            font=("Arial", 12, "bold"),
            foreground="#1c5d99",
        )
        self.lbl_estado.pack(side="left")
        ttk.Button(
            cont_estado, text="?", width=2, command=self._mostrar_info_perfil
        ).pack(side="left", padx=(6, 0))
        self.pb_estado = ttk.Progressbar(main_frame, maximum=100)
        self.pb_estado.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5)

        self.texto_resultados = scrolledtext.ScrolledText(
            main_frame, width=80, height=20, font=("Consolas", 10), wrap=tk.WORD
        )
        self.texto_resultados.grid(row=5, column=0, columnspan=3, sticky="ew", pady=10)

        btn_graficas = ttk.Button(
            main_frame, text="📈 Ver Análisis Gráfico", command=self.mostrar_graficas
        )
        btn_graficas.grid(row=6, column=0, columnspan=3, pady=10, sticky="ew")

        self.configurar_estilos()

    # Configurar estilos para la interfaz gráfica
    def configurar_estilos(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background="#f0f8ff")
        style.configure("TLabel", background="#f0f8ff", font=("Arial", 10))
        style.configure("TLabelframe", background="#f0f8ff", font=("Arial", 10, "bold"))
        style.configure(
            "TLabelframe.Label", background="#f0f8ff", font=("Arial", 10, "bold")
        )
        style.configure("Accent.TButton", font=("Arial", 11, "bold"))

    # Mostrar análisis en GUI
    def _mostrar_analisis_gui(self):
        resultado = self.obtener_recomendacion()
        self.ultimo_resultado = resultado
        if resultado:
            metricas = resultado["metricas"]
            reporte = (
                f"{'='*80}\n"
                f"INFORME FINANCIERO\n"
                f"{'='*80}\n"
                f"Fecha del análisis: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"Ratio de Ahorro: {metricas['ratio_ahorro']:.1f}%\n"
                f"Ratio Gasto Esencial: {metricas['ratio_gasto_essencial']:.1f}%\n"
                f"Estabilidad Ingresos: {metricas['estabilidad_ingresos']:.1f}%\n"
                f"Ingreso Promedio: ${metricas['ingreso_promedio']:.2f}\n"
                f"Ahorro Promedio: ${metricas['ahorro_promedio']:.2f}\n\n"
                f"Recomendación: {resultado['categoria']} ({resultado['valor']:.1f})\n\n"
                f"{resultado['explicacion']}\n"
                f"{'='*80}\n"
            )
            self.texto_resultados.delete(1.0, tk.END)
            self.texto_resultados.insert(1.0, reporte)
            self._actualizar_estado_gui(resultado)

    # Actualizar estado en GUI
    def _actualizar_estado_gui(self, resultado):
        if hasattr(self, "pb_estado"):
            self.lbl_estado.config(
                text=f"Perfil {resultado['categoria']} · {resultado['valor']:.1f}"
            )
            self.pb_estado["value"] = resultado["valor"]

    # Mostrar gráficas en GUI
    def mostrar_graficas(self):
        """Dibuja gráficas en ventana nueva (solo GUI)."""
        if self.root is None:
            print("Graficas disponibles solo en modo GUI.")
            return
        ventana_graficas = tk.Toplevel(self.root)
        ventana_graficas.title("Análisis Gráfico - Sistema Difuso")
        ventana_graficas.geometry("1000x800")

        notebook = ttk.Notebook(ventana_graficas)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        frame_membresia = ttk.Frame(notebook)
        notebook.add(frame_membresia, text="Funciones de Membresía")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        self.ratio_ahorro.view(ax=axes[0, 0])
        axes[0, 0].set_title("Ratio de Ahorro (%)")
        self.ratio_gasto_essencial.view(ax=axes[0, 1])
        axes[0, 1].set_title("Ratio Gasto Esencial (%)")
        self.estabilidad_ingresos.view(ax=axes[1, 0])
        axes[1, 0].set_title("Estabilidad de Ingresos (%)")
        self.recomendacion.view(ax=axes[1, 1])
        axes[1, 1].set_title("Recomendación")

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, frame_membresia)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        frame_datos = ttk.Frame(notebook)
        notebook.add(frame_datos, text="Datos Financieros")
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

        datos = self.datos_financieros
        if datos is None or datos.empty:
            axes2[0, 0].text(0.5, 0.5, "No hay datos para graficar", ha="center")
        else:
            axes2[0, 0].plot(datos["fecha"], datos["ingreso"], label="Ingresos")
            axes2[0, 0].plot(
                datos["fecha"], datos["gasto_essencial"], label="Gastos Esenciales"
            )
            axes2[0, 0].plot(
                datos["fecha"],
                datos["gasto_no_essencial"],
                label="Gastos No Esenciales",
            )
            axes2[0, 0].legend()
            axes2[0, 0].tick_params(axis="x", rotation=45)

            axes2[0, 1].plot(datos["fecha"], datos["ahorro"], color="green")
            axes2[0, 1].tick_params(axis="x", rotation=45)

            ratios_ahorro = (datos["ahorro"] / (datos["ingreso"] + 1e-9)) * 100
            ratios_essencial = (
                datos["gasto_essencial"] / (datos["ingreso"] + 1e-9)
            ) * 100
            axes2[1, 0].plot(
                datos["fecha"], ratios_ahorro, label="Ratio Ahorro", color="blue"
            )
            axes2[1, 0].plot(
                datos["fecha"],
                ratios_essencial,
                label="Ratio Gasto Essencial",
                color="red",
            )
            axes2[1, 0].legend()
            axes2[1, 0].tick_params(axis="x", rotation=45)

            axes2[1, 1].hist(datos["ingreso"], bins=15, alpha=0.7, color="orange")

        plt.tight_layout()
        canvas2 = FigureCanvasTkAgg(fig2, frame_datos)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill="both", expand=True)

    # Modo consola
    def correr_en_consola(self):
        """Imprime un resumen del análisis en consola (modo headless)."""
        resultado = self.obtener_recomendacion()
        metricas = resultado["metricas"]
        print("=" * 80)
        print("INFORME FINANCIERO (modo consola)")
        print("=" * 80)
        print(f"Fecha del análisis: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"Ratio de Ahorro: {metricas['ratio_ahorro']:.1f}%")
        print(f"Ratio Gasto Esencial: {metricas['ratio_gasto_essencial']:.1f}%")
        print(f"Estabilidad Ingresos: {metricas['estabilidad_ingresos']:.1f}%")
        print(f"Ingreso Promedio: ${metricas['ingreso_promedio']:.2f}")
        print(f"Ahorro Promedio: ${metricas['ahorro_promedio']:.2f}")
        print(f"Recomendación: {resultado['categoria']} ({resultado['valor']:.1f})")
        print(resultado["explicacion"])
        print("=" * 80)
        print(
            f"Componentes -> Difuso: {resultado['valor_difuso']:.1f} | Neuronal: {resultado['valor_neuronal']:.1f}"
        )
        print("=" * 80)

    def _mostrar_info_ratio(self):
        mensaje = (
            "Ratio de Ahorro = (Ahorro promedio / Ingreso promedio) * 100.\n"
            "Mientras mayor sea, más margen tienes para invertir o cubrir emergencias."
        )
        (
            messagebox.showinfo("¿Qué es el Ratio de Ahorro?", mensaje)
            if messagebox
            else print(mensaje)
        )

    def _mostrar_info_gasto(self):
        mensaje = (
            "Gasto Esencial % refleja qué parte de tu ingreso se va a necesidades básicas.\n"
            "Si supera ~60%, conviene revisar gastos para liberar ahorro."
        )
        (
            messagebox.showinfo("¿Qué es el Gasto Esencial?", mensaje)
            if messagebox
            else print(mensaje)
        )

    def _mostrar_info_estabilidad(self):
        mensaje = (
            "Estabilidad de Ingresos compara la variación de tus ingresos recientes.\n"
            "Un valor cercano a 100% indica pagos constantes; bajo implica volatilidad."
        )
        (
            messagebox.showinfo("¿Qué es la Estabilidad de Ingresos?", mensaje)
            if messagebox
            else print(mensaje)
        )

    def _mostrar_info_perfil(self):
        if self.ultimo_resultado:
            res = self.ultimo_resultado
            mensaje = (
                f"Perfil {res['categoria']} ({res['valor']:.1f}).\n"
                f"· Difuso: {res['valor_difuso']:.1f}\n"
                f"· Neuronal: {res['valor_neuronal']:.1f}\n\n"
                f"{res['explicacion']}"
            )
        else:
            mensaje = "Ejecuta el análisis para conocer tu perfil financiero."
        (
            messagebox.showinfo("Detalle del Perfil", mensaje)
            if messagebox
            else print(mensaje)
        )
