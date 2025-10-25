"""Repositorio de datos - Implementación con pandas"""

import pandas as pd
from typing import List, Optional
from datetime import datetime

from ports.output_ports import IDatosRepository
from domain.entities import DatosHipoteca


class DatosRepository(IDatosRepository):
    """
    Implementación del repositorio de datos con pandas
    """

    def __init__(self):
        self.df_cache = None

    def cargar_datos(self, ruta: str) -> pd.DataFrame:
        """Carga datos desde un archivo"""
        try:
            if ruta.endswith(".xlsx") or ruta.endswith(".xls"):
                df = pd.read_excel(ruta)
            elif ruta.endswith(".csv"):
                df = pd.read_csv(ruta)
            else:
                raise ValueError("Formato de archivo no soportado")

            # Procesar fechas
            if "fecha" in df.columns:
                df["fecha"] = pd.to_datetime(df["fecha"])
                df.set_index("fecha", inplace=True)
            else:
                df.index = pd.date_range(start="2025-01-01", periods=len(df), freq="M")

            # Crear gastos_fijos si no existe
            if "gastos_fijos" not in df.columns:
                if "intereses" in df.columns and "seguros" in df.columns:
                    df["gastos_fijos"] = df["intereses"] + df["seguros"]
                else:
                    df["gastos_fijos"] = 0

            self.df_cache = df
            return df

        except Exception as e:
            print(f"Error cargando datos: {e}")
            return pd.DataFrame()

    def obtener_datos_historicos(
        self,
        fecha_inicio: Optional[datetime] = None,
        fecha_fin: Optional[datetime] = None,
    ) -> List[DatosHipoteca]:
        """Obtiene datos históricos"""
        if self.df_cache is None:
            return []

        df = self.df_cache

        # Filtrar por fecha si se proporciona
        if fecha_inicio:
            df = df[df.index >= fecha_inicio]
        if fecha_fin:
            df = df[df.index <= fecha_fin]

        # Convertir a entidades
        datos = []
        for fecha, row in df.iterrows():
            dato = DatosHipoteca(
                fecha=fecha,
                capital=row.get("capital", 0),
                gastos_fijos=row.get("gastos_fijos", 0),
                total_mensual=row.get("total_mensual", 0),
                tasa_uvr=row.get("tasa_uvr"),
                tasa_dtf=row.get("tasa_dtf"),
                inflacion_ipc=row.get("inflacion_ipc"),
                tipo_pago=row.get("tipo_pago", "Ordinario"),
            )
            datos.append(dato)

        return datos

    def guardar_datos(self, datos: List[DatosHipoteca]) -> bool:
        """Guarda datos históricos"""
        # Por implementar si se necesita
        return True
