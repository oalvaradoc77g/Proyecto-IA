"""Repositorio de modelos - Implementación con persistencia en JSON"""

import json
import os
from typing import List, Optional
from datetime import datetime

from ports.output_ports import IModeloRepository
from domain.entities import Modelo


class ModeloRepository(IModeloRepository):
    """
    Implementación del repositorio de modelos con persistencia en archivos JSON
    """

    def __init__(self, data_dir: str = "data/models"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.index_file = os.path.join(data_dir, "index.json")
        self._inicializar_index()

    def _inicializar_index(self):
        """Inicializa el archivo índice si no existe"""
        if not os.path.exists(self.index_file):
            with open(self.index_file, "w") as f:
                json.dump({"modelos": [], "activo": None}, f)

    def _leer_index(self) -> dict:
        """Lee el archivo índice"""
        with open(self.index_file, "r") as f:
            return json.load(f)

    def _escribir_index(self, index: dict):
        """Escribe el archivo índice"""
        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=4)

    def guardar(self, modelo: Modelo) -> bool:
        """Guarda un modelo"""
        try:
            # Guardar archivo del modelo
            modelo_file = os.path.join(self.data_dir, f"{modelo.id}.json")
            with open(modelo_file, "w") as f:
                json.dump(modelo.to_dict(), f, indent=4)

            # Actualizar índice
            index = self._leer_index()
            if modelo.id not in index["modelos"]:
                index["modelos"].append(modelo.id)

            if modelo.activo:
                index["activo"] = modelo.id

            self._escribir_index(index)
            return True

        except Exception as e:
            print(f"Error guardando modelo: {e}")
            return False

    def obtener_por_id(self, modelo_id: str) -> Optional[Modelo]:
        """Obtiene un modelo por ID"""
        try:
            modelo_file = os.path.join(self.data_dir, f"{modelo_id}.json")
            if not os.path.exists(modelo_file):
                return None

            with open(modelo_file, "r") as f:
                data = json.load(f)

            return Modelo(
                id=data["id"],
                tipo=data["tipo"],
                fecha_entrenamiento=datetime.fromisoformat(data["fecha_entrenamiento"]),
                metricas=data["metricas"],
                parametros=data["parametros"],
                version=data["version"],
                activo=data["activo"],
            )

        except Exception as e:
            print(f"Error obteniendo modelo: {e}")
            return None

    def obtener_activo(self) -> Optional[Modelo]:
        """Obtiene el modelo activo"""
        try:
            index = self._leer_index()
            modelo_id = index.get("activo")

            if modelo_id:
                return self.obtener_por_id(modelo_id)
            return None

        except Exception as e:
            print(f"Error obteniendo modelo activo: {e}")
            return None

    def listar_todos(self) -> List[Modelo]:
        """Lista todos los modelos"""
        try:
            index = self._leer_index()
            modelos = []

            for modelo_id in index["modelos"]:
                modelo = self.obtener_por_id(modelo_id)
                if modelo:
                    modelos.append(modelo)

            return modelos

        except Exception as e:
            print(f"Error listando modelos: {e}")
            return []

    def actualizar(self, modelo: Modelo) -> bool:
        """Actualiza un modelo existente"""
        return self.guardar(modelo)
