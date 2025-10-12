# 🚀 Guía Rápida - Proyecto Reorganizado

## 📂 Estructura Final

```
CURSO IA/
├── 📄 README.md              → Documentación principal
├── 📄 REORGANIZACION.md      → Detalles de cambios realizados
├── 📄 requirements.txt       → Dependencias
├── 📄 .gitignore            → Configuración Git
│
├── 📁 data/                  → TODOS LOS DATOS AQUÍ
│   ├── raw/                 → Datos originales
│   ├── predictions/         → Resultados de predicciones
│   └── *.pkl                → Modelos entrenados
│
├── 📁 src/                   → CÓDIGO DEL PROYECTO
│   ├── 🐍 main.py           → ⭐ SCRIPT PRINCIPAL
│   ├── core/                → Modelos ML
│   ├── services/            → APIs externas
│   ├── utils/               → Utilidades
│   └── examples/            → Ejemplos
│
├── 📁 notebooks/             → Jupyter notebooks
│   └── experiments/
│
└── 📁 ejercicios/            → Ejercicios del curso (separados)
```

## ⚡ Comandos Esenciales

### Ejecutar Análisis Principal

```powershell
# Método recomendado
& ".\.venv\Scripts\python.exe" src/main.py
```

### Activar Entorno Virtual

```powershell
.\.venv\Scripts\Activate
```

### Instalar Dependencias

```powershell
pip install -r requirements.txt
```

## 📍 Ubicaciones Importantes

| ¿Qué busco?      | ¿Dónde está?                                 |
| ---------------- | -------------------------------------------- |
| Datos CSV        | `data/raw/Datos Movimientos Financieros.csv` |
| Script principal | `src/main.py`                                |
| Modelos ML       | `src/core/modelo_hibrido.py`                 |
| Datos externos   | `src/services/external_data_service.py`      |
| Predicciones     | `data/predictions/`                          |
| Ejercicios curso | `ejercicios/`                                |

## 🔄 Cambios Clave vs Versión Anterior

| Antes                                          | Ahora                     |
| ---------------------------------------------- | ------------------------- |
| `src/Proyectos/prediccion_hipoteca.py`         | `src/main.py`             |
| `src/data/Datos...csv`                         | `data/raw/Datos...csv`    |
| `src/Ejercicios/`                              | `ejercicios/`             |
| Código duplicado en `external_data_service.py` | ✅ Corregido              |
| Sin `.gitignore`                               | ✅ Agregado               |
| README genérico                                | ✅ Documentación completa |

## 💡 Consejos

1. **Siempre usar el entorno virtual**: `.venv`
2. **Datos sensibles**: Agregar a `.gitignore` si es necesario
3. **Commit de cambios**: Recuerda versionar tu trabajo
4. **Documentar**: Actualizar README si agregas funcionalidades

## 🎯 Lo Que Funciona Ahora

✅ Análisis de tendencias financieras  
✅ Categorización automática de gastos  
✅ Sugerencias de ahorro  
✅ Análisis de gastos hormiga  
✅ Visualizaciones con matplotlib  
✅ Integración con APIs del Banco de la República  
✅ Modelos de predicción híbridos (Ridge + ARIMA)

## 📞 Referencias

- **README.md**: Documentación completa
- **REORGANIZACION.md**: Detalles de cambios
- **requirements.txt**: Lista de dependencias

---

**Proyecto listo para usar** ✨  
**Última reorganización**: Octubre 12, 2025
