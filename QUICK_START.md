# 🚀 Guía de Inicio Rápido - Arquitectura Hexagonal

## 📦 Instalación

1. **Activar entorno virtual:**

```powershell
.\.venv\Scripts\Activate
```

2. **Instalar dependencias (si no están instaladas):**

```powershell
pip install -r requirements.txt
```

## 🎯 Ejecutar la Aplicación

### Opción 1: Usando el nuevo main (Arquitectura Hexagonal)

```powershell
python src/main.py
```

### Opción 2: Usando el código legacy

```powershell
python src/Proyectos/prediccion_hipoteca.py
```

## 📋 Flujo de Uso

### 1️⃣ Entrenar Modelo

```
🏠 SISTEMA DE PREDICCIÓN DE CUOTAS HIPOTECARIAS
1. Entrenar modelo
👉 Seleccione una opción: 1

📁 Ingrese la ruta del archivo de datos: C:\path\to\datos.xlsx
```

### 2️⃣ Realizar Predicciones

```
👉 Seleccione una opción: 2

📅 Número de meses a predecir (default: 6): 6
📊 ¿Incluir intervalos de confianza? (s/n, default: s): s
```

### 3️⃣ Ver Modelo Activo

```
👉 Seleccione una opción: 3
```

## 🗂️ Estructura del Proyecto

```
src/
├── main.py                    # 🚀 PUNTO DE ENTRADA
├── domain/                    # 💎 Lógica de negocio
├── application/               # 🔧 Casos de uso
├── infrastructure/            # 🏗️ Implementaciones
├── ports/                     # 🔌 Interfaces
└── presentation/              # 🎯 CLI
```

## 📊 Datos de Ejemplo

Tu archivo Excel/CSV debe tener estas columnas:

```
fecha | capital | intereses | seguros | total_mensual | tipo_pago
```

O alternativamente:

```
fecha | capital | gastos_fijos | total_mensual | tipo_pago
```

## 🔍 Verificar Instalación

```powershell
# Verificar Python
python --version

# Verificar paquetes
pip list | findstr "pandas numpy scikit-learn"

# Verificar estructura
tree /F src
```

## ⚠️ Solución de Problemas

### Error: "No module named 'domain'"

```powershell
# Ejecutar desde la raíz del proyecto
cd "C:\Users\omaroalvaradoc\Documents\Personal\Proyectos\CURSO IA"
python src/main.py
```

### Error: "No se pueden cargar los datos"

Verifica que tu archivo tenga:

- Formato correcto (.xlsx o .csv)
- Columnas requeridas
- Datos numéricos válidos

### Error: "No hay modelo activo"

Primero entrena un modelo (Opción 1 del menú)

## 📝 Ejemplos Rápidos

### Entrenar y Predecir en un Flujo

```powershell
# 1. Ejecutar aplicación
python src/main.py

# 2. Entrenar modelo (Opción 1)
# Ingresar ruta de datos

# 3. Hacer predicciones (Opción 2)
# Configurar parámetros

# 4. Ver resultados en data/predictions/
```

## 🎓 Próximos Pasos

1. ✅ Familiarizarse con el menú CLI
2. ✅ Entrenar tu primer modelo
3. ✅ Generar predicciones
4. ✅ Explorar el código en `src/domain/`
5. ✅ Leer `ARQUITECTURA_HEXAGONAL.md`

## 📞 Referencias

- **Arquitectura**: Ver `ARQUITECTURA_HEXAGONAL.md`
- **Código Legacy**: `src/Proyectos/prediccion_hipoteca.py`
- **Datos**: `data/raw/`
- **Resultados**: `data/predictions/`

---

💡 **Tip**: Usa `Ctrl+C` para salir de la aplicación en cualquier momento
