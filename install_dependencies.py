"""
Script para instalar todas las dependencias necesarias
"""

import subprocess
import sys
import os

def install_package(package):
    """Instala un paquete usando pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} instalado correctamente")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Error instalando {package}")
        return False

def main():
    print("📦 Iniciando instalación de dependencias...\n")
    
    # Leer requirements.txt
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    if not os.path.exists(requirements_path):
        print("❌ No se encontró el archivo requirements.txt")
        return
    
    with open(requirements_path, 'r') as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"Paquetes a instalar: {len(packages)}\n")
    
    # Instalar cada paquete
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
        print()
    
    print("="*60)
    print(f"✅ Instalación completada: {success_count}/{len(packages)} paquetes")
    print("="*60)
    
    if success_count == len(packages):
        print("\n🎉 ¡Todas las dependencias instaladas correctamente!")
        print("Ahora puedes ejecutar: python src/main.py")
    else:
        print("\n⚠️ Algunas dependencias no se instalaron correctamente")

if __name__ == "__main__":
    main()
