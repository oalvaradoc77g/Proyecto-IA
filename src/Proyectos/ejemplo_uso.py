from modelo_hibrido import ModeloHibrido
import pandas as pd

def main():
    # Cargar datos base (tu Excel con gastos)
    df_base = pd.read_excel('hipoteca_extractos_ene_sep_2025.xlsx')
    
    # Crear y preparar modelo
    modelo = ModeloHibrido(orden_arima_auto=True)
    
    # Cargar y combinar con datos macroeconómicos
    df = modelo.cargar_y_preparar_datos(
        df_base,
        fecha_inicio="2025-01-01",
        fecha_fin="2025-12-31"
    )
    
    if df is not None:
        if modelo.entrenar(df):
            # Continuar con predicciones...
            predicciones = modelo.predecir_futuro(n_predicciones=6)
            print("\n🔮 PREDICCIONES FUTURAS:")
            print(predicciones)

if __name__ == "__main__":
    main()
