import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/186/data.csv"


def cargar_datos(url: str) -> pd.DataFrame:
    # Intentar leer con separador ';' (formato típico del dataset), si falla usar coma
    try:
        df = pd.read_csv(url, sep=";")
        # Si solo leyó 1 columna, reintentar con coma
        if df.shape[1] == 1:
            df = pd.read_csv(url)
    except Exception:
        df = pd.read_csv(url)
    return df


def main():
    df = cargar_datos(URL)
    # Subconjunto con quality en 5, 6, 7
    df_sub = df[df["quality"].isin([5, 6, 7])].copy()
    shape = df_sub.shape
    print("Tamaño del nuevo dataframe:", shape)

    # Mapear a la opción correcta
    opciones = {
        "a": (6053, 13),
        "b": (4000, 13),
        "c": (6497, 13),
        "d": (4974, 13),
    }
    letra = next((k for k, v in opciones.items() if v == shape), "N/A")
    print(f"Respuesta: {letra}")


if __name__ == "__main__":
    main()
