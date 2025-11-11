import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

URL = "https://archive.ics.uci.edu/static/public/186/data.csv"


def cargar_dataset(url: str) -> pd.DataFrame:
    # Leer con ';' y fallback a coma si fuera necesario
    df = pd.read_csv(url, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(url)
    return df


def main():
    # 1. Cargar
    df = cargar_dataset(URL)

    # 2. Subconjunto quality 5 y 6
    df_sub = df[df["quality"].isin([5, 6])].copy()

    # 3. Definir X (columnas 0-10), y = quality
    X = df_sub.iloc[:, 0:11]  # primeras 11 columnas (hasta 'alcohol')
    y = df_sub["quality"]

    # Split reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=10, stratify=y
    )

    # 4. Pipeline: escalado + MLP
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(10, 10),
                    activation="logistic",
                    random_state=10,
                    max_iter=500,
                ),
            ),
        ]
    )

    # 5. Entrenar
    pipeline.fit(X_train, y_train)

    # 6. Evaluar
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acc_redondeado = round(acc, 2)

    # Opciones
    opciones = {0.90: "a", 0.80: "b", 1.00: "c", 0.70: "d"}
    letra = opciones.get(acc_redondeado, "N/A")

    # Resultados
    print(f"Tamaño subconjunto: {df_sub.shape}")
    print(f"Accuracy (test): {acc:.4f} (≈ {acc_redondeado:.2f})")
    print(f"Respuesta seleccionada: {letra}")


if __name__ == "__main__":
    main()
