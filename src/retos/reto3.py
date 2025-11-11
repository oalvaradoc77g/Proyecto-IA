import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

URL = "https://archive.ics.uci.edu/static/public/186/data.csv"


def cargar_dataset(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(url)
    except Exception:
        df = pd.read_csv(url)
    return df


def main():
    # 1. Cargar
    df = cargar_dataset(URL)

    # 2. Subconjunto quality 5 y 6
    df_sub = df[df["quality"].isin([5, 6])].copy()

    # 3. Definir X (columnas 0-9) y 4. y='alcohol'
    X = df_sub.iloc[:, 0:10]
    y = df_sub["alcohol"]

    # Split reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=10
    )

    # 5. Pipeline: escalado + MLPRegressor
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(100,),
                    activation="tanh",
                    random_state=10,
                    max_iter=1000,
                ),
            ),
        ]
    )

    pipe.fit(X_train, y_train)

    # 6. Predicción y MSE
    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_round = round(mse, 2)

    # Opciones
    opciones = {0.87: "a", 0.57: "b", 0.67: "c", 0.77: "d"}
    letra = opciones.get(mse_round, "N/A")

    print(f"Tamaño subconjunto: {df_sub.shape}")
    print(f"MSE (test): {mse:.4f} (≈ {mse_round:.2f})")
    print(f"Respuesta seleccionada: {letra}")


if __name__ == "__main__":
    main()
