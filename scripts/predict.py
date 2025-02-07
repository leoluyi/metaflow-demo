from src.pipelines.predict import predict
from src.utils.config import load_config

if __name__ == "__main__":
    data_config = load_config("configs/base/data.ini")

    predict(
        input_path=data_config["data"]["predict_path"],
        model_path=data_config["paths"]["model"],
        output_path=data_config["paths"]["predictions"],
    )
