from climate_twin import create_app
from climate_twin.training.model_comparison import calculate_metrics


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=5010)
