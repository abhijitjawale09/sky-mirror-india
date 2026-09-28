import os
from climate_twin import create_app
from climate_twin.training.model_comparison import calculate_metrics


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5010))
    app.run(host="0.0.0.0", port=port, debug=False)

