(() => {
  const initialState = window.__INITIAL_STATE__ || {};
  const mapElement = document.getElementById("map");
  const forecastChartCanvas = document.getElementById("forecastChart");

  const regionName = document.getElementById("region-name");
  const snapshotDate = document.getElementById("snapshot-date");
  const predictedRainfall = document.getElementById("predicted-rainfall");
  const predictedTmax = document.getElementById("predicted-tmax");
  const monsoonMetric = document.getElementById("metric-monsoon");
  const heatMetric = document.getElementById("metric-heat");
  const stabilityMetric = document.getElementById("metric-stability");
  const rainfallMae = document.getElementById("mae-rainfall");
  const tmaxMae = document.getElementById("mae-tmax");
  const tminMae = document.getElementById("mae-tmin");

  const rainfallSlider = document.getElementById("rainfall-slider");
  const tempSlider = document.getElementById("temp-slider");
  const horizonSlider = document.getElementById("horizon-slider");
  const rainfallValue = document.getElementById("rainfall-value");
  const tempValue = document.getElementById("temp-value");
  const horizonValue = document.getElementById("horizon-value");

  let selectedRegion = initialState.region || "Kerala Coast";
  let forecastChart = null;
  let map = null;
  let regionLayer = null;

  const palette = {
    rain: "#64f4d8",
    temp: "#77a8ff",
    min: "#ffbf75",
    text: "rgba(232, 242, 255, 0.9)",
    grid: "rgba(255, 255, 255, 0.1)",
  };

  function bootstrap() {
    bindControls();
    renderMap(initialState.pilot_regions || []);
    renderChart(initialState.forecast || {});
    attachRegionButtons();
  }

  function bindControls() {
    const syncLabels = () => {
      rainfallValue.textContent = rainfallSlider.value;
      tempValue.textContent = Number.parseFloat(tempSlider.value).toFixed(1);
      horizonValue.textContent = horizonSlider.value;
    };

    const handleInput = () => {
      syncLabels();
    };
    
    const handleChange = () => {
      syncLabels();
      fetchPrediction();
    };

    [rainfallSlider, tempSlider, horizonSlider].forEach((input) => {
      input.addEventListener("input", handleInput);
      input.addEventListener("change", handleChange);
    });
    
    syncLabels();
  }

  function attachRegionButtons() {
    document.querySelectorAll("[data-region-switch]").forEach((button) => {
      button.addEventListener("click", () => {
        selectedRegion = button.dataset.regionSwitch;
        fetchPrediction();
      });
    });
  }

  async function fetchPrediction() {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        region: selectedRegion,
        rainfall_delta_pct: Number(rainfallSlider.value),
        temp_delta_c: Number(tempSlider.value),
        horizon_days: Number(horizonSlider.value),
      }),
    });

    const payload = await response.json();
    updateDashboard(payload);
  }

  function updateDashboard(payload) {
    regionName.textContent = payload.region;
    snapshotDate.textContent = payload.snapshot.date;
    predictedRainfall.textContent = `${payload.metrics.predicted_rainfall} mm`;
    predictedTmax.textContent = `${payload.metrics.predicted_tmax} C`;
    monsoonMetric.textContent = payload.metrics.monsoon_pulse;
    heatMetric.textContent = payload.metrics.heat_stress;
    stabilityMetric.textContent = payload.metrics.stability;
    rainfallMae.textContent = formatMetric(payload.model_metrics.mae_rainfall_mm);
    tmaxMae.textContent = formatMetric(payload.model_metrics.mae_tmax_c);
    tminMae.textContent = formatMetric(payload.model_metrics.mae_tmin_c);

    renderMap(payload.pilot_regions, payload.region, payload.snapshot, payload.metrics);
    renderChart(payload.forecast);
  }

  function formatMetric(value) {
    if (value === undefined || value === null) {
      return "0.00";
    }
    return Number(value).toFixed(2);
  }

  function renderMap(regions, activeRegion = selectedRegion, snapshot = initialState.snapshot, metrics = initialState.metrics) {
    const center = [22.5, 80.5];
    if (!map) {
      map = L.map(mapElement, { zoomControl: false }).setView(center, 5);
      L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
        attribution: '&copy; OpenStreetMap &copy; CARTO',
      }).addTo(map);
      L.control.zoom({ position: "topright" }).addTo(map);
    }

    if (regionLayer) {
      regionLayer.remove();
    }

    regionLayer = L.layerGroup();
    regions.forEach((region, index) => {
      const intensity = activeRegion === region.name ? 1.45 : 0.75 + index * 0.03;
      const marker = L.circleMarker([region.latitude, region.longitude], {
        radius: 8 + intensity * 4,
        color: intensity > 1 ? "#64f4d8" : "#77a8ff",
        fillColor: intensity > 1 ? "#64f4d8" : "#77a8ff",
        fillOpacity: 0.28,
        weight: 1,
      }).bindTooltip(`<strong>${region.name}</strong><br/>Intensity ${intensity.toFixed(2)}`);
      marker.addTo(regionLayer);
    });

    regionLayer.addTo(map);

    if (snapshot && metrics) {
      L.popup({ closeButton: false, autoClose: true })
        .setLatLng([snapshot.latitude, snapshot.longitude])
        .setContent(`<div style="font-family: Space Grotesk, sans-serif; color: #07111f;"><strong>${activeRegion}</strong><br/>Rainfall ${metrics.predicted_rainfall} mm<br/>Tmax ${metrics.predicted_tmax} C</div>`)
        .openOn(map);
    }
  }

  function renderChart(forecast) {
    const labels = forecast.labels || [];
    const rainfall = forecast.rainfall || [];
    const tmax = forecast.tmax || [];
    const tmin = forecast.tmin || [];

    const context = forecastChartCanvas.getContext("2d");
    if (!forecastChart) {
      forecastChart = new Chart(context, {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: "Rainfall (mm)",
              data: rainfall,
              borderColor: palette.rain,
              backgroundColor: "rgba(100, 244, 216, 0.18)",
              fill: true,
              tension: 0.34,
              yAxisID: 'y1',
            },
            {
              label: "Max Temp (C)",
              data: tmax,
              borderColor: palette.temp,
              backgroundColor: "rgba(119, 168, 255, 0.1)",
              fill: false,
              tension: 0.34,
            },
            {
              label: "Min Temp (C)",
              data: tmin,
              borderColor: palette.min,
              backgroundColor: "rgba(255, 191, 117, 0.1)",
              fill: false,
              tension: 0.34,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              labels: { color: palette.text, usePointStyle: true, pointStyle: "line" },
              position: 'top',
            },
          },
          scales: {
            x: {
              ticks: { color: palette.text },
              grid: { color: palette.grid },
            },
            y: {
              position: 'left',
              ticks: { color: palette.text },
              grid: { color: palette.grid },
            },
            y1: {
              position: 'right',
              ticks: { color: palette.text },
              grid: { display: false },
            },
          },
        },
      });
      return;
    }

    forecastChart.data.labels = labels;
    forecastChart.data.datasets[0].data = rainfall;
    forecastChart.data.datasets[1].data = tmax;
    forecastChart.data.datasets[2].data = tmin;
    forecastChart.update();
  }

  document.addEventListener("DOMContentLoaded", bootstrap);
})();
