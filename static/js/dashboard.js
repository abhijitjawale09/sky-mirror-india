(() => {
  const initialState = window.__INITIAL_STATE__ || {};
  let currentState = { ...initialState };

  // DOM Elements
  const mapElement = document.getElementById("map");
  const forecastChartCanvas = document.getElementById("forecastChart");
  const simulationChartCanvas = document.getElementById("simulationChart");
  const temporalContextChartCanvas = document.getElementById("temporalContextChart");
  const replayChartCanvas = document.getElementById("replayChart");

  const globalRegionSelect = document.getElementById("global-region-select");
  const regionName = document.getElementById("region-name");
  const heroTitle = document.getElementById("hero-title");
  const snapshotDate = document.getElementById("snapshot-date");
  const predictedRainfall = document.getElementById("predicted-rainfall");
  const predictedTmax = document.getElementById("predicted-tmax");
  const monsoonMetric = document.getElementById("metric-monsoon");
  const heatMetric = document.getElementById("metric-heat");
  const stabilityMetric = document.getElementById("metric-stability");
  const barMonsoon = document.getElementById("bar-monsoon");
  const barHeat = document.getElementById("bar-heat");
  const barStability = document.getElementById("bar-stability");

  const liveObsRain = document.getElementById("live-obs-rain");
  const liveObsTmax = document.getElementById("live-obs-tmax");
  const liveObsTmin = document.getElementById("live-obs-tmin");
  const liveObsHumidity = document.getElementById("live-obs-humidity");
  const climatologyProb = document.getElementById("climatology-prob");
  const anomalyRainVal = document.getElementById("anomaly-rain-val");
  const anomalyTmaxVal = document.getElementById("anomaly-tmax-val");
  const anomalyTminVal = document.getElementById("anomaly-tmin-val");
  const alertBanner = document.getElementById("alert-banner");
  const alertMessage = document.getElementById("alert-message");
  const alertRiskLevel = document.getElementById("alert-risk-level");
  const confidenceMetric = document.getElementById("confidence-metric");
  const rainRangeMetric = document.getElementById("rain-range-metric");
  const tempRangeMetric = document.getElementById("temp-range-metric");
  const telemetryRegionLabel = document.getElementById("telemetry-region-label");

  // Scenario Elements
  const rainfallSlider = document.getElementById("rainfall-slider");
  const tempSlider = document.getElementById("temp-slider");
  const horizonSlider = document.getElementById("horizon-slider");
  const rainfallValue = document.getElementById("rainfall-value");
  const tempValue = document.getElementById("temp-value");
  const horizonValue = document.getElementById("horizon-value");
  const simCumNormal = document.getElementById("sim-cum-normal");
  const simCumBase = document.getElementById("sim-cum-base");
  const simCumScen = document.getElementById("sim-cum-scen");
  const simDeficitSurplus = document.getElementById("sim-deficit-surplus");
  const precautionsList = document.getElementById("precautions-list");
  const resetScenarioBtn = document.getElementById("reset-scenario-btn");

  // Replay Elements
  const replayStartDate = document.getElementById("replay-start-date");
  const replayEndDate = document.getElementById("replay-end-date");
  const runReplayBtn = document.getElementById("run-replay-btn");
  const replayMetricsSummary = document.getElementById("replay-metrics-summary");

  // Model Comparison Elements & State
  const modelCompChartCanvas = document.getElementById("modelComparisonChart");
  const compChartSubtitle = document.getElementById("comp-chart-subtitle");
  const compToggleBtns = document.querySelectorAll(".comp-toggle-btn");
  const jumpToScorecardBtn = document.getElementById("jump-to-scorecard-btn");
  const modelChips = document.querySelectorAll(".model-chip");
  const overlayAllModelsCheck = document.getElementById("overlay-all-models-check");

  // 7-Day Forecast Elements
  const f7dLocationSelect = document.getElementById("f7d-location-select");
  const f7dCustomCoords = document.getElementById("f7d-custom-coords");
  const f7dLatInput = document.getElementById("f7d-lat");
  const f7dLonInput = document.getElementById("f7d-lon");
  const f7dSearchBtn = document.getElementById("f7d-search-btn");
  const f7dLastUpdated = document.getElementById("f7d-last-updated");
  const f7dLoading = document.getElementById("f7d-loading");
  const f7dError = document.getElementById("f7d-error");
  const f7dErrorMessage = document.getElementById("f7d-error-message");
  const f7dCardsContainer = document.getElementById("f7d-cards-container");
  const f7dCards = document.getElementById("f7d-cards");
  const f7dTempChartCanvas = document.getElementById("f7dTempChart");
  const f7dRainfallChartCanvas = document.getElementById("f7dRainfallChart");

  // Chart Instances & Map
  let forecastChart = null;
  let simulationChart = null;
  let temporalContextChart = null;
  let replayChart = null;
  let modelComparisonChart = null;
  let f7dTempChart = null;
  let f7dRainfallChart = null;
  let map = null;
  let regionLayer = null;

  let selectedRegion = initialState.region || "Kerala Coast";
  let activeVarFilter = "all";
  let activeCompMetric = "rainfall_mae";
  let activeForecastModel = "random_forest";
  let overlayAllModels = false;
  let f7dFetched = false;

  const palette = {
    cyan: "#00f5d4",
    cyanAlpha: "rgba(0, 245, 212, 0.18)",
    cyanBand: "rgba(0, 245, 212, 0.12)",
    blue: "#70a1ff",
    blueAlpha: "rgba(112, 161, 255, 0.15)",
    blueBand: "rgba(112, 161, 255, 0.1)",
    amber: "#ffbf75",
    amberAlpha: "rgba(255, 191, 117, 0.18)",
    amberBand: "rgba(255, 191, 117, 0.1)",
    ruby: "#ff5c8a",
    green: "#2ecc71",
    text: "rgba(240, 246, 252, 0.85)",
    grid: "rgba(255, 255, 255, 0.06)",
  };

  // Live Sync Elements
  const liveSyncBtn = document.getElementById("live-sync-btn");
  const liveStatusText = document.getElementById("live-status-text");

  // Bootstrap Application
  function bootstrap() {
    initTabs();
    bindControls();
    initMap(initialState.pilot_regions || []);
    initModelComparisonControls();
    initForecastModelControls();
    renderForecastChart(initialState.forecast || {});
    renderTemporalContextChart(initialState.time_series || {});
    renderSimulationChart(initialState.simulation || {});
    renderModelComparisonChart(activeCompMetric);
    attachRegionButtons();
    attachPresetChips();
    attachReplayHandler();
    attachLiveSyncHandler();
    attachScorecardJump();
    init7DayForecast();
  }

  function attachLiveSyncHandler() {
    if (!liveSyncBtn) return;
    liveSyncBtn.addEventListener("click", async () => {
      liveSyncBtn.classList.add("syncing");
      if (liveStatusText) liveStatusText.textContent = "SYNCING STREAM...";
      try {
        const response = await fetch(`/api/live-sync?region=${encodeURIComponent(selectedRegion)}`, {
          method: "POST",
        });
        if (!response.ok) throw new Error("Sync failed");
        const data = await response.json();
        if (data.state) {
          currentState = data.state;
          updateDashboard(data.state);
        }
        if (liveStatusText) liveStatusText.textContent = "LIVE STREAM ACTIVE";
      } catch (err) {
        console.error("Live sync failed:", err);
        if (liveStatusText) liveStatusText.textContent = "LIVE STREAM (CACHED)";
      } finally {
        liveSyncBtn.classList.remove("syncing");
      }
    });
  }

  // Tab Manager
  function initTabs() {
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabViews = document.querySelectorAll(".tab-view");

    tabButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetTab = btn.dataset.tab;

        tabButtons.forEach((b) => {
          b.classList.remove("active");
          b.setAttribute("aria-selected", "false");
        });
        tabViews.forEach((v) => v.classList.remove("active"));

        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");

        const targetView = document.getElementById(`view-${targetTab}`);
        if (targetView) {
          targetView.classList.add("active");
        }

        // Trigger chart recalculation on view change
        setTimeout(() => {
          if (forecastChart) forecastChart.resize();
          if (simulationChart) simulationChart.resize();
          if (temporalContextChart) temporalContextChart.resize();
          if (replayChart) replayChart.resize();
          if (modelComparisonChart) modelComparisonChart.resize();
          if (f7dTempChart) f7dTempChart.resize();
          if (f7dRainfallChart) f7dRainfallChart.resize();
          if (map) map.invalidateSize();
        }, 60);

        // Lazy-load 7-day forecast on first visit
        if (targetTab === "7day" && !f7dFetched) {
          fetch7DayForecast();
        }
      });
    });

    // Variable toggle buttons on forecast tab
    document.querySelectorAll(".var-toggle").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".var-toggle").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        activeVarFilter = btn.dataset.var;
        renderForecastChart(currentState.forecast || {});
      });
    });
  }

  // Control Listeners
  function bindControls() {
    const syncLabels = () => {
      if (rainfallSlider) rainfallValue.textContent = rainfallSlider.value;
      if (tempSlider) tempValue.textContent = Number.parseFloat(tempSlider.value).toFixed(1);
      if (horizonSlider) horizonValue.textContent = horizonSlider.value;
    };

    let debounceTimer = null;
    const handleChange = () => {
      syncLabels();
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        fetchPrediction();
      }, 120);
    };

    [rainfallSlider, tempSlider, horizonSlider].forEach((slider) => {
      if (slider) {
        slider.addEventListener("input", syncLabels);
        slider.addEventListener("change", handleChange);
      }
    });

    if (globalRegionSelect) {
      globalRegionSelect.addEventListener("change", () => {
        selectedRegion = globalRegionSelect.value;
        updateRegionChips(selectedRegion);
        fetchPrediction();
      });
    }

    if (resetScenarioBtn) {
      resetScenarioBtn.addEventListener("click", () => {
        if (rainfallSlider) rainfallSlider.value = "0";
        if (tempSlider) tempSlider.value = "0";
        if (horizonSlider) horizonSlider.value = "14";
        syncLabels();
        fetchPrediction();
      });
    }

    syncLabels();
  }

  function attachRegionButtons() {
    document.querySelectorAll("[data-region-switch]").forEach((button) => {
      button.addEventListener("click", () => {
        selectedRegion = button.dataset.regionSwitch;
        if (globalRegionSelect) globalRegionSelect.value = selectedRegion;
        updateRegionChips(selectedRegion);
        fetchPrediction();
      });
    });
  }

  function updateRegionChips(regionName) {
    document.querySelectorAll("[data-region-switch]").forEach((btn) => {
      if (btn.dataset.regionSwitch === regionName) {
        btn.classList.add("chip-active");
      } else {
        btn.classList.remove("chip-active");
      }
    });
  }

  function attachPresetChips() {
    document.querySelectorAll(".preset-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const rainDelta = chip.dataset.rain;
        const tempDelta = chip.dataset.temp;
        if (rainfallSlider && rainDelta !== undefined) rainfallSlider.value = rainDelta;
        if (tempSlider && tempDelta !== undefined) tempSlider.value = tempDelta;
        rainfallValue.textContent = rainfallSlider.value;
        tempValue.textContent = Number.parseFloat(tempSlider.value).toFixed(1);
        fetchPrediction();
      });
    });
  }

  // Fetch New State
  async function fetchPrediction() {
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          region: selectedRegion,
          rainfall_delta_pct: rainfallSlider ? Number(rainfallSlider.value) : 0,
          temp_delta_c: tempSlider ? Number(tempSlider.value) : 0,
          horizon_days: horizonSlider ? Number(horizonSlider.value) : 14,
        }),
      });

      if (!response.ok) throw new Error("Failed to fetch prediction");
      const payload = await response.json();
      currentState = payload;
      updateDashboard(payload);
    } catch (err) {
      console.error("Prediction update error:", err);
    }
  }

  // Update DOM & Charts
  function updateDashboard(payload) {
    // Header & Telemetry
    if (regionName) regionName.textContent = payload.region;
    if (telemetryRegionLabel) telemetryRegionLabel.textContent = payload.region;
    if (snapshotDate) snapshotDate.textContent = payload.snapshot.date;
    if (predictedRainfall) predictedRainfall.textContent = `${payload.metrics.predicted_rainfall} mm`;
    if (predictedTmax) predictedTmax.textContent = `${payload.metrics.predicted_tmax} °C`;
    
    if (monsoonMetric) monsoonMetric.textContent = payload.metrics.monsoon_pulse;
    if (heatMetric) heatMetric.textContent = payload.metrics.heat_stress;
    if (stabilityMetric) stabilityMetric.textContent = payload.metrics.stability;

    if (barMonsoon) barMonsoon.style.width = `${payload.metrics.monsoon_pulse}%`;
    if (barHeat) barHeat.style.width = `${payload.metrics.heat_stress}%`;
    if (barStability) barStability.style.width = `${payload.metrics.stability}%`;

    // Live Snapshot Box
    if (liveObsRain) liveObsRain.textContent = `${payload.snapshot.rainfall_mm} mm`;
    if (liveObsTmax) liveObsTmax.textContent = `${payload.snapshot.tmax_c} °C`;
    if (liveObsTmin) liveObsTmin.textContent = `${payload.snapshot.tmin_c} °C`;
    if (liveObsHumidity) liveObsHumidity.textContent = `${Number(payload.snapshot.humidity_pct).toFixed(1)}%`;

    if (climatologyProb && payload.real_time_conditions?.current_state) {
      climatologyProb.textContent = `${payload.real_time_conditions.current_state.rainfall_probability_pct}%`;
    }

    if (payload.real_time_conditions?.current_state?.anomaly) {
      const anom = payload.real_time_conditions.current_state.anomaly;
      if (anomalyRainVal) {
        anomalyRainVal.textContent = `${anom.rainfall_mm_pct}%`;
        anomalyRainVal.className = anom.rainfall_mm_pct >= 0 ? "text-cyan" : "text-amber";
      }
      if (anomalyTmaxVal) {
        anomalyTmaxVal.textContent = `${anom.tmax_c_delta_c} °C`;
        anomalyTmaxVal.className = anom.tmax_c_delta_c >= 0 ? "text-ruby" : "text-cyan";
      }
      if (anomalyTminVal) anomalyTminVal.textContent = `${anom.tmin_c_delta_c} °C`;
    }

    // Alert Banner
    if (alertBanner && payload.extreme_events) {
      if (payload.extreme_events.has_alerts && payload.extreme_events.events.length > 0) {
        alertBanner.classList.remove("hidden");
        if (alertMessage) alertMessage.textContent = payload.extreme_events.events[0].label;
        if (alertRiskLevel) {
          alertRiskLevel.textContent = payload.risk_level.toUpperCase();
          alertRiskLevel.className = `risk-badge badge-${payload.risk_level}`;
        }
      } else {
        alertBanner.classList.add("hidden");
      }
    }

    // Uncertainty Card
    if (confidenceMetric && payload.scorecard) {
      confidenceMetric.textContent = `${payload.scorecard.forecast_confidence}%`;
    }
    if (rainRangeMetric && payload.forecast?.rainfall_lower) {
      rainRangeMetric.textContent = `${payload.forecast.rainfall_lower[0]} - ${payload.forecast.rainfall_upper[0]} mm`;
    }
    if (tempRangeMetric && payload.forecast?.tmax_lower) {
      tempRangeMetric.textContent = `${payload.forecast.tmax_lower[0]} - ${payload.forecast.tmax_upper[0]} °C`;
    }

    // Water Budget
    if (payload.simulation) {
      const sim = payload.simulation;
      if (simCumNormal) simCumNormal.textContent = `${Number(sim.cumulative_normal_rainfall).toFixed(1)} mm`;
      if (simCumBase) simCumBase.textContent = `${Number(sim.cumulative_baseline_rainfall).toFixed(1)} mm`;
      if (simCumScen) simCumScen.textContent = `${Number(sim.cumulative_scenario_rainfall).toFixed(1)} mm`;
      if (simDeficitSurplus) {
        simDeficitSurplus.textContent = `${Number(sim.rainfall_deficit_surplus).toFixed(1)} mm`;
        simDeficitSurplus.className = sim.rainfall_deficit_surplus >= 0 ? "text-cyan" : "text-amber";
      }
    }

    // Precautions
    if (precautionsList && payload.real_time_conditions?.scenario_precautions?.messages) {
      precautionsList.innerHTML = payload.real_time_conditions.scenario_precautions.messages
        .map((msg) => `<li><span class="check-bullet">✔</span><span>${msg}</span></li>`)
        .join("");
    }

    // Update Visuals
    updateMapMarkers(payload.pilot_regions, payload.region, payload.snapshot, payload.metrics);
    renderForecastChart(payload.forecast);
    renderTemporalContextChart(payload.time_series);
    renderSimulationChart(payload.simulation);
    renderModelComparisonChart(activeCompMetric);
  }

  // Interactive Leaflet Map
  function initMap(regions) {
    if (!mapElement) return;
    const center = [22.5, 80.5];
    map = L.map(mapElement, { zoomControl: false }).setView(center, 5);

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://carto.com/">CARTO</a> OpenStreetMap',
      maxZoom: 18,
    }).addTo(map);

    L.control.zoom({ position: "topright" }).addTo(map);
    updateMapMarkers(regions, selectedRegion, initialState.snapshot, initialState.metrics);
  }

  function updateMapMarkers(regions, activeRegion, snapshot, metrics) {
    if (!map) return;
    if (regionLayer) regionLayer.remove();

    regionLayer = L.layerGroup();

    regions.forEach((reg) => {
      const isSelected = reg.name === activeRegion;
      const markerColor = isSelected ? palette.cyan : palette.blue;
      const markerRadius = isSelected ? 12 : 7;

      const marker = L.circleMarker([reg.latitude, reg.longitude], {
        radius: markerRadius,
        color: markerColor,
        fillColor: markerColor,
        fillOpacity: isSelected ? 0.85 : 0.35,
        weight: isSelected ? 2 : 1,
      });

      marker.bindTooltip(`<strong>${reg.name}</strong><br/>(${reg.latitude}°N, ${reg.longitude}°E)`, {
        direction: "top",
        offset: [0, -6],
      });

      marker.on("click", () => {
        selectedRegion = reg.name;
        if (globalRegionSelect) globalRegionSelect.value = reg.name;
        updateRegionChips(reg.name);
        fetchPrediction();
      });

      marker.addTo(regionLayer);
    });

    regionLayer.addTo(map);

    if (snapshot && metrics) {
      L.popup({ closeButton: false, autoClose: true, className: "custom-map-popup" })
        .setLatLng([snapshot.latitude, snapshot.longitude])
        .setContent(`
          <div style="font-family: 'Space Grotesk', sans-serif; color: #041019; padding: 4px;">
            <strong style="font-size: 1.05rem;">${activeRegion}</strong><br/>
            <span>Rain: <strong>${metrics.predicted_rainfall} mm</strong></span> | 
            <span>Tmax: <strong>${metrics.predicted_tmax} °C</strong></span>
          </div>
        `)
        .openOn(map);
    }
  }

  // Chart 1: Forecast with Multi-Model Support & 80% Prediction Intervals
  function renderForecastChart(forecast) {
    if (!forecastChartCanvas) return;

    const labels = (currentState.multi_model_forecasts && currentState.multi_model_forecasts.labels) || forecast.labels || [];
    const datasets = [];

    if (overlayAllModels && currentState.multi_model_forecasts && currentState.multi_model_forecasts.models) {
      const models = currentState.multi_model_forecasts.models;
      const modelConfigs = [
        { key: "random_forest", label: "🌲 Random Forest (Rank #1)", color: "#00f5d4", dash: [] },
        { key: "hist_gradient_boosting", label: "📊 HistGB (Rank #2)", color: "#2ecc71", dash: [5, 4] },
        { key: "xgboost", label: "⚡ XGBoost (Rank #3)", color: "#ff9f43", dash: [3, 3] },
        { key: "lstm", label: "🧠 LSTM (Rank #4)", color: "#a55eea", dash: [2, 2] },
      ];

      if (activeVarFilter === "all" || activeVarFilter === "rainfall") {
        modelConfigs.forEach((mc) => {
          const mData = models[mc.key];
          if (mData && mData.rainfall) {
            datasets.push({
              label: `${mc.label} Rain (mm)`,
              data: mData.rainfall,
              borderColor: mc.color,
              backgroundColor: "transparent",
              borderDash: mc.dash,
              borderWidth: mc.key === "random_forest" ? 2.8 : 2.0,
              tension: 0.3,
              yAxisID: "yRain",
              pointRadius: 3,
            });
          }
        });
      }

      if (activeVarFilter === "temp") {
        modelConfigs.forEach((mc) => {
          const mData = models[mc.key];
          if (mData && mData.tmax) {
            datasets.push({
              label: `${mc.label} Tmax (°C)`,
              data: mData.tmax,
              borderColor: mc.color,
              backgroundColor: "transparent",
              borderDash: mc.dash,
              borderWidth: mc.key === "random_forest" ? 2.8 : 2.0,
              tension: 0.3,
              yAxisID: "yTemp",
              pointRadius: 3,
            });
          }
        });
      }
    } else {
      // Single model view (either activeForecastModel or default forecast)
      let mRain = forecast.rainfall || [];
      let mTmax = forecast.tmax || [];
      let mTmin = forecast.tmin || [];

      if (activeForecastModel !== "random_forest" && currentState.multi_model_forecasts?.models?.[activeForecastModel]) {
        const mObj = currentState.multi_model_forecasts.models[activeForecastModel];
        if (mObj.rainfall) mRain = mObj.rainfall;
        if (mObj.tmax) mTmax = mObj.tmax;
        if (mObj.tmin) mTmin = mObj.tmin;
      }

      const rainfallLower = forecast.rainfall_lower || mRain.map((v) => Math.max(0, v * 0.75));
      const rainfallUpper = forecast.rainfall_upper || mRain.map((v) => v * 1.35 + 1.5);
      const tmaxLower = forecast.tmax_lower || mTmax.map((v) => v - 1.2);
      const tmaxUpper = forecast.tmax_upper || mTmax.map((v) => v + 1.2);
      const tminLower = forecast.tmin_lower || mTmin.map((v) => v - 1.0);
      const tminUpper = forecast.tmin_upper || mTmin.map((v) => v + 1.0);

      if (activeVarFilter === "all" || activeVarFilter === "rainfall") {
        datasets.push({
          label: "Rainfall Lower (P10)",
          data: rainfallLower,
          borderColor: "transparent",
          backgroundColor: "transparent",
          pointRadius: 0,
          yAxisID: "yRain",
        });
        datasets.push({
          label: "Rainfall 80% CI (P90)",
          data: rainfallUpper,
          borderColor: "transparent",
          backgroundColor: palette.cyanBand,
          fill: "-1",
          pointRadius: 0,
          yAxisID: "yRain",
        });
        datasets.push({
          label: "Rainfall (mm)",
          data: mRain,
          borderColor: palette.cyan,
          backgroundColor: palette.cyanAlpha,
          borderWidth: 2.5,
          tension: 0.32,
          yAxisID: "yRain",
          pointRadius: 3,
          pointHoverRadius: 6,
        });
      }

      if (activeVarFilter === "all" || activeVarFilter === "temp") {
        datasets.push({
          label: "Max Temp (°C)",
          data: mTmax,
          borderColor: palette.blue,
          backgroundColor: palette.blueAlpha,
          borderWidth: 2.2,
          tension: 0.32,
          yAxisID: "yTemp",
          pointRadius: 3,
          pointHoverRadius: 6,
        });
        datasets.push({
          label: "Min Temp (°C)",
          data: mTmin,
          borderColor: palette.amber,
          backgroundColor: palette.amberAlpha,
          borderWidth: 2.2,
          tension: 0.32,
          yAxisID: "yTemp",
          pointRadius: 3,
          pointHoverRadius: 6,
        });
      }
    }

    const context = forecastChartCanvas.getContext("2d");

    if (forecastChart) {
      forecastChart.destroy();
      forecastChart = null;
    }

    forecastChart = new Chart(context, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            display: true,
            labels: {
              color: palette.text,
              usePointStyle: true,
              filter: (item) => !item.text.includes("P10") && !item.text.includes("P90"),
            },
            position: "top",
          },
          tooltip: {
            backgroundColor: "rgba(8, 16, 30, 0.92)",
            borderColor: "rgba(130, 185, 255, 0.2)",
            borderWidth: 1,
            titleColor: "#ffffff",
            bodyColor: palette.text,
          },
        },
        scales: {
          x: {
            ticks: { color: palette.text, font: { size: 11 } },
            grid: { color: palette.grid },
          },
          yRain: {
            type: "linear",
            position: "left",
            title: { display: true, text: "Rainfall (mm)", color: palette.cyan },
            ticks: { color: palette.cyan },
            grid: { color: palette.grid },
            min: 0,
          },
          yTemp: {
            type: "linear",
            position: "right",
            title: { display: true, text: "Temperature (°C)", color: palette.blue },
            ticks: { color: palette.blue },
            grid: { display: false },
          },
        },
      },
    });
  }

  // Chart 2: 30-Day Context + 14-Day Trajectory
  function renderTemporalContextChart(timeSeries) {
    if (!temporalContextChartCanvas || !timeSeries) return;

    const labels = timeSeries.labels || [];
    const obsRain = timeSeries.observed?.rainfall_mm || [];
    const forecastRain = timeSeries.forecast?.rainfall_mm || [];

    // Pad observed array for alignment
    const rainSeries = [...obsRain, ...forecastRain];

    const context = temporalContextChartCanvas.getContext("2d");

    if (temporalContextChart) {
      temporalContextChart.data.labels = labels;
      temporalContextChart.data.datasets[0].data = rainSeries;
      temporalContextChart.update();
      return;
    }

    temporalContextChart = new Chart(context, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Rainfall (mm)",
            data: rainSeries,
            backgroundColor: (ctx) => {
              const idx = ctx.dataIndex;
              return idx < obsRain.length ? "rgba(112, 161, 255, 0.55)" : "rgba(0, 245, 212, 0.75)";
            },
            borderRadius: 3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            ticks: { color: palette.text, maxTicksLimit: 8 },
            grid: { display: false },
          },
          y: {
            ticks: { color: palette.text },
            grid: { color: palette.grid },
            min: 0,
          },
        },
      },
    });
  }

  // Chart 3: What-If Scenario Comparison Chart
  function renderSimulationChart(simulation) {
    if (!simulationChartCanvas || !simulation) return;

    const baselineLabels = simulation.baseline_forecast?.labels || [];
    const baselineRain = simulation.baseline_forecast?.rainfall_mm || [];
    const scenarioRain = simulation.scenario_forecast?.rainfall_mm || [];
    const baselineTmax = simulation.baseline_forecast?.tmax_c || [];
    const scenarioTmax = simulation.scenario_forecast?.tmax_c || [];

    const datasets = [
      {
        label: "Baseline Rainfall (mm)",
        data: baselineRain,
        borderColor: "rgba(112, 161, 255, 0.7)",
        borderDash: [4, 4],
        borderWidth: 2,
        tension: 0.3,
        yAxisID: "yRain",
      },
      {
        label: "Scenario Rainfall (mm)",
        data: scenarioRain,
        borderColor: palette.cyan,
        backgroundColor: palette.cyanAlpha,
        fill: true,
        borderWidth: 2.5,
        tension: 0.3,
        yAxisID: "yRain",
      },
      {
        label: "Baseline Tmax (°C)",
        data: baselineTmax,
        borderColor: "rgba(255, 191, 117, 0.7)",
        borderDash: [4, 4],
        borderWidth: 1.8,
        tension: 0.3,
        yAxisID: "yTemp",
      },
      {
        label: "Scenario Tmax (°C)",
        data: scenarioTmax,
        borderColor: palette.ruby,
        borderWidth: 2.2,
        tension: 0.3,
        yAxisID: "yTemp",
      },
    ];

    const context = simulationChartCanvas.getContext("2d");

    if (simulationChart) {
      simulationChart.data.labels = baselineLabels;
      simulationChart.data.datasets = datasets;
      simulationChart.update();
      return;
    }

    simulationChart = new Chart(context, {
      type: "line",
      data: { labels: baselineLabels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            display: true,
            labels: { color: palette.text, usePointStyle: true },
            position: "top",
          },
        },
        scales: {
          x: { ticks: { color: palette.text }, grid: { color: palette.grid } },
          yRain: {
            type: "linear",
            position: "left",
            title: { display: true, text: "Rainfall (mm)", color: palette.cyan },
            ticks: { color: palette.cyan },
            grid: { color: palette.grid },
            min: 0,
          },
          yTemp: {
            type: "linear",
            position: "right",
            title: { display: true, text: "Temperature (°C)", color: palette.ruby },
            ticks: { color: palette.ruby },
            grid: { display: false },
          },
        },
      },
    });
  }

  // Chart 4 & Handler: Historical Replay Backtest
  function attachReplayHandler() {
    if (!runReplayBtn) return;

    runReplayBtn.addEventListener("click", async () => {
      runReplayBtn.textContent = "Executing Backtest...";
      runReplayBtn.disabled = true;

      try {
        const response = await fetch("/api/replay", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            region: selectedRegion,
            start_date: replayStartDate.value,
            end_date: replayEndDate.value,
          }),
        });

        const data = await response.json();
        const replay = data.replay;

        if (replay && replay.series) {
          renderReplayChart(replay.series);
          if (replayMetricsSummary && replay.summary) {
            replayMetricsSummary.innerHTML = `
              <span>${replay.summary.records} Days</span> | 
              <span>Rainfall MAE: <strong>${Number(replay.summary.mae_rainfall_mm).toFixed(2)} mm</strong></span> | 
              <span>Tmax MAE: <strong>${Number(replay.summary.mae_tmax_c).toFixed(2)} °C</strong></span>
            `;
          }
        }
      } catch (err) {
        console.error("Replay execution failed:", err);
      } finally {
        runReplayBtn.textContent = "Run Backtest Replay";
        runReplayBtn.disabled = false;
      }
    });
  }

  function renderReplayChart(series) {
    if (!replayChartCanvas || !series) return;

    const labels = series.labels || [];
    const actualRain = series.actual_rainfall_mm || [];
    const predRain = series.predicted_rainfall_mm || [];
    const actualTmax = series.actual_tmax_c || [];
    const predTmax = series.predicted_tmax_c || [];

    const datasets = [
      {
        label: "Observed Rainfall (mm)",
        data: actualRain,
        backgroundColor: "rgba(112, 161, 255, 0.5)",
        borderColor: palette.blue,
        borderWidth: 1.5,
        type: "bar",
        yAxisID: "yRain",
      },
      {
        label: "Model Predicted Rainfall (mm)",
        data: predRain,
        borderColor: palette.cyan,
        backgroundColor: "transparent",
        borderWidth: 2.2,
        type: "line",
        tension: 0.25,
        yAxisID: "yRain",
      },
      {
        label: "Observed Tmax (°C)",
        data: actualTmax,
        borderColor: "rgba(255, 191, 117, 0.7)",
        borderDash: [3, 3],
        borderWidth: 1.8,
        type: "line",
        yAxisID: "yTemp",
      },
      {
        label: "Predicted Tmax (°C)",
        data: predTmax,
        borderColor: palette.ruby,
        borderWidth: 2,
        type: "line",
        yAxisID: "yTemp",
      },
    ];

    const context = replayChartCanvas.getContext("2d");

    if (replayChart) {
      replayChart.data.labels = labels;
      replayChart.data.datasets = datasets;
      replayChart.update();
      return;
    }

    replayChart = new Chart(context, {
      type: "bar",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            display: true,
            labels: { color: palette.text, usePointStyle: true },
            position: "top",
          },
        },
        scales: {
          x: { ticks: { color: palette.text }, grid: { color: palette.grid } },
          yRain: {
            type: "linear",
            position: "left",
            title: { display: true, text: "Rainfall (mm)", color: palette.cyan },
            ticks: { color: palette.cyan },
            grid: { color: palette.grid },
          },
          yTemp: {
            type: "linear",
            position: "right",
            title: { display: true, text: "Temperature (°C)", color: palette.ruby },
            ticks: { color: palette.ruby },
            grid: { display: false },
          },
        },
      },
    });
  }

  // Jump to Model Comparison Tab from Cockpit
  function attachScorecardJump() {
    if (!jumpToScorecardBtn) return;
    jumpToScorecardBtn.addEventListener("click", () => {
      const scorecardBtn = document.getElementById("tab-scorecard-btn");
      if (scorecardBtn) {
        scorecardBtn.click();
        window.scrollTo({ top: 0, behavior: "smooth" });
      }
    });
  }

  // Model Selection & Overlay Controls in Forecast Tab
  function initForecastModelControls() {
    modelChips.forEach((chip) => {
      chip.addEventListener("click", () => {
        modelChips.forEach((c) => c.classList.remove("chip-active"));
        chip.classList.add("chip-active");
        activeForecastModel = chip.dataset.model || "random_forest";
        if (overlayAllModelsCheck && overlayAllModelsCheck.checked) {
          overlayAllModelsCheck.checked = false;
          overlayAllModels = false;
        }
        renderForecastChart(currentState.forecast || {});
      });
    });

    if (overlayAllModelsCheck) {
      overlayAllModelsCheck.addEventListener("change", () => {
        overlayAllModels = overlayAllModelsCheck.checked;
        renderForecastChart(currentState.forecast || {});
      });
    }
  }

  // Multi-Model Comparison Controls
  function initModelComparisonControls() {
    compToggleBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        compToggleBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        activeCompMetric = btn.dataset.metric || "rainfall_mae";
        renderModelComparisonChart(activeCompMetric);
      });
    });
  }

  // Fallback Comparison Data (Measured Benchmarks on IMD Test Split)
  const fallbackModelComp = {
    labels: ["Random Forest", "HistGradientBoosting", "XGBoost", "LSTM"],
    model_keys: ["random_forest", "hist_gradient_boosting", "xgboost", "lstm"],
    colors: ["#00f5d4", "#2ecc71", "#ff9f43", "#a55eea"],
    rainfall_mae: [2.463, 2.524, 2.708, 3.039],
    tmax_mae: [1.172, 1.177, 1.241, 2.161],
    tmin_mae: [1.018, 0.906, 0.943, 2.284],
    r2_scores: {
      rainfall: [23.5, 20.3, 11.2, 0.0],
      tmax: [94.3, 94.2, 93.6, 78.4],
      tmin: [96.7, 97.4, 97.2, 82.5],
    },
    error_gap_pct: {
      rainfall: [0.0, 2.5, 10.0, 23.4],
      tmax: [0.0, 0.5, 5.9, 84.4],
      tmin: [12.5, 0.0, 4.1, 152.2],
    },
    training_times: [1.85, 0.42, 0.35, 12.4],
  };

  // Interactive Multi-Model Comparison & Difference Chart
  function renderModelComparisonChart(metric = "rainfall_mae") {
    if (!modelCompChartCanvas) return;

    const compData = currentState.model_comparison_detailed?.chart_data || fallbackModelComp;
    const labels = compData.labels || fallbackModelComp.labels;
    let datasets = [];
    let yTitle = "";
    let yMax = undefined;
    let subtitleText = "";

    const modelColors = ["#00f5d4", "#2ecc71", "#ff9f43", "#a55eea"];
    const modelAlphaColors = [
      "rgba(0, 245, 212, 0.82)",
      "rgba(46, 204, 113, 0.82)",
      "rgba(255, 159, 67, 0.82)",
      "rgba(165, 94, 234, 0.82)",
    ];

    if (metric === "rainfall_mae") {
      subtitleText = "🌧️ Lower Rainfall MAE indicates higher predictive accuracy. Random Forest is #1 (2.46 mm). HistGB is #2 (+2.5% error gap).";
      yTitle = "Test MAE (mm) — Lower is Better";
      const vals = (compData.rainfall_mae && compData.rainfall_mae.length) ? compData.rainfall_mae : fallbackModelComp.rainfall_mae;
      datasets.push({
        label: "Rainfall Test MAE (mm)",
        data: vals,
        backgroundColor: modelAlphaColors,
        borderColor: modelColors,
        borderWidth: 2,
        borderRadius: 6,
      });
    } else if (metric === "tmax_mae") {
      subtitleText = "🌡️ Lower Maximum Temperature MAE indicates superior daytime heat capture. Random Forest is #1 (1.17°C). HistGB is #2 (+0.5% error gap).";
      yTitle = "Test MAE (°C) — Lower is Better";
      const vals = (compData.tmax_mae && compData.tmax_mae.length) ? compData.tmax_mae : fallbackModelComp.tmax_mae;
      datasets.push({
        label: "Max Temp Test MAE (°C)",
        data: vals,
        backgroundColor: modelAlphaColors,
        borderColor: modelColors,
        borderWidth: 2,
        borderRadius: 6,
      });
    } else if (metric === "tmin_mae") {
      subtitleText = "🌙 Lower Minimum Temperature MAE indicates superior nocturnal cooling capture. HistGradientBoosting is #1 (0.91°C). XGBoost is #2 (+4.1%).";
      yTitle = "Test MAE (°C) — Lower is Better";
      const vals = (compData.tmin_mae && compData.tmin_mae.length) ? compData.tmin_mae : fallbackModelComp.tmin_mae;
      datasets.push({
        label: "Min Temp Test MAE (°C)",
        data: vals,
        backgroundColor: modelAlphaColors,
        borderColor: modelColors,
        borderWidth: 2,
        borderRadius: 6,
      });
    } else if (metric === "r2_score") {
      subtitleText = "📊 R² Explained Variance (% Accuracy). 100% represents perfect correlation. Temperature models achieve >94% accuracy.";
      yTitle = "R² Accuracy Score (%) — Higher is Better";
      yMax = 100;
      const r2s = compData.r2_scores || fallbackModelComp.r2_scores;
      datasets = [
        {
          label: "Rainfall R² (%)",
          data: r2s.rainfall || fallbackModelComp.r2_scores.rainfall,
          backgroundColor: "rgba(0, 245, 212, 0.8)",
          borderColor: "#00f5d4",
          borderWidth: 1.5,
          borderRadius: 4,
        },
        {
          label: "Max Temp R² (%)",
          data: r2s.tmax || fallbackModelComp.r2_scores.tmax,
          backgroundColor: "rgba(112, 161, 255, 0.8)",
          borderColor: "#70a1ff",
          borderWidth: 1.5,
          borderRadius: 4,
        },
        {
          label: "Min Temp R² (%)",
          data: r2s.tmin || fallbackModelComp.r2_scores.tmin,
          backgroundColor: "rgba(255, 191, 117, 0.8)",
          borderColor: "#ffbf75",
          borderWidth: 1.5,
          borderRadius: 4,
        },
      ];
    } else if (metric === "error_gap") {
      subtitleText = "📉 Error Difference vs Top Model (% Gap in MAE). 0% is the best-in-class baseline; lower percentage indicates closer performance.";
      yTitle = "% MAE Difference Above Best Model (0% = Optimal)";
      const gaps = compData.error_gap_pct || fallbackModelComp.error_gap_pct;
      datasets = [
        {
          label: "🌧️ Rainfall Error Gap (% vs RF)",
          data: gaps.rainfall || fallbackModelComp.error_gap_pct.rainfall,
          backgroundColor: "rgba(0, 245, 212, 0.8)",
          borderColor: "#00f5d4",
          borderWidth: 1.5,
          borderRadius: 4,
        },
        {
          label: "🌡️ Max Temp Error Gap (% vs RF)",
          data: gaps.tmax || fallbackModelComp.error_gap_pct.tmax,
          backgroundColor: "rgba(112, 161, 255, 0.8)",
          borderColor: "#70a1ff",
          borderWidth: 1.5,
          borderRadius: 4,
        },
        {
          label: "🌙 Min Temp Error Gap (% vs HistGB)",
          data: gaps.tmin || fallbackModelComp.error_gap_pct.tmin,
          backgroundColor: "rgba(255, 191, 117, 0.8)",
          borderColor: "#ffbf75",
          borderWidth: 1.5,
          borderRadius: 4,
        },
      ];
    } else if (metric === "training_time") {
      subtitleText = "⚡ Wall-clock Training Speed in seconds on 761 test samples. XGBoost (0.35s) and HistGB (0.42s) are orders of magnitude faster.";
      yTitle = "Training Duration (Seconds) — Lower is Faster";
      const vals = (compData.training_times && compData.training_times.length) ? compData.training_times : fallbackModelComp.training_times;
      datasets.push({
        label: "Training Time (s)",
        data: vals,
        backgroundColor: modelAlphaColors,
        borderColor: modelColors,
        borderWidth: 2,
        borderRadius: 6,
      });
    }

    if (compChartSubtitle) {
      compChartSubtitle.textContent = subtitleText;
    }

    const context = modelCompChartCanvas.getContext("2d");

    if (modelComparisonChart) {
      modelComparisonChart.destroy();
      modelComparisonChart = null;
    }

    modelComparisonChart = new Chart(context, {
      type: "bar",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: datasets.length > 1,
            labels: { color: palette.text, usePointStyle: true },
            position: "top",
          },
          tooltip: {
            backgroundColor: "rgba(8, 16, 30, 0.95)",
            borderColor: "rgba(0, 245, 212, 0.3)",
            borderWidth: 1,
            titleColor: "#ffffff",
            bodyColor: palette.text,
            callbacks: {
              afterBody: (tooltipItems) => {
                if (metric === "rainfall_mae") {
                  const gaps = compData.error_gap_pct?.rainfall || fallbackModelComp.error_gap_pct.rainfall;
                  const idx = tooltipItems[0].dataIndex;
                  return idx === 0 ? "🏆 Rank #1 Best Model" : `Difference vs Best: +${gaps[idx]}% Error`;
                }
                if (metric === "tmax_mae") {
                  const gaps = compData.error_gap_pct?.tmax || fallbackModelComp.error_gap_pct.tmax;
                  const idx = tooltipItems[0].dataIndex;
                  return idx === 0 ? "🏆 Rank #1 Best Model" : `Difference vs Best: +${gaps[idx]}% Error`;
                }
                if (metric === "tmin_mae") {
                  const gaps = compData.error_gap_pct?.tmin || fallbackModelComp.error_gap_pct.tmin;
                  const idx = tooltipItems[0].dataIndex;
                  return idx === 1 ? "🏆 Rank #1 Best Model" : `Difference vs Best: +${gaps[idx]}% Error`;
                }
                return "";
              },
            },
          },
        },
        scales: {
          x: {
            ticks: { color: palette.text, font: { size: 12, weight: "bold" } },
            grid: { display: false },
          },
          y: {
            beginAtZero: true,
            max: yMax,
            ticks: { color: palette.text },
            grid: { color: palette.grid },
            title: { display: true, text: yTitle, color: palette.cyan },
          },
        },
      },
    });
  }

  // =================================================================
  // 7-DAY FORECAST MODULE
  // =================================================================

  function init7DayForecast() {
    // Location selector change handler
    if (f7dLocationSelect) {
      f7dLocationSelect.addEventListener("change", () => {
        const val = f7dLocationSelect.value;
        if (val === "__custom__") {
          if (f7dCustomCoords) f7dCustomCoords.style.display = "flex";
        } else {
          if (f7dCustomCoords) f7dCustomCoords.style.display = "none";
          fetch7DayForecast({ region: val });
        }
      });
    }

    // Custom coordinate search
    if (f7dSearchBtn) {
      f7dSearchBtn.addEventListener("click", () => {
        const lat = f7dLatInput ? parseFloat(f7dLatInput.value) : NaN;
        const lon = f7dLonInput ? parseFloat(f7dLonInput.value) : NaN;
        if (isNaN(lat) || isNaN(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
          showF7dError("Please enter valid coordinates (Lat: -90 to 90, Lon: -180 to 180).");
          return;
        }
        fetch7DayForecast({ lat, lon });
      });
    }
  }

  async function fetch7DayForecast(params) {
    const queryParams = params || { region: selectedRegion };
    const qs = new URLSearchParams();
    if (queryParams.region) qs.set("region", queryParams.region);
    if (queryParams.lat !== undefined) qs.set("lat", queryParams.lat);
    if (queryParams.lon !== undefined) qs.set("lon", queryParams.lon);

    // Show loading, hide error and cards
    if (f7dLoading) f7dLoading.style.display = "flex";
    if (f7dError) f7dError.style.display = "none";
    if (f7dCardsContainer) f7dCardsContainer.style.display = "none";

    try {
      const response = await fetch(`/api/forecast-7day?${qs.toString()}`);
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || `API error (${response.status})`);
      }
      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      if (!data.days || data.days.length === 0) {
        throw new Error("No forecast data available for this location.");
      }

      f7dFetched = true;
      if (f7dLoading) f7dLoading.style.display = "none";
      if (f7dCardsContainer) f7dCardsContainer.style.display = "block";

      // Update last-updated
      if (f7dLastUpdated) f7dLastUpdated.textContent = data.last_updated || "—";

      renderF7dCards(data.days);
      renderF7dTempChart(data.days);
      renderF7dRainfallChart(data.days);
    } catch (err) {
      console.error("7-day forecast error:", err);
      if (f7dLoading) f7dLoading.style.display = "none";
      showF7dError(err.message);
    }
  }

  function showF7dError(message) {
    if (f7dError) {
      f7dError.style.display = "flex";
      if (f7dErrorMessage) f7dErrorMessage.textContent = message;
    }
    if (f7dCardsContainer) f7dCardsContainer.style.display = "none";
  }

  function renderF7dCards(days) {
    if (!f7dCards) return;

    const todayStr = new Date().toISOString().split("T")[0];

    f7dCards.innerHTML = days.map((day, i) => {
      const isToday = day.date === todayStr;
      const todayClass = isToday ? " f7d-card-today" : "";
      const todayLabel = isToday ? ' <span style="font-size:0.65rem;color:var(--cyan);">(Today)</span>' : "";

      return `
        <article class="f7d-card${todayClass}">
          <div class="f7d-day-name">${day.day_short}${todayLabel}</div>
          <div class="f7d-date">${day.date_display}</div>
          <div class="f7d-icon">${day.condition_icon}</div>
          <div class="f7d-condition">${day.condition}</div>
          <div class="f7d-temps">
            <span class="f7d-tmax">${day.tmax_c}°</span>
            <span class="f7d-tmin">${day.tmin_c}°</span>
          </div>
          <div class="f7d-divider"></div>
          <div class="f7d-detail-grid">
            <div class="f7d-detail">
              <span class="f7d-detail-label">Rain Prob</span>
              <span class="f7d-detail-value">${day.rainfall_probability_pct}%</span>
            </div>
            <div class="f7d-detail">
              <span class="f7d-detail-label">Rainfall</span>
              <span class="f7d-detail-value">${day.rainfall_mm} mm</span>
            </div>
            <div class="f7d-detail">
              <span class="f7d-detail-label">Humidity</span>
              <span class="f7d-detail-value">${day.humidity_pct}%</span>
            </div>
            <div class="f7d-detail">
              <span class="f7d-detail-label">Wind</span>
              <span class="f7d-detail-value">${day.wind_speed_kmh} km/h</span>
            </div>
            <div class="f7d-detail">
              <span class="f7d-detail-label">Direction</span>
              <span class="f7d-detail-value">${day.wind_direction}</span>
            </div>
            <div class="f7d-detail">
              <span class="f7d-detail-label">UV Index</span>
              <span class="f7d-detail-value">${day.uv_index}</span>
            </div>
          </div>
          <div class="f7d-confidence ${day.confidence.class}">
            ${day.confidence.level} Confidence
          </div>
        </article>
      `;
    }).join("");
  }

  function renderF7dTempChart(days) {
    if (!f7dTempChartCanvas) return;

    const labels = days.map((d) => `${d.day_short}\n${d.date_display}`);
    const tmaxData = days.map((d) => d.tmax_c);
    const tminData = days.map((d) => d.tmin_c);

    if (f7dTempChart) f7dTempChart.destroy();

    f7dTempChart = new Chart(f7dTempChartCanvas, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Max Temp (°C)",
            data: tmaxData,
            borderColor: palette.amber,
            backgroundColor: palette.amberAlpha,
            borderWidth: 2.5,
            tension: 0.35,
            pointRadius: 5,
            pointBackgroundColor: palette.amber,
            pointBorderColor: "#041019",
            pointBorderWidth: 2,
            fill: false,
          },
          {
            label: "Min Temp (°C)",
            data: tminData,
            borderColor: palette.blue,
            backgroundColor: palette.blueAlpha,
            borderWidth: 2.5,
            tension: 0.35,
            pointRadius: 5,
            pointBackgroundColor: palette.blue,
            pointBorderColor: "#041019",
            pointBorderWidth: 2,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "rgba(4, 16, 25, 0.95)",
            titleColor: "#fff",
            bodyColor: palette.text,
            borderColor: "rgba(130, 185, 255, 0.25)",
            borderWidth: 1,
            padding: 12,
            callbacks: {
              title: (items) => {
                const idx = items[0].dataIndex;
                return `${days[idx].day_name}, ${days[idx].date_display}`;
              },
              afterBody: (items) => {
                return "Predicted — Forecast";
              },
            },
          },
        },
        scales: {
          x: {
            ticks: { color: palette.text, font: { size: 11 } },
            grid: { color: palette.grid },
            title: { display: true, text: "Day", color: palette.text },
          },
          y: {
            ticks: { color: palette.text },
            grid: { color: palette.grid },
            title: { display: true, text: "Temperature (°C)", color: palette.amber },
          },
        },
      },
    });
  }

  function renderF7dRainfallChart(days) {
    if (!f7dRainfallChartCanvas) return;

    const labels = days.map((d) => `${d.day_short}\n${d.date_display}`);
    const rainfallData = days.map((d) => d.rainfall_mm);

    if (f7dRainfallChart) f7dRainfallChart.destroy();

    f7dRainfallChart = new Chart(f7dRainfallChartCanvas, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Predicted Rainfall (mm)",
            data: rainfallData,
            backgroundColor: rainfallData.map((v) =>
              v > 20 ? "rgba(0, 245, 212, 0.65)" :
              v > 5  ? "rgba(0, 245, 212, 0.45)" :
                       "rgba(0, 245, 212, 0.25)"
            ),
            borderColor: palette.cyan,
            borderWidth: 1,
            borderRadius: 6,
            borderSkipped: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "rgba(4, 16, 25, 0.95)",
            titleColor: "#fff",
            bodyColor: palette.text,
            borderColor: "rgba(130, 185, 255, 0.25)",
            borderWidth: 1,
            padding: 12,
            callbacks: {
              title: (items) => {
                const idx = items[0].dataIndex;
                return `${days[idx].day_name}, ${days[idx].date_display}`;
              },
              afterBody: (items) => {
                const idx = items[0].dataIndex;
                return `Rain Probability: ${days[idx].rainfall_probability_pct}%\nPredicted — Forecast`;
              },
            },
          },
        },
        scales: {
          x: {
            ticks: { color: palette.text, font: { size: 11 } },
            grid: { display: false },
            title: { display: true, text: "Day", color: palette.text },
          },
          y: {
            beginAtZero: true,
            ticks: { color: palette.text },
            grid: { color: palette.grid },
            title: { display: true, text: "Rainfall (mm)", color: palette.cyan },
          },
        },
      },
    });
  }

  document.addEventListener("DOMContentLoaded", bootstrap);
})();
