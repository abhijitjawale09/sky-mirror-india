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

  // Chart Instances & Map
  let forecastChart = null;
  let simulationChart = null;
  let temporalContextChart = null;
  let replayChart = null;
  let map = null;
  let regionLayer = null;

  let selectedRegion = initialState.region || "Kerala Coast";
  let activeVarFilter = "all";

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
    renderForecastChart(initialState.forecast || {});
    renderTemporalContextChart(initialState.time_series || {});
    renderSimulationChart(initialState.simulation || {});
    attachRegionButtons();
    attachPresetChips();
    attachReplayHandler();
    attachLiveSyncHandler();
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
          if (map) map.invalidateSize();
        }, 60);
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

  // Chart 1: Forecast with 80% Prediction Intervals
  function renderForecastChart(forecast) {
    if (!forecastChartCanvas) return;

    const labels = forecast.labels || [];
    const rainfall = forecast.rainfall || [];
    const rainfallLower = forecast.rainfall_lower || rainfall.map((v) => Math.max(0, v * 0.75));
    const rainfallUpper = forecast.rainfall_upper || rainfall.map((v) => v * 1.35 + 1.5);

    const tmax = forecast.tmax || [];
    const tmaxLower = forecast.tmax_lower || tmax.map((v) => v - 1.2);
    const tmaxUpper = forecast.tmax_upper || tmax.map((v) => v + 1.2);

    const tmin = forecast.tmin || [];
    const tminLower = forecast.tmin_lower || tmin.map((v) => v - 1.0);
    const tminUpper = forecast.tmin_upper || tmin.map((v) => v + 1.0);

    const datasets = [];

    if (activeVarFilter === "all" || activeVarFilter === "rainfall") {
      // Shaded Prediction Interval Band
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
      // Point Estimate
      datasets.push({
        label: "Rainfall (mm)",
        data: rainfall,
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
        data: tmax,
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
        data: tmin,
        borderColor: palette.amber,
        backgroundColor: palette.amberAlpha,
        borderWidth: 2.2,
        tension: 0.32,
        yAxisID: "yTemp",
        pointRadius: 3,
        pointHoverRadius: 6,
      });
    }

    const context = forecastChartCanvas.getContext("2d");

    if (forecastChart) {
      forecastChart.data.labels = labels;
      forecastChart.data.datasets = datasets;
      forecastChart.update();
      return;
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

  document.addEventListener("DOMContentLoaded", bootstrap);
})();
