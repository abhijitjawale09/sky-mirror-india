from pathlib import Path
import pandas as pd
import pytest

from climate_twin.data.loader import get_region_profile
from climate_twin.services.live_feed import LiveFeedService


def test_live_feed_service_fetch_region():
    service = LiveFeedService(past_days=5)
    profile = get_region_profile("Kerala Coast")
    df = service.fetch_region_live_data(profile)

    assert not df.empty
    assert "date" in df.columns
    assert "rainfall_mm" in df.columns
    assert "tmax_c" in df.columns
    assert "tmin_c" in df.columns
    assert "humidity_pct" in df.columns
    assert df["region"].iloc[0] == "Kerala Coast"
    assert len(df) >= 5


def test_live_sync_all_regions():
    service = LiveFeedService(past_days=7)
    result = service.sync_all_regions()

    assert result.status in {"success", "cached_fallback"}
    assert result.records_count > 0
    assert len(result.regions_synced) == 7
    assert result.latest_date is not None
