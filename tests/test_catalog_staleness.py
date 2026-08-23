"""The catalog must be able to tell you it is old.

Before this suite, `ModelCatalog.load()` ran every model through `upsert()`,
which stamps `updated_at = now`. A catalog written in 2019 therefore reported as
fetched today, and one load-and-save cycle wrote that laundered date back to
disk, destroying the real one. The reconciliation report told operators "the
catalog is stale, run refresh" while nothing in the package could detect
staleness at all.

Every test here fails if the fetch timestamp goes back to being derived from the
load. The regression test at the top is the one that mattered.
"""

import contextlib
import io
import json
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

from openrouter_model_router import ModelCatalog, ModelInfo
from openrouter_model_router.catalog import (
    DEFAULT_MAX_AGE_DAYS,
    STALENESS_UNVERIFIABLE,
    STALENESS_CLOCK_SKEW,
    STALENESS_EMPTY,
    STALENESS_FRESH,
    STALENESS_NEVER_FETCHED,
    STALENESS_STALE,
    STALENESS_UNPARSEABLE,
)
from openrouter_model_router.cli import main

DAY = 86_400.0


def stamp(seconds_ago: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - seconds_ago))


def priced_model(model_id: str = "vendor/priced") -> ModelInfo:
    return ModelInfo(
        id=model_id,
        context_length=200_000,
        input_cost_per_million=1.0,
        output_cost_per_million=3.0,
        capabilities=("text", "long_context"),
    )


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class FetchTimestampSurvivesLoadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "catalog.json"

    def _write_aged(self, seconds_ago: float) -> str:
        when = stamp(seconds_ago)
        catalog = ModelCatalog([priced_model()], updated_at=when, fetched_at=when)
        catalog.save(self.path)
        return when

    def test_load_does_not_restamp_the_fetch_time(self):
        """THE REGRESSION. A catalog fetched in 2019 must still say 2019."""

        ModelCatalog([priced_model()], updated_at="2019-01-01T00:00:00Z", fetched_at="2019-01-01T00:00:00Z").save(
            self.path
        )

        loaded = ModelCatalog.load(self.path)

        self.assertEqual(loaded.fetched_at, "2019-01-01T00:00:00Z")
        self.assertEqual(loaded.updated_at, "2019-01-01T00:00:00Z")
        self.assertGreater(loaded.age_seconds(), 6 * 365 * DAY)

    def test_a_load_and_save_cycle_does_not_destroy_the_fetch_date(self):
        """The old bug was not cosmetic: it overwrote the real date on disk."""

        when = self._write_aged(30 * DAY)

        ModelCatalog.load(self.path).save(self.path)

        self.assertEqual(json.loads(self.path.read_text())["fetched_at"], when)

    def test_recording_an_outcome_does_not_make_the_prices_newer(self):
        catalog = ModelCatalog([priced_model()], fetched_at=stamp(30 * DAY))

        catalog.record_outcome("vendor/priced", success=True, latency_ms=900)

        self.assertTrue(catalog.is_stale())
        self.assertGreater(catalog.age_seconds(), 29 * DAY)

    def test_upsert_moves_updated_at_but_never_fetched_at(self):
        catalog = ModelCatalog([priced_model()], fetched_at=stamp(30 * DAY))
        before = catalog.fetched_at

        catalog.upsert(priced_model("vendor/another"))

        self.assertEqual(catalog.fetched_at, before)
        self.assertNotEqual(catalog.updated_at, before)

    def _write_v1(self, updated_at):
        self.path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "updated_at": updated_at,
                    "models": [priced_model().to_dict()],
                }
            )
        )

    def test_a_recent_looking_v1_catalog_is_unverifiable_not_fresh(self):
        """A v1 updated_at was restamped on every load, so it is an UPPER BOUND
        on the fetch time: such a catalog can only ever look fresher than it is.

        Caught while running the shipped demo against a real v1 catalog on disk,
        which reported `[ok] 0.1d` from a timestamp the old code had written at
        load time. Inheriting a laundered number is not detecting staleness."""

        self._write_v1(stamp(3600))

        report = ModelCatalog.load(self.path).staleness()

        self.assertEqual(report["status"], STALENESS_UNVERIFIABLE)
        self.assertFalse(report["fresh"])
        self.assertTrue(report["fetched_at_is_derived"])
        self.assertIn("upper bound", report["reason"])

    def test_a_v1_catalog_older_than_the_limit_is_plainly_stale(self):
        """Sound in this direction: the real fetch was at or before updated_at."""

        self._write_v1(stamp(90 * DAY))

        self.assertEqual(ModelCatalog.load(self.path).staleness()["status"], STALENESS_STALE)

    def test_one_refresh_clears_the_unverifiable_state(self):
        from openrouter_model_router import FakeTransport

        self._write_v1(stamp(3600))
        current = ModelCatalog.load(self.path)
        payload = {"data": [{"id": "vendor/priced", "context_length": 8000, "pricing": {"prompt": "0.000001", "completion": "0.000002"}}]}

        current.merge(ModelCatalog.refresh_from_openrouter(transport=FakeTransport([FakeTransport.json_response(payload)])))

        report = current.staleness()
        self.assertFalse(report["fetched_at_is_derived"])
        self.assertEqual(report["status"], STALENESS_FRESH)

    def test_a_v1_catalog_with_no_fetched_at_falls_back_to_updated_at(self):
        """Legacy catalogs must be judged stale, not excused as unknowable."""

        self.path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "updated_at": "2019-01-01T00:00:00Z",
                    "models": [priced_model().to_dict()],
                }
            )
        )

        loaded = ModelCatalog.load(self.path)

        self.assertEqual(loaded.fetched_at, "2019-01-01T00:00:00Z")
        self.assertEqual(loaded.staleness()["status"], STALENESS_STALE)


    def test_a_null_fetched_at_does_not_borrow_updated_at(self):
        """`"fetched_at": null` means never fetched, and must not read as fresh.

        Caught by a failing test while building this branch: the legacy fallback
        was keyed on the value being falsy rather than the key being absent, so a
        never-fetched catalog came back from disk wearing updated_at - which
        upsert() restamps to "now" - and reported fresh. That is the original
        laundering bug rebuilt in the fix for it.
        """

        ModelCatalog([priced_model()], fetched_at=None).save(self.path)
        self.assertIsNone(json.loads(self.path.read_text())["fetched_at"])

        loaded = ModelCatalog.load(self.path)

        self.assertIsNone(loaded.fetched_at)
        self.assertEqual(loaded.staleness()["status"], STALENESS_NEVER_FETCHED)


class StalenessTests(unittest.TestCase):
    def test_a_recent_catalog_is_fresh(self):
        catalog = ModelCatalog([priced_model()], fetched_at=stamp(2 * DAY))

        report = catalog.staleness()

        self.assertEqual(report["status"], STALENESS_FRESH)
        self.assertTrue(report["fresh"])
        self.assertAlmostEqual(report["age_days"], 2.0, places=2)

    def test_a_ninety_day_old_catalog_is_stale(self):
        catalog = ModelCatalog([priced_model()], fetched_at=stamp(90 * DAY))

        report = catalog.staleness()

        self.assertEqual(report["status"], STALENESS_STALE)
        self.assertTrue(report["stale"])
        self.assertAlmostEqual(report["age_days"], 90.0, places=1)
        self.assertIn("refresh", report["reason"])

    def test_the_boundary_is_the_threshold_not_a_vibe(self):
        just_inside = ModelCatalog([priced_model()], fetched_at=stamp(DEFAULT_MAX_AGE_DAYS * DAY - 3600))
        just_outside = ModelCatalog([priced_model()], fetched_at=stamp(DEFAULT_MAX_AGE_DAYS * DAY + 3600))

        self.assertTrue(just_inside.staleness()["fresh"])
        self.assertFalse(just_outside.staleness()["fresh"])

    def test_an_unfetched_catalog_is_not_fresh(self):
        report = ModelCatalog([priced_model()], fetched_at=None).staleness()

        self.assertEqual(report["status"], STALENESS_NEVER_FETCHED)
        self.assertTrue(report["stale"])

    def test_the_bootstrap_catalog_never_reads_as_fresh(self):
        """It was never fetched from anywhere; it must not satisfy a freshness gate."""

        self.assertEqual(ModelCatalog.bootstrap().staleness()["status"], STALENESS_NEVER_FETCHED)

    def test_an_unparseable_timestamp_is_not_fresh(self):
        report = ModelCatalog([priced_model()], fetched_at="last tuesday").staleness()

        self.assertEqual(report["status"], STALENESS_UNPARSEABLE)
        self.assertIsNone(report["age_days"])
        self.assertTrue(report["stale"])

    def test_a_future_timestamp_is_a_broken_clock_not_infinite_freshness(self):
        future = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 400 * DAY))

        report = ModelCatalog([priced_model()], fetched_at=future).staleness()

        self.assertEqual(report["status"], STALENESS_CLOCK_SKEW)
        self.assertTrue(report["stale"])

    def test_an_empty_catalog_is_never_fresh_however_recently_it_was_written(self):
        """Zero models is a configuration failure, not a pass."""

        report = ModelCatalog([], fetched_at=stamp(60)).staleness()

        self.assertEqual(report["status"], STALENESS_EMPTY)
        self.assertTrue(report["stale"])

    def test_a_refresh_stamps_the_fetch_time(self):
        from openrouter_model_router import FakeTransport

        payload = {"data": [{"id": "vendor/x", "context_length": 8000, "pricing": {"prompt": "0.000001", "completion": "0.000002"}}]}

        catalog = ModelCatalog.refresh_from_openrouter(transport=FakeTransport([FakeTransport.json_response(payload)]))

        self.assertIsNotNone(catalog.fetched_at)
        self.assertTrue(catalog.staleness()["fresh"])

    def test_merging_an_old_catalog_in_does_not_age_a_fresh_one(self):
        fresh = ModelCatalog([priced_model()], fetched_at=stamp(60))
        old = ModelCatalog([priced_model("vendor/old")], fetched_at=stamp(400 * DAY))

        fresh.merge(old)

        self.assertTrue(fresh.staleness()["fresh"])

    def test_merging_a_fresh_catalog_into_an_old_one_refreshes_its_fetch_time(self):
        old = ModelCatalog([priced_model()], fetched_at=stamp(400 * DAY))
        fresh_stamp = stamp(60)

        old.merge(ModelCatalog([priced_model("vendor/new")], fetched_at=fresh_stamp))

        self.assertEqual(old.fetched_at, fresh_stamp)
        self.assertTrue(old.staleness()["fresh"])


class RefreshRefusesToWriteNothingTests(unittest.TestCase):
    """`refresh` writing an empty catalog is a silent, total failure.

    Every later estimate would be $0.00 with a successful-looking refresh behind
    it. The upstream call is stubbed rather than made: no credential, no network.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "catalog.json"

    @staticmethod
    def _stub(catalog):
        return unittest.mock.patch.object(
            ModelCatalog, "refresh_from_openrouter", classmethod(lambda cls, **kwargs: catalog)
        )

    def test_a_real_refresh_writes_the_catalog_and_exits_zero(self):
        incoming = ModelCatalog([priced_model()], fetched_at=stamp(0))

        with self._stub(incoming):
            code, out, _ = run_cli(["refresh", "--catalog", str(self.path)])

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["models"], 1)
        self.assertTrue(self.path.exists())

    # --- NEGATIVE CONTROL -------------------------------------------------
    def test_an_empty_upstream_payload_fails_and_leaves_the_old_catalog_alone(self):
        ModelCatalog([priced_model()], fetched_at=stamp(DAY)).save(self.path)
        before = self.path.read_text()

        with self._stub(ModelCatalog([], fetched_at=stamp(0))):
            code, _, err = run_cli(["refresh", "--catalog", str(self.path)])

        self.assertEqual(code, 1)
        self.assertIn("0 models", err)
        self.assertEqual(self.path.read_text(), before, "a failed refresh must not touch the catalog")


class CatalogStatusCommandTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "catalog.json"

    def _write(self, seconds_ago, models=None):
        when = None if seconds_ago is None else stamp(seconds_ago)
        ModelCatalog(models if models is not None else [priced_model()], fetched_at=when).save(self.path)

    def test_fresh_catalog_exits_zero(self):
        self._write(DAY)

        code, out, _ = run_cli(["catalog-status", "--catalog", str(self.path)])

        self.assertEqual(code, 0)
        self.assertIn("FRESH", out)

    # --- NEGATIVE CONTROLS ------------------------------------------------
    def test_ninety_day_old_catalog_exits_non_zero(self):
        self._write(90 * DAY)

        code, out, err = run_cli(["catalog-status", "--catalog", str(self.path)])

        self.assertEqual(code, 1)
        self.assertIn("STALE", out)
        self.assertIn("not fresh", err)

    def test_empty_catalog_exits_non_zero(self):
        self._write(60, models=[])

        code, _, err = run_cli(["catalog-status", "--catalog", str(self.path)])

        self.assertEqual(code, 1)
        self.assertIn("0 models", err)

    def test_absent_catalog_exits_non_zero(self):
        code, _, err = run_cli(["catalog-status", "--catalog", str(Path(self._tmp.name) / "nope.json")])

        self.assertEqual(code, 1)
        self.assertIn("no catalog at", err)

    def test_json_output_carries_the_status(self):
        self._write(90 * DAY)

        code, out, _ = run_cli(["catalog-status", "--catalog", str(self.path), "--json"])

        payload = json.loads(out)
        self.assertEqual(code, 1)
        self.assertEqual(payload["status"], STALENESS_STALE)
        self.assertFalse(payload["fresh"])


if __name__ == "__main__":
    unittest.main()
