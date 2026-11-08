"""Every number quoted in Section V and in the ReadMe, read from results/results.json (written by
generate_results.py) and compared with the printed value (tolerance: half a unit in the last
printed digit).  This is the test a reader runs after reproducing the results."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stats                                # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "results.json")


@unittest.skipUnless(os.path.exists(RESULTS), "results/results.json not found: run generate_results.py first")
class TestPrintedNumbers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = stats.rows(stats.load(RESULTS))

    def test_every_row_passes(self):
        failed = [(label, value, printed) for label, value, printed, ok in self.table if ok is False]
        self.assertEqual(failed, [], "\n".join(f"{l}: computed {v!r}, printed {p}" for l, v, p in failed))

    def test_table_is_complete(self):
        self.assertGreaterEqual(sum(1 for r in self.table if r[3] is not None), 60)


if __name__ == "__main__":
    unittest.main()
