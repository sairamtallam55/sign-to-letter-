import unittest
from preprocess import (
    DUAL_HAND_FEATURE_COUNT,
    SINGLE_HAND_FEATURE_COUNT,
    normalize_single_hand,
    session_wise_split,
    to_canonical_features,
    validate_landmarks,
)


class TestPythonPreprocess(unittest.TestCase):
    def _sample_hand(self):
        return [(0.5 + i * 0.01, 0.5 - i * 0.015, 0.0) for i in range(21)]

    def test_validate_and_normalize(self):
        hand = self._sample_hand()
        self.assertTrue(validate_landmarks(hand))
        feats = normalize_single_hand(hand)
        self.assertEqual(len(feats), SINGLE_HAND_FEATURE_COUNT)
        self.assertEqual(feats[0], 0.0)

        scaled = [(x * 2.0 + 0.1, y * 2.0 - 0.2, z * 2.0) for (x, y, z) in hand]
        feats_scaled = normalize_single_hand(scaled)
        for a, b in zip(feats, feats_scaled):
            self.assertAlmostEqual(a, b, places=4)

    def test_canonical_padding(self):
        padded = to_canonical_features([0.1] * SINGLE_HAND_FEATURE_COUNT)
        self.assertEqual(len(padded), DUAL_HAND_FEATURE_COUNT)
        self.assertEqual(padded[-1], 0.0)

    def test_session_wise_split(self):
        records = [
            {"label": lbl, "session_id": f"s{s}", "features": [0.0] * DUAL_HAND_FEATURE_COUNT}
            for lbl in ("A", "B", "C")
            for s in range(5)
        ]
        train, val, test, strategy = session_wise_split(records)
        self.assertEqual(strategy, "session-wise")
        train_sessions = {r["session_id"] for r in train}
        test_sessions = {r["session_id"] for r in test}
        self.assertTrue(train_sessions.isdisjoint(test_sessions))


if __name__ == "__main__":
    unittest.main()
