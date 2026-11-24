import unittest
from src.url_features import FEATURE_COLUMNS, extract_url_features, make_feature_frame

class UrlFeatureTests(unittest.TestCase):
    def test_https_and_login(self):
        f = extract_url_features("https://example.com/login?x=2")
        self.assertEqual(f["is_https"], 1)
        self.assertEqual(f["has_login_token"], 1)
        self.assertEqual(f["has_ip_host"], 0)

    def test_ip_hostname(self):
        f = extract_url_features("http://127.0.0.1:8080/p")
        self.assertEqual(f["has_ip_host"], 1)
        self.assertEqual(f["has_port"], 1)

    def test_schema(self):
        frame = make_feature_frame(["example.com", "https://example.org"])
        self.assertEqual(frame.shape, (2, len(FEATURE_COLUMNS)))
        self.assertEqual(list(frame.columns), FEATURE_COLUMNS)

    def test_empty(self):
        with self.assertRaises(ValueError):
            extract_url_features("")

if __name__ == "__main__":
    unittest.main()
