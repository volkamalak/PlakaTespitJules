import os
import sys
import unittest

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detector import LicensePlateDetector

class TestDetector(unittest.TestCase):
    def test_detection_runs(self):
        detector = LicensePlateDetector() # This should load yolov8n.pt since best.pt is missing

        image_path = os.path.join(os.path.dirname(__file__), 'bus.jpg')
        if not os.path.exists(image_path):
            self.fail("Test image not found")

        processed_img, coords, duration = detector.detect(image_path)

        self.assertIsNotNone(processed_img)
        self.assertIsInstance(coords, list)
        self.assertGreater(duration, 0)

        print(f"Detected {len(coords)} objects in {duration:.4f}s")
        for c in coords:
            print(c)

        # Optional: Save output to verify visually if possible
        output_path = os.path.join(os.path.dirname(__file__), 'output_test.jpg')
        processed_img.save(output_path)
        print(f"Saved test output to {output_path}")

if __name__ == '__main__':
    unittest.main()
