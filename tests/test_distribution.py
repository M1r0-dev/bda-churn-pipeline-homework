"""Static distribution contract; real Docker build/run is recorded separately."""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DistributionTests(unittest.TestCase):
    def test_python_dependencies_are_pinned(self):
        path = ROOT / "requirements.txt"
        self.assertTrue(path.is_file(), "requirements.txt must exist")
        requirements = dict(
            line.split("==") for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        )
        expected = {
            "numpy": "2.1.3", "pandas": "2.3.3", "scikit-learn": "1.6.1",
            "pyarrow": "24.0.0", "joblib": "1.4.2",
        }
        for package, version in expected.items():
            self.assertEqual(requirements.get(package), version)
        self.assertTrue(all(re.fullmatch(r"\d+\.\d+(?:\.\d+)?(?:\.post\d+)?", v)
                            for v in requirements.values()))

    def test_container_runs_public_cli_with_bundled_inputs(self):
        path = ROOT / "Dockerfile"
        self.assertTrue(path.is_file(), "Dockerfile must exist")
        lines = path.read_text().splitlines()
        self.assertRegex(lines[0], r"^FROM python:3\.11-slim@sha256:[0-9a-f]{64}$")
        self.assertIn("WORKDIR /app", lines)
        self.assertIn("COPY requirements.txt .", lines)
        self.assertIn("COPY --chmod=644 pipeline.py config.json ./", lines)
        self.assertIn("COPY data/ ./data/", lines)
        command = next(line[4:] for line in lines if line.startswith("CMD "))
        self.assertEqual(json.loads(command), ["python", "pipeline.py", "--config", "config.json"])
        self.assertTrue(any("--only-binary=:all:" in line for line in lines))

    def test_build_context_excludes_generated_and_private_files(self):
        path = ROOT / ".dockerignore"
        self.assertTrue(path.is_file(), ".dockerignore must exist")
        ignored = set(path.read_text().splitlines())
        self.assertEqual(ignored, {
            "*", "!Dockerfile", "!requirements.txt", "!pipeline.py",
            "!config.json", "!data/", "!data/**",
        })


if __name__ == "__main__":
    unittest.main()
