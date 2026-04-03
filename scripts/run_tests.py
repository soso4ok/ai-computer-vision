#!/usr/bin/env python3
"""
AI Computer Vision Server - Test Automation Script

Runs tests and generates reports for the computer vision application.
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import os


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuite:
    """Collection of test results."""
    name: str
    results: List[TestResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)
    
    @property
    def total(self) -> int:
        return len(self.results)
    
    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return sum(r.duration for r in self.results)


class TestRunner:
    """Test automation runner."""
    
    def __init__(self, project_dir: Path, verbose: bool = False):
        self.project_dir = project_dir
        self.verbose = verbose
        self.suites: List[TestSuite] = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {
            "INFO": "\033[0;32m",
            "WARN": "\033[1;33m",
            "ERROR": "\033[0;31m",
            "DEBUG": "\033[0;34m"
        }
        color = colors.get(level, "")
        reset = "\033[0m"
        
        if level == "DEBUG" and not self.verbose:
            return
            
        print(f"{color}[{timestamp}] [{level}]{reset} {message}")
    
    def run_command(
        self,
        command: List[str],
        cwd: Optional[Path] = None,
        timeout: int = 300
    ) -> tuple:
        """Run a shell command and return (returncode, stdout, stderr)."""
        self.log(f"Running: {' '.join(command)}", "DEBUG")
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.project_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)
    
    def run_unit_tests(self) -> TestSuite:
        """Run pytest unit tests."""
        suite = TestSuite(name="Unit Tests", start_time=datetime.now())
        self.log("Running unit tests...")
        
        # Check if pytest is available
        returncode, stdout, stderr = self.run_command(
            [sys.executable, "-m", "pytest", "--version"]
        )
        
        if returncode != 0:
            self.log("pytest not installed, installing...", "WARN")
            self.run_command([sys.executable, "-m", "pip", "install", "pytest", "pytest-asyncio"])
        
        # Run pytest with JSON output
        returncode, stdout, stderr = self.run_command([
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "-q"
        ])
        
        # Parse output
        if returncode == 0:
            suite.results.append(TestResult(
                name="pytest",
                passed=True,
                duration=0,
                message="All tests passed",
                details={"output": stdout}
            ))
        else:
            suite.results.append(TestResult(
                name="pytest",
                passed=False,
                duration=0,
                message="Tests failed",
                details={"output": stdout, "errors": stderr}
            ))
        
        suite.end_time = datetime.now()
        return suite
    
    def run_lint_checks(self) -> TestSuite:
        """Run code quality checks."""
        suite = TestSuite(name="Lint Checks", start_time=datetime.now())
        self.log("Running lint checks...")
        
        # Check Python syntax
        self.log("Checking Python syntax...", "DEBUG")
        src_files = list((self.project_dir / "src").glob("*.py"))
        
        for src_file in src_files:
            start = time.time()
            returncode, stdout, stderr = self.run_command([
                sys.executable, "-m", "py_compile", str(src_file)
            ])
            duration = time.time() - start
            
            suite.results.append(TestResult(
                name=f"syntax:{src_file.name}",
                passed=returncode == 0,
                duration=duration,
                message="" if returncode == 0 else stderr
            ))
        
        # Try to run ruff or flake8 if available
        for linter in ["ruff", "flake8"]:
            returncode, _, _ = self.run_command([sys.executable, "-m", linter, "--version"])
            if returncode == 0:
                self.log(f"Running {linter}...", "DEBUG")
                start = time.time()
                returncode, stdout, stderr = self.run_command([
                    sys.executable, "-m", linter, "src/"
                ])
                duration = time.time() - start
                
                suite.results.append(TestResult(
                    name=linter,
                    passed=returncode == 0,
                    duration=duration,
                    message=stdout if returncode != 0 else "No issues found"
                ))
                break
        
        suite.end_time = datetime.now()
        return suite
    
    def run_docker_tests(self) -> TestSuite:
        """Test Docker build and basic functionality."""
        suite = TestSuite(name="Docker Tests", start_time=datetime.now())
        self.log("Running Docker tests...")
        
        # Check if Docker is available
        returncode, stdout, stderr = self.run_command(["docker", "--version"])
        if returncode != 0:
            suite.results.append(TestResult(
                name="docker_available",
                passed=False,
                duration=0,
                message="Docker is not installed or not accessible"
            ))
            suite.end_time = datetime.now()
            return suite
        
        suite.results.append(TestResult(
            name="docker_available",
            passed=True,
            duration=0,
            message=stdout.strip()
        ))
        
        # Test Dockerfile syntax (dry run)
        dockerfile = self.project_dir / "Dockerfile"
        if dockerfile.exists():
            self.log("Validating Dockerfile...", "DEBUG")
            start = time.time()
            
            # Try to parse Dockerfile
            try:
                with open(dockerfile) as f:
                    content = f.read()
                has_from = "FROM" in content.upper()
                suite.results.append(TestResult(
                    name="dockerfile_valid",
                    passed=has_from,
                    duration=time.time() - start,
                    message="Dockerfile is valid" if has_from else "Invalid Dockerfile"
                ))
            except Exception as e:
                suite.results.append(TestResult(
                    name="dockerfile_valid",
                    passed=False,
                    duration=time.time() - start,
                    message=str(e)
                ))
        
        suite.end_time = datetime.now()
        return suite
    
    def run_integration_tests(self) -> TestSuite:
        """Run integration tests."""
        suite = TestSuite(name="Integration Tests", start_time=datetime.now())
        self.log("Running integration tests...")
        
        # Test config loading
        config_file = self.project_dir / "configs" / "default.yaml"
        if config_file.exists():
            try:
                import yaml
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                
                suite.results.append(TestResult(
                    name="config_load",
                    passed=True,
                    duration=0,
                    message="Configuration loaded successfully",
                    details={"zones": len(config.get("zones", []))}
                ))
            except Exception as e:
                suite.results.append(TestResult(
                    name="config_load",
                    passed=False,
                    duration=0,
                    message=str(e)
                ))
        
        # Test module imports
        self.log("Testing module imports...", "DEBUG")
        modules_to_test = [
            ("detector", "PersonDetector"),
            ("tracker", "ObjectTracker"),
            ("dwell_tracker", "DwellTracker"),
            ("zone_manager", "ZoneManager"),
            ("api", "APIServer")
        ]
        
        # Add src to path
        src_dir = str(self.project_dir / "src")
        
        for module_name, class_name in modules_to_test:
            start = time.time()
            returncode, stdout, stderr = self.run_command([
                sys.executable, "-c",
                f"import sys; sys.path.insert(0, '{src_dir}'); "
                f"from {module_name} import {class_name}; print('OK')"
            ])
            duration = time.time() - start
            
            suite.results.append(TestResult(
                name=f"import:{module_name}",
                passed=returncode == 0 and "OK" in stdout,
                duration=duration,
                message=stderr if returncode != 0 else "Import successful"
            ))
        
        suite.end_time = datetime.now()
        return suite
    
    def run_all(self) -> List[TestSuite]:
        """Run all test suites."""
        self.log("Starting test automation")
        self.log(f"Project directory: {self.project_dir}")
        
        suites = []
        
        # Run each test suite
        suites.append(self.run_lint_checks())
        suites.append(self.run_unit_tests())
        suites.append(self.run_integration_tests())
        suites.append(self.run_docker_tests())
        
        self.suites = suites
        return suites
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("  TEST SUMMARY")
        print("=" * 60)
        
        total_passed = 0
        total_failed = 0
        
        for suite in self.suites:
            status = "✅" if suite.failed == 0 else "❌"
            print(f"\n{status} {suite.name}: {suite.passed}/{suite.total} passed ({suite.duration:.2f}s)")
            
            for result in suite.results:
                icon = "✓" if result.passed else "✗"
                color = "\033[0;32m" if result.passed else "\033[0;31m"
                reset = "\033[0m"
                print(f"  {color}{icon}{reset} {result.name}")
                if not result.passed and result.message:
                    print(f"      {result.message[:100]}")
            
            total_passed += suite.passed
            total_failed += suite.failed
        
        print("\n" + "-" * 60)
        print(f"Total: {total_passed} passed, {total_failed} failed")
        print("=" * 60)
        
        return total_failed == 0
    
    def export_report(self, output_path: Path):
        """Export test results to JSON."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "project": str(self.project_dir),
            "suites": []
        }
        
        for suite in self.suites:
            suite_data = {
                "name": suite.name,
                "passed": suite.passed,
                "failed": suite.failed,
                "total": suite.total,
                "duration": suite.duration,
                "results": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "duration": r.duration,
                        "message": r.message
                    }
                    for r in suite.results
                ]
            }
            report["suites"].append(suite_data)
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        self.log(f"Report exported to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Computer Vision Test Automation"
    )
    parser.add_argument(
        "--project-dir", "-d",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Project directory path"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Export results to JSON file"
    )
    parser.add_argument(
        "--suite", "-s",
        choices=["unit", "lint", "docker", "integration", "all"],
        default="all",
        help="Test suite to run"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner(args.project_dir, verbose=args.verbose)
    
    # Run tests
    if args.suite == "all":
        runner.run_all()
    elif args.suite == "unit":
        runner.suites = [runner.run_unit_tests()]
    elif args.suite == "lint":
        runner.suites = [runner.run_lint_checks()]
    elif args.suite == "docker":
        runner.suites = [runner.run_docker_tests()]
    elif args.suite == "integration":
        runner.suites = [runner.run_integration_tests()]
    
    # Print summary
    success = runner.print_summary()
    
    # Export if requested
    if args.output:
        runner.export_report(args.output)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
