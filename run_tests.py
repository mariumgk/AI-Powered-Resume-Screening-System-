#!/usr/bin/env python3
"""
Test runner script for the AI Resume System test suite.

This script provides a convenient way to run all tests or specific test categories
with proper reporting and configuration.
"""

import unittest
import sys
import os
import argparse
import time
from io import StringIO


class ColoredText:
    """ANSI color codes for test output formatting."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


class TestResult(unittest.TextTestResult):
    """Custom test result class with enhanced reporting."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_results = []
        self.start_times = {}
        self.verbosity = verbosity
    
    def startTest(self, test):
        super().startTest(test)
        self.start_times[test] = time.time()
        if self.verbosity > 1:
            self.stream.write(f"  {ColoredText.BLUE}RUNNING{ColoredText.END}: {test._testMethodName}\n")
            self.stream.flush()
    
    def addSuccess(self, test):
        super().addSuccess(test)
        duration = time.time() - self.start_times[test]
        self.test_results.append({
            'test': str(test),
            'status': 'PASS',
            'duration': duration,
            'message': None
        })
        if self.verbosity > 1:
            self.stream.write(f"  {ColoredText.GREEN}PASS{ColoredText.END}: {test._testMethodName} ({duration:.3f}s)\n")
    
    def addError(self, test, err):
        super().addError(test, err)
        duration = time.time() - self.start_times[test]
        self.test_results.append({
            'test': str(test),
            'status': 'ERROR',
            'duration': duration,
            'message': str(err[1])
        })
        if self.verbosity > 1:
            self.stream.write(f"  {ColoredText.RED}ERROR{ColoredText.END}: {test._testMethodName} ({duration:.3f}s)\n")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        duration = time.time() - self.start_times[test]
        self.test_results.append({
            'test': str(test),
            'status': 'FAIL',
            'duration': duration,
            'message': str(err[1])
        })
        if self.verbosity > 1:
            self.stream.write(f"  {ColoredText.RED}FAIL{ColoredText.END}: {test._testMethodName} ({duration:.3f}s)\n")
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.test_results.append({
            'test': str(test),
            'status': 'SKIP',
            'duration': 0,
            'message': reason
        })
        if self.verbosity > 1:
            self.stream.write(f"  {ColoredText.YELLOW}SKIP{ColoredText.END}: {test._testMethodName} - {reason}\n")


class ColoredTestRunner(unittest.TextTestRunner):
    """Custom test runner with colored output."""
    
    def __init__(self, **kwargs):
        kwargs['resultclass'] = TestResult
        super().__init__(**kwargs)


def discover_and_run_tests(test_pattern=None, test_category=None, verbosity=2):
    """
    Discover and run tests with optional filtering.
    
    Args:
        test_pattern: Specific test pattern to match (e.g., 'test_flask_app')
        test_category: Category of tests to run ('unit', 'integration', 'performance', 'frontend')
        verbosity: Verbosity level (0-2)
    
    Returns:
        TestResult object with test results
    """
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Discover tests
    loader = unittest.TestLoader()
    
    if test_pattern:
        # Run specific test file or pattern
        if test_pattern.endswith('.py'):
            test_pattern = test_pattern[:-3]
        
        try:
            suite = loader.loadTestsFromName(f'tests.{test_pattern}')
        except (ImportError, AttributeError):
            # Try to find test files matching pattern
            test_dir = os.path.join(project_root, 'tests')
            suite = loader.discover(test_dir, pattern=f'{test_pattern}*.py')
    elif test_category:
        # Run tests by category
        test_dir = os.path.join(project_root, 'tests')
        category_patterns = {
            'unit': 'test_*.py',
            'integration': 'test_flask_inference_contract.py',
            'performance': 'test_performance.py',
            'frontend': 'test_frontend_ui.py',
            'flask': 'test_flask_inference_contract.py',
            'inference': 'test_inference.py'
        }
        
        pattern = category_patterns.get(test_category, 'test_*.py')
        suite = loader.discover(test_dir, pattern=pattern)
    else:
        # Run all tests
        test_dir = os.path.join(project_root, 'tests')
        suite = loader.discover(test_dir, pattern='test_*.py')
    
    # Run tests with custom runner
    runner = ColoredTestRunner(verbosity=verbosity, stream=sys.stdout)
    result = runner.run(suite)
    
    return result


def print_summary(result):
    """Print a detailed test summary."""
    print(f"\n{ColoredText.BOLD}{'='*60}{ColoredText.END}")
    print(f"{ColoredText.BOLD}TEST SUMMARY{ColoredText.END}")
    print(f"{ColoredText.BOLD}{'='*60}{ColoredText.END}")
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skips = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skips
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"{ColoredText.GREEN}Passed: {passed}{ColoredText.END}")
    print(f"{ColoredText.RED}Failed: {failures}{ColoredText.END}")
    print(f"{ColoredText.RED}Errors: {errors}{ColoredText.END}")
    print(f"{ColoredText.YELLOW}Skipped: {skips}{ColoredText.END}")
    
    # Calculate success rate
    if total_tests > 0:
        success_rate = (passed / total_tests) * 100
        status_color = ColoredText.GREEN if success_rate >= 90 else ColoredText.YELLOW if success_rate >= 70 else ColoredText.RED
        print(f"\nSuccess Rate: {status_color}{success_rate:.1f}%{ColoredText.END}")
    
    # Performance summary if available
    if hasattr(result, 'test_results') and result.test_results:
        durations = [r['duration'] for r in result.test_results if r['status'] in ['PASS', 'FAIL', 'ERROR']]
        if durations:
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            total_duration = sum(durations)
            
            print(f"\n{ColoredText.BLUE}Performance Metrics:{ColoredText.END}")
            print(f"  Total Time: {total_duration:.2f}s")
            print(f"  Average Time: {avg_duration:.3f}s")
            print(f"  Max Time: {max_duration:.3f}s")
    
    # Show failures and errors
    if failures > 0:
        print(f"\n{ColoredText.RED}FAILURES:{ColoredText.END}")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if errors > 0:
        print(f"\n{ColoredText.RED}ERRORS:{ColoredText.END}")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    print(f"\n{ColoredText.BOLD}{'='*60}{ColoredText.END}")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description='AI Resume System Test Runner')
    parser.add_argument('--pattern', '-p', help='Specific test pattern to run (e.g., test_flask_app)')
    parser.add_argument('--category', '-c', choices=['unit', 'integration', 'performance', 'frontend', 'flask', 'inference'],
                       help='Category of tests to run')
    parser.add_argument('--verbosity', '-v', type=int, choices=[0, 1, 2], default=2,
                       help='Verbosity level (0=quiet, 1=normal, 2=verbose)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available test files and exit')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Run quick tests only (exclude performance tests)')
    
    args = parser.parse_args()
    
    # List available tests if requested
    if args.list:
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests')
        print(f"{ColoredText.BOLD}Available Test Files:{ColoredText.END}")
        
        for file in os.listdir(test_dir):
            if file.startswith('test_') and file.endswith('.py'):
                test_name = file[:-3]  # Remove .py extension
                description = {
                    'test_flask_inference_contract': 'Flask inference route contract tests (mocked + optional integration)',
                    'test_inference': 'Inference engine and smoke tests',
                    'test_frontend_ui': 'Frontend UI components and templates (opt-in via RUN_FRONTEND_UI_TESTS=1)',
                    'test_performance': 'Performance and load tests (opt-in via RUN_PERFORMANCE_TESTS=1)'
                }.get(test_name, 'General tests')
                
                print(f"  {ColoredText.BLUE}{test_name}{ColoredText.END}: {description}")
        
        print(f"\n{ColoredText.YELLOW}Categories:{ColoredText.END}")
        print(f"  {ColoredText.GREEN}unit{ColoredText.END}: All unit tests")
        print(f"  {ColoredText.GREEN}integration{ColoredText.END}: Integration tests only")
        print(f"  {ColoredText.GREEN}performance{ColoredText.END}: Performance tests only")
        print(f"  {ColoredText.GREEN}frontend{ColoredText.END}: Frontend UI tests only")
        print(f"  {ColoredText.GREEN}flask{ColoredText.END}: Flask app tests only")
        print(f"  {ColoredText.GREEN}inference{ColoredText.END}: Inference tests only")
        return
    
    # Adjust for quick mode (exclude performance tests)
    if args.quick:
        if args.category == 'performance':
            print(f"{ColoredText.YELLOW}Warning: Cannot run performance tests in quick mode{ColoredText.END}")
            return
        elif args.category is None:
            args.category = 'unit'  # Default to unit tests in quick mode
    
    print(f"{ColoredText.BOLD}AI Resume System Test Suite{ColoredText.END}")
    print(f"{ColoredText.BOLD}{'='*60}{ColoredText.END}")
    
    if args.pattern:
        print(f"Running tests matching pattern: {ColoredText.BLUE}{args.pattern}{ColoredText.END}")
    elif args.category:
        print(f"Running {ColoredText.BLUE}{args.category}{ColoredText.END} tests")
    else:
        print(f"Running {ColoredText.BLUE}all{ColoredText.END} tests")
    
    if args.quick:
        print(f"Mode: {ColoredText.YELLOW}Quick tests only{ColoredText.END}")
    
    print(f"{ColoredText.BOLD}{'='*60}{ColoredText.END}\n")
    
    # Run tests
    start_time = time.time()
    result = discover_and_run_tests(
        test_pattern=args.pattern,
        test_category=args.category,
        verbosity=args.verbosity
    )
    total_time = time.time() - start_time
    
    # Print summary
    print_summary(result)
    
    print(f"\nTotal execution time: {total_time:.2f}s")
    
    # Exit with appropriate code
    if result.wasSuccessful():
        print(f"\n{ColoredText.GREEN}All tests passed!{ColoredText.END}")
        sys.exit(0)
    else:
        print(f"\n{ColoredText.RED}Some tests failed.{ColoredText.END}")
        sys.exit(1)


if __name__ == '__main__':
    main()
