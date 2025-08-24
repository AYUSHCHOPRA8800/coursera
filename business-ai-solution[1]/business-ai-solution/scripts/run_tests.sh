#!/bin/bash

# Business AI Solution - Test Runner Script
# This script runs all unit tests with proper isolation and reporting

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTS_DIR="$PROJECT_ROOT/tests"
REPORTS_DIR="$PROJECT_ROOT/reports"
COVERAGE_DIR="$PROJECT_ROOT/coverage"

# Create necessary directories
mkdir -p "$REPORTS_DIR"
mkdir -p "$COVERAGE_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Business AI Solution - Test Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print section headers
print_section() {
    echo -e "${YELLOW}$1${NC}"
    echo "----------------------------------------"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python and pip installation
print_section "Checking Environment"
if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

if ! command_exists pip3; then
    echo -e "${RED}Error: pip3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python 3: $(python3 --version)${NC}"
echo -e "${GREEN}✓ pip3: $(pip3 --version)${NC}"

# Install test dependencies if not already installed
print_section "Installing Test Dependencies"
if [ ! -f "$PROJECT_ROOT/requirements.txt" ]; then
    echo -e "${YELLOW}Warning: requirements.txt not found, creating basic test requirements${NC}"
    cat > "$PROJECT_ROOT/requirements.txt" << EOF
fastapi==0.104.1
uvicorn==0.24.0
pytest==7.4.3
pytest-cov==4.1.0
pytest-mock==3.12.0
pandas==2.1.3
numpy==1.25.2
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
requests==2.31.0
httpx==0.25.2
EOF
fi

pip3 install -r "$PROJECT_ROOT/requirements.txt" --quiet
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Set up test environment
print_section "Setting Up Test Environment"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export TESTING="true"
export LOG_LEVEL="WARNING"  # Reduce log noise during tests

# Create test data directories
mkdir -p "$PROJECT_ROOT/test_data"
mkdir -p "$PROJECT_ROOT/test_models"
mkdir -p "$PROJECT_ROOT/test_logs"

echo -e "${GREEN}✓ Test environment configured${NC}"

# Function to run tests with coverage
run_test_suite() {
    local test_file="$1"
    local test_name="$2"
    local coverage_file="$3"
    
    echo -e "${BLUE}Running $test_name...${NC}"
    
    # Run tests with coverage
    python3 -m pytest "$test_file" \
        --cov="$PROJECT_ROOT/app" \
        --cov-report=term-missing \
        --cov-report=html:"$COVERAGE_DIR/$coverage_file" \
        --cov-report=xml:"$COVERAGE_DIR/$coverage_file.xml" \
        --junit-xml="$REPORTS_DIR/${test_name}_results.xml" \
        --tb=short \
        -v
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ $test_name passed${NC}"
    else
        echo -e "${RED}✗ $test_name failed${NC}"
    fi
    
    return $exit_code
}

# Run individual test suites
print_section "Running Unit Tests"

# Track overall test results
overall_success=true

# Test 1: API Tests
if [ -f "$TESTS_DIR/test_api.py" ]; then
    if run_test_suite "$TESTS_DIR/test_api.py" "API Tests" "api_coverage"; then
        echo -e "${GREEN}✓ API tests completed successfully${NC}"
    else
        echo -e "${RED}✗ API tests failed${NC}"
        overall_success=false
    fi
else
    echo -e "${RED}✗ API test file not found${NC}"
    overall_success=false
fi

echo ""

# Test 2: Model Tests
if [ -f "$TESTS_DIR/test_model.py" ]; then
    if run_test_suite "$TESTS_DIR/test_model.py" "Model Tests" "model_coverage"; then
        echo -e "${GREEN}✓ Model tests completed successfully${NC}"
    else
        echo -e "${RED}✗ Model tests failed${NC}"
        overall_success=false
    fi
else
    echo -e "${RED}✗ Model test file not found${NC}"
    overall_success=false
fi

echo ""

# Test 3: Logging Tests
if [ -f "$TESTS_DIR/test_logging.py" ]; then
    if run_test_suite "$TESTS_DIR/test_logging.py" "Logging Tests" "logging_coverage"; then
        echo -e "${GREEN}✓ Logging tests completed successfully${NC}"
    else
        echo -e "${RED}✗ Logging tests failed${NC}"
        overall_success=false
    fi
else
    echo -e "${RED}✗ Logging test file not found${NC}"
    overall_success=false
fi

echo ""

# Run all tests together for overall coverage
print_section "Running All Tests Together"
echo -e "${BLUE}Running comprehensive test suite...${NC}"

python3 -m pytest "$TESTS_DIR/" \
    --cov="$PROJECT_ROOT/app" \
    --cov-report=term-missing \
    --cov-report=html:"$COVERAGE_DIR/overall_coverage" \
    --cov-report=xml:"$COVERAGE_DIR/overall_coverage.xml" \
    --junit-xml="$REPORTS_DIR/overall_results.xml" \
    --tb=short \
    -v

overall_exit_code=$?

# Generate test summary
print_section "Test Summary"

if [ $overall_exit_code -eq 0 ] && [ "$overall_success" = true ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  ALL TESTS PASSED! 🎉${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  SOME TESTS FAILED! ❌${NC}"
    echo -e "${RED}========================================${NC}"
fi

# Display coverage summary
if [ -f "$COVERAGE_DIR/overall_coverage.xml" ]; then
    echo ""
    echo -e "${BLUE}Coverage Summary:${NC}"
    python3 -c "
import xml.etree.ElementTree as ET
try:
    tree = ET.parse('$COVERAGE_DIR/overall_coverage.xml')
    root = tree.getroot()
    for package in root.findall('.//package'):
        name = package.get('name', 'Unknown')
        line_rate = float(package.get('line-rate', 0)) * 100
        branch_rate = float(package.get('branch-rate', 0)) * 100
        print(f'  {name}: {line_rate:.1f}% line coverage, {branch_rate:.1f}% branch coverage')
except Exception as e:
    print(f'  Could not parse coverage report: {e}')
"
fi

# Display test reports location
echo ""
echo -e "${BLUE}Test Reports Generated:${NC}"
echo "  - Coverage HTML: $COVERAGE_DIR/overall_coverage/index.html"
echo "  - Coverage XML: $COVERAGE_DIR/overall_coverage.xml"
echo "  - JUnit XML: $REPORTS_DIR/overall_results.xml"

# Clean up test environment
print_section "Cleaning Up"
rm -rf "$PROJECT_ROOT/test_data"
rm -rf "$PROJECT_ROOT/test_models"
rm -rf "$PROJECT_ROOT/test_logs"

echo -e "${GREEN}✓ Test environment cleaned up${NC}"

# Final status
echo ""
if [ $overall_exit_code -eq 0 ] && [ "$overall_success" = true ]; then
    echo -e "${GREEN}🎉 All tests completed successfully!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Please check the output above.${NC}"
    exit 1
fi
