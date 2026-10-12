#!/bin/bash

# Script to run all inductor tests one by one and document results
# Usage: cd torch-spyre-git-main && bash ../mock-device/run_all_inductor_tests.sh

export TORCH_SPYRE_MOCK_DEVICE=1

# Output files
RESULTS_FILE="test_results_$(date +%Y%m%d_%H%M%S).txt"
FAILURES_DIR="test_failures_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$FAILURES_DIR"

# Files to skip unconditionally
SKIP_FILES=(
)

# Per-file memory floor overrides (MB). Files not listed use COLLECTION_MEM_FLOOR.
declare -A FILE_MEM_FLOOR=(
    ["test_inductor_ops.py"]=200
    ["test_inductor_ops_lx_planning.py"]=200
)

# Files that are too large to run in one pytest invocation — run in batches instead.
# Value = number of tests per batch. Set to 0 to disable batch mode for a file.
#
# Everything runs single-process. pytest-xdist was tried here and made things
# strictly worse: importing test_inductor_ops.py allocates its test inputs eagerly
# (~6.7 GB resident), so every xdist worker pays that cost in full and N workers
# multiply it. Four workers against a 12.5 GB container is an instant OOM. One
# process fits; parallelism does not.
declare -A FILE_BATCH_SIZE=(
    ["test_inductor_ops.py"]=0
    ["test_inductor_ops_lx_planning.py"]=0
)

# ---------------------------------------------------------------------------
# Memory helpers
# Returns available RAM in MB from /proc/meminfo (Linux inside container).
_free_mem_mb() {
    awk '/^MemAvailable:/ { printf "%d", $2/1024 }' /proc/meminfo 2>/dev/null || echo 0
}

# Minimum free MB required to safely attempt collection of a test file.
# Only needs to be high enough to prevent collection-time OOM kills on the
# heaviest files (test_inductor_ops.py etc.). Most files need far less.
# Tune with env var: INDUCTOR_COLLECTION_MEM_MB=N bash run_all_inductor_tests.sh
COLLECTION_MEM_FLOOR="${INDUCTOR_COLLECTION_MEM_MB:-200}"

echo "========================================" | tee "$RESULTS_FILE"
echo "Inductor Tests Results" | tee -a "$RESULTS_FILE"
echo "Started: $(date)" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Initialize counters
total_files=0
passed_files=0
failed_files=0
skipped_files=0
cannot_run_files=0
total_tests=0
passed_tests=0
failed_tests=0
skipped_tests=0   # _MOCK_TOO_HEAVY only (from conftest MOCK_COUNTS)
xfailed_tests=0   # _MOCK_KNOWN_FAILURES only (from conftest MOCK_COUNTS)

# Temp file for collecting unique error messages across all failures
UNIQUE_ERRORS_FILE=$(mktemp /tmp/inductor_unique_errors.XXXXXX)

# Function to check if file should be skipped
should_skip() {
    local filename=$(basename "$1")
    for skip in "${SKIP_FILES[@]}"; do
        if [ "$filename" = "$skip" ]; then
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# Parse counts from a pytest output string.
# _x = _MOCK_KNOWN_FAILURES, _s = _MOCK_TOO_HEAVY, both reported by conftest.
# Sets: _p (passed), _f (unexpected failures), _s (too-heavy), _x (known-failures)
_parse_counts() {
    local out="$1"
    _p=$(echo "$out" | grep -oE "[0-9]+ passed"  | grep -oE "[0-9]+" | head -1)
    _f=$(echo "$out" | grep -oE "[0-9]+ failed"  | grep -oE "[0-9]+" | head -1)
    # _x / _s come from conftest's MOCK_COUNTS line, NOT from pytest's aggregate
    # "N xfailed" / "N skipped". pytest's numbers fold in the test suite's own
    # xfails (PARAMS "expect_fail", *_known_xfail_* tests — ~277 of them) and the
    # pytest.skip() calls inside test bodies (~50, mostly in
    # test_inductor_normalization_scalars.py). Those are upstream torch-spyre
    # decisions and have nothing to do with the mock device, so counting them
    # here would badly overstate what the mock cannot do.
    _x=$(echo "$out" | grep -oE "MOCK_COUNTS known_failures=[0-9]+" | grep -oE "[0-9]+" | head -1)
    _s=$(echo "$out" | grep -oE "too_heavy=[0-9]+" | grep -oE "[0-9]+" | head -1)
    _p=${_p:-0}; _f=${_f:-0}; _s=${_s:-0}; _x=${_x:-0}
}

# ---------------------------------------------------------------------------
# Helper: run a list of test IDs in batches of BATCH_SIZE, accumulating
# pass/fail/skip counts into file_passed/file_failed/file_skipped.
# Also appends combined output to the variable named in $4 (by nameref).
# Usage: _run_in_batches <test_id_list_file> <batch_size> <test_name> <output_var>
_run_in_batches() {
    local ids_file="$1"
    local batch_size="$2"
    local tname="$3"
    local -n _out_ref=$4   # nameref to accumulate output

    local batch=()
    local batch_num=0
    local killed_count=0

    while IFS= read -r tid || [ ${#batch[@]} -gt 0 ]; do
        # Add to current batch (skip blank lines)
        if [ -n "$tid" ]; then
            batch+=("$tid")
        fi

        # Flush when batch is full OR we've consumed all input (EOF with leftover)
        if [ ${#batch[@]} -eq "$batch_size" ] || { [ -z "$tid" ] && [ ${#batch[@]} -gt 0 ]; }; then
            [ ${#batch[@]} -eq 0 ] && break
            batch_num=$((batch_num + 1))
            echo "  🔹 Batch $batch_num (${#batch[@]} tests)..." | tee -a "$RESULTS_FILE"

            b_out=$(python3 -m pytest "${batch[@]}" -v -p no:cacheprovider 2>&1)
            b_exit=$?
            _out_ref+="$b_out"$'\n'

            b_has_summary=0
            if echo "$b_out" | grep -qE "=+ [0-9]+ (passed|failed|error)"; then
                b_has_summary=1
            fi

            if [ "$b_exit" -ne 0 ] && [ "$b_has_summary" -eq 0 ]; then
                # Batch was killed — retry each test individually (no parallelism, minimal memory)
                echo "  ⚠️  Batch $batch_num killed (exit $b_exit) — retrying individually (sequential)..." | tee -a "$RESULTS_FILE"
                for tid2 in "${batch[@]}"; do
                    t_out=$(python3 -m pytest "$tid2" -v -p no:cacheprovider 2>&1)
                    t_exit=$?
                    _out_ref+="$t_out"$'\n'
                    t_has_summary=0
                    if echo "$t_out" | grep -qE "=+ [0-9]+ (passed|failed|error)"; then
                        t_has_summary=1
                    fi
                    if [ "$t_exit" -ne 0 ] && [ "$t_has_summary" -eq 0 ]; then
                        killed_count=$((killed_count + 1))
                        echo "    💀 KILLED: $tid2" | tee -a "$RESULTS_FILE"
                        echo "$tid2: Killed by OOM/signal (exit $t_exit)" >> "$FAILURES_DIR/$tname.log"
                    else
                        _parse_counts "$t_out"
                        file_passed=$((file_passed + _p))
                        file_failed=$((file_failed + _f))
                        file_skipped=$((file_skipped + _s))
                        file_xfailed=$((file_xfailed + _x))
                    fi
                done
            else
                _parse_counts "$b_out"
                file_passed=$((file_passed + _p))
                file_failed=$((file_failed + _f))
                file_skipped=$((file_skipped + _s))
                file_xfailed=$((file_xfailed + _x))
            fi

            batch=()
        fi
        # Stop loop at EOF (empty $tid means we finished the file)
        [ -z "$tid" ] && break
    done < "$ids_file"

    if [ "$killed_count" -gt 0 ]; then
        echo "  ⚠️  $killed_count individual test(s) killed (OOM/signal)" | tee -a "$RESULTS_FILE"
    fi
}

# Find all test files in tests/inductor/
for test_file in tests/inductor/test_*.py; do
    if [ -f "$test_file" ]; then
        test_name=$(basename "$test_file")
        
        # Check if file should be skipped
        if should_skip "$test_file"; then
            skipped_files=$((skipped_files + 1))
            # Count tests via collect-only; if collection fails/OOMs, count is 0.
            file_test_count=$(python3 -m pytest "$test_file" --collect-only -q \
                --no-header -p no:cacheprovider 2>/dev/null \
                | grep -c "::" 2>/dev/null || echo 0)
            file_test_count=${file_test_count:-0}
            total_tests=$((total_tests + file_test_count))
            skipped_tests=$((skipped_tests + file_test_count))
            echo "⏭️  SKIPPED: $test_name ($file_test_count tests)" | tee -a "$RESULTS_FILE"
            echo "" | tee -a "$RESULTS_FILE"
            continue
        fi
        
        total_files=$((total_files + 1))

        echo "----------------------------------------" | tee -a "$RESULTS_FILE"
        echo "Running: $test_name" | tee -a "$RESULTS_FILE"
        echo "----------------------------------------" | tee -a "$RESULTS_FILE"

        # Pre-flight memory check
        free_mb=$(_free_mem_mb)
        mem_floor="${FILE_MEM_FLOOR[$test_name]:-$COLLECTION_MEM_FLOOR}"
        if [ "$free_mb" -lt "$mem_floor" ]; then
            cannot_run_files=$((cannot_run_files + 1))
            echo "  ⚠️  Insufficient memory to attempt collection" \
                 "(${free_mb} MB free, need ${mem_floor} MB)" | tee -a "$RESULTS_FILE"
            echo "⚠️  CANNOT RUN: $test_name (insufficient memory: ${free_mb} MB free)" | tee -a "$RESULTS_FILE"
            echo "" | tee -a "$RESULTS_FILE"
            continue
        fi

        batch_size="${FILE_BATCH_SIZE[$test_name]:-0}"

        if [ "$batch_size" -gt 0 ]; then
            # ---------------------------------------------------------------
            # BATCH MODE: collect test IDs first, then run in fixed-size chunks
            # ---------------------------------------------------------------
            echo "  📊 Collecting test IDs (batch mode, $batch_size per batch)..." | tee -a "$RESULTS_FILE"
            ids_tmpfile=$(mktemp /tmp/inductor_ids.XXXXXX)
            python3 -m pytest "$test_file" --collect-only -q --no-header -p no:cacheprovider 2>/dev/null \
                | grep "::" | sed 's/ (.*)//' > "$ids_tmpfile"

            test_count=$(wc -l < "$ids_tmpfile")
            test_count=${test_count//[[:space:]]/}

            if [ "$test_count" -eq 0 ]; then
                rm -f "$ids_tmpfile"
                cannot_run_files=$((cannot_run_files + 1))
                echo "  ⚠️  collect-only returned no tests (possible OOM during collection)" | tee -a "$RESULTS_FILE"
                echo "⚠️  CANNOT RUN: $test_name (collection failed)" | tee -a "$RESULTS_FILE"
                echo "" | tee -a "$RESULTS_FILE"
                continue
            fi

            sed -i "s/Collecting test IDs (batch mode.*$/Detected $test_count tests (batch mode, $batch_size per batch)/" "$RESULTS_FILE" 2>/dev/null || true

            file_passed=0; file_failed=0; file_skipped=0; file_xfailed=0
            output=""
            _run_in_batches "$ids_tmpfile" "$batch_size" "$test_name" output
            rm -f "$ids_tmpfile"

            if [ "$file_failed" -gt 0 ]; then exit_code=1; else exit_code=0; fi

        else
            # ---------------------------------------------------------------
            # NORMAL MODE: single pytest invocation
            # ---------------------------------------------------------------
            output=$(python3 -m pytest "$test_file" -v -p no:cacheprovider 2>&1)
            exit_code=$?

            test_count=$(echo "$output" | grep -oE "collected [0-9]+ item" | grep -oE "[0-9]+" | head -1)
            test_count=${test_count:-0}

            _parse_counts "$output"
            file_passed=$_p
            file_failed=$_f
            file_skipped=$_s
            file_xfailed=$_x

            has_summary=0
            if echo "$output" | grep -qE "=+ [0-9]+ (passed|failed|error)"; then
                has_summary=1
            fi

            if [ "$exit_code" -ne 0 ] && [ "$has_summary" -eq 0 ]; then
                if [ "$test_count" -eq 0 ]; then
                    cannot_run_files=$((cannot_run_files + 1))
                    echo "  ⚠️  Killed during collection (exit $exit_code, ${free_mb} MB free)" | tee -a "$RESULTS_FILE"
                    echo "⚠️  CANNOT RUN: $test_name (killed during collection)" | tee -a "$RESULTS_FILE"
                    echo "" | tee -a "$RESULTS_FILE"
                    continue
                else
                    # Killed mid-run — retry each test individually
                    echo "  ⚠️  Killed mid-run (exit $exit_code) after $test_count collected — retrying one-by-one..." | tee -a "$RESULTS_FILE"
                    test_ids=$(python3 -m pytest "$test_file" --collect-only -q --no-header -p no:cacheprovider 2>/dev/null \
                        | grep "::" | sed 's/ (.*)//')
                    if [ -z "$test_ids" ]; then
                        cannot_run_files=$((cannot_run_files + 1))
                        echo "  ⚠️  collect-only also failed — treating as cannot-run" | tee -a "$RESULTS_FILE"
                        echo "⚠️  CANNOT RUN: $test_name (killed during collection retry)" | tee -a "$RESULTS_FILE"
                        echo "" | tee -a "$RESULTS_FILE"
                        continue
                    fi
                    file_passed=0; file_failed=0; file_skipped=0; file_xfailed=0
                    killed_count=0
                    combined_output=""
                    while IFS= read -r tid; do
                        [ -z "$tid" ] && continue
                        t_out=$(python3 -m pytest "$tid" -v -p no:cacheprovider 2>&1)
                        t_exit=$?
                        combined_output+="$t_out"$'\n'
                        t_has_summary=0
                        if echo "$t_out" | grep -qE "=+ [0-9]+ (passed|failed|error)"; then t_has_summary=1; fi
                        if [ "$t_exit" -ne 0 ] && [ "$t_has_summary" -eq 0 ]; then
                            killed_count=$((killed_count + 1))
                            echo "    💀 KILLED: $tid" | tee -a "$RESULTS_FILE"
                            echo "$tid: Killed by OOM/signal (exit $t_exit)" >> "$FAILURES_DIR/$test_name.log"
                        else
                            _parse_counts "$t_out"
                            file_passed=$((file_passed + _p))
                            file_failed=$((file_failed + _f))
                            file_skipped=$((file_skipped + _s))
                            file_xfailed=$((file_xfailed + _x))
                        fi
                    done <<< "$test_ids"
                    [ "$killed_count" -gt 0 ] && echo "  ⚠️  $killed_count test(s) killed during retry" | tee -a "$RESULTS_FILE"
                    output="$combined_output"
                    if [ "$file_failed" -gt 0 ] || [ "$killed_count" -gt 0 ]; then exit_code=1; else exit_code=0; fi
                fi
            fi
        fi

        # Total = passed + _MOCK_KNOWN_FAILURES + _MOCK_TOO_HEAVY (+ any hard
        # failure). The suite's own xfails and in-body skips are deliberately
        # excluded: they are not tests the mock device is failing or refusing.
        total_tests=$((total_tests + file_passed + file_failed + file_xfailed + file_skipped))
        passed_tests=$((passed_tests + file_passed))
        failed_tests=$((failed_tests + file_failed))
        skipped_tests=$((skipped_tests + file_skipped))
        xfailed_tests=$((xfailed_tests + file_xfailed))

        if [ $exit_code -eq 0 ]; then
            passed_files=$((passed_files + 1))
            echo "✅ PASSED: $test_name ($file_passed passed)" | tee -a "$RESULTS_FILE"
        else
            failed_files=$((failed_files + 1))
            echo "❌ FAILED: $test_name ($file_passed passed, $file_failed UNEXPECTED)" | tee -a "$RESULTS_FILE"

            # Save full output for this file's failures
            echo "$output" > "$FAILURES_DIR/$test_name.log"

            # Extract one-liner error messages from FAILED lines.
            echo "$output" | grep "^FAILED" | sed 's/^FAILED [^ ]* - //' >> "$UNIQUE_ERRORS_FILE"
        fi
        
        echo "" | tee -a "$RESULTS_FILE"
    fi
done

# Print summary
echo "========================================" | tee -a "$RESULTS_FILE"
echo "SUMMARY" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
echo "Total test files found: $((total_files + skipped_files))" | tee -a "$RESULTS_FILE"
echo "  ⏭️  Skipped files: $skipped_files" | tee -a "$RESULTS_FILE"
echo "  📝 Tested files: $total_files" | tee -a "$RESULTS_FILE"
echo "    ✅ Passed files: $passed_files" | tee -a "$RESULTS_FILE"
echo "    ❌ Failed files: $failed_files" | tee -a "$RESULTS_FILE"
echo "    ⚠️  Cannot run (insufficient memory): $cannot_run_files" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
# Total counts only what the mock device actually accounts for:
#   passed + _MOCK_KNOWN_FAILURES (failed) + _MOCK_TOO_HEAVY (skipped)
# The suite's own xfails (~277: PARAMS "expect_fail", *_known_xfail_* tests) and
# its in-body pytest.skip() calls (~50) are excluded — they are upstream
# torch-spyre decisions that behave identically on real hardware.
echo "Total individual tests: $total_tests" | tee -a "$RESULTS_FILE"
echo "  ✅ Passed tests: $passed_tests" | tee -a "$RESULTS_FILE"
echo "  ❌ Failed tests (_MOCK_KNOWN_FAILURES): $xfailed_tests" | tee -a "$RESULTS_FILE"
echo "  ⏭️  Skipped tests (_MOCK_TOO_HEAVY): $skipped_tests" | tee -a "$RESULTS_FILE"
if [ "$failed_tests" -gt 0 ]; then
    echo "  🔥 Unexpected failures (not on any list): $failed_tests" | tee -a "$RESULTS_FILE"
fi
echo "" | tee -a "$RESULTS_FILE"

# Calculate percentages
if [ $total_files -gt 0 ]; then
    pass_rate=$((passed_files * 100 / total_files))
    echo "File pass rate: $pass_rate%" | tee -a "$RESULTS_FILE"
fi

if [ $total_tests -gt 0 ]; then
    test_pass_rate=$((passed_tests * 100 / total_tests))
    echo "Test pass rate: $test_pass_rate%" | tee -a "$RESULTS_FILE"
fi

echo "" | tee -a "$RESULTS_FILE"
echo "Finished: $(date)" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE" | tee -a "$RESULTS_FILE"
if [ $failed_files -gt 0 ]; then
    echo "Failure details saved to: $FAILURES_DIR/" | tee -a "$RESULTS_FILE"
fi

# Print unique errors summary (grouped by error type, with per-pattern counts)
if [ -s "$UNIQUE_ERRORS_FILE" ]; then
    python3 << PYEOF
import re
from collections import defaultdict, OrderedDict

results_file = "$RESULTS_FILE"
temp_file = "$UNIQUE_ERRORS_FILE"

def normalize(msg):
    """Replace variable parts so same-root errors collapse to one pattern."""
    msg = re.sub(r'0x[0-9a-fA-F]+', '<addr>', msg)           # hex addresses
    msg = re.sub(r'__\w+_\d+', lambda m: re.sub(r'\d+', '<N>', m.group()), msg)  # generated names e.g. __subclass_Tensor_42
    msg = re.sub(r'\([0-9, ]+\)', '(<shape>)', msg)           # shape tuples e.g. (256, 67)
    msg = re.sub(r'\b\d+\b', '<N>', msg)                      # bare integers
    msg = re.sub(r"'[^']*'", '<name>', msg)                   # quoted names / paths
    return msg.strip()

# counts[key] = number of tests with this pattern
# canonical[key] = one representative original message
counts = defaultdict(int)
canonical = OrderedDict()
key_to_class = {}

with open(temp_file) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(':', 1)
        error_class = parts[0].strip()
        msg = parts[1].strip() if len(parts) > 1 else ''
        key = error_class + ': ' + normalize(msg)
        counts[key] += 1
        if key not in canonical:
            canonical[key] = line
            key_to_class[key] = error_class

# Group keys by error class
groups = defaultdict(list)
for key in canonical:
    groups[key_to_class[key]].append(key)

lines_out = []
lines_out.append('')
lines_out.append('=' * 40)
lines_out.append('UNIQUE ERROR PATTERNS (grouped by error type)')
lines_out.append('=' * 40)

for error_class in sorted(groups.keys()):
    patterns = groups[error_class]
    total_tests = sum(counts[k] for k in patterns)
    lines_out.append('')
    lines_out.append(f'[{error_class}]  {len(patterns)} pattern(s), {total_tests} test(s) affected')
    for key in patterns:
        lines_out.append(f'  \u2022 [{counts[key]} test(s)]  {canonical[key]}')

lines_out.append('')
lines_out.append(f'Total unique error patterns : {len(canonical)}')
lines_out.append(f'Total error classes         : {len(groups)}')
lines_out.append('=' * 40)

output = '\n'.join(lines_out) + '\n'
print(output)
with open(results_file, 'a') as f:
    f.write(output)
PYEOF
fi

# Clean up temp file
rm -f "$UNIQUE_ERRORS_FILE"

echo ""

