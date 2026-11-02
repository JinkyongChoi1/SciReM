"""Run an agent-written R diagnostic in an SGE-isolated, no-network work directory."""
import csv
import hashlib
import math
import os
from pathlib import Path
import re
import resource
import subprocess

from discovery_pilot import DATA, R_HOME, R_LIBRARY
from validate_local_synthetic import sandbox

# Match the primary Runner's survival environment; keep generated code unchanged.
R_DIAGNOSTIC_SETUP = '''suppressPackageStartupMessages(library(survival))
# Read package methods, not assumed summary-object fields. vcov preserves the fit's SE choice.
research_estimate <- function(fit, term) {
  b <- stats::coef(fit)
  v <- stats::vcov(fit)
  if (length(term) != 1L || !term %in% names(b) || !term %in% rownames(v))
    stop("Requested coefficient is absent from the fitted model")
  variance <- v[term, term]
  if (!is.finite(b[term]) || !is.finite(variance) || variance < 0)
    stop("Requested coefficient or covariance is not estimable")
  c(coefficient=unname(b[term]), se=sqrt(unname(variance)))
}
'''
DIAGNOSTIC_ENVIRONMENT = (
    "Runner preloads survival (strata and cluster are available). Use unqualified strata()/cluster() "
    "inside survival formulas and explicitly select method='efron' for clogit when Efron is intended. "
    "For coefficient and SE extraction use research_estimate(fit, term), which returns named numeric "
    "elements coefficient and se using coef(fit) and vcov(fit). It preserves the fitted model's covariance "
    "choice; it does not add clustering. summary(fit)$robust.se is not a valid extraction interface. "
    "R logical predictors receive TRUE coefficient suffixes; for intended binary 0/1 predictors, "
    "explicitly convert the selected logical columns to integer before fitting, or extract their exact "
    "fitted names from coef(fit). Never substitute a different coefficient for a missing requested term. "
    "Use existing primary estimates/covariance when sufficient; generate only the missing diagnostic. "
    "The error receipt contains available code-grounded details; raw host log paths are provenance, "
    "not files accessible to this planning session.")


class DiagnosticOutputError(ValueError):
    """A safe, non-data-bearing diagnostic output contract error."""

    def __init__(self, message, receipt=None):
        super().__init__(message)
        self.receipt = receipt


def execution_error(stderr, code="", returncode=None):
    """Classify known failures without echoing data-bearing R output into prompts."""
    # An R call is echoed only when it is literally present in agent-authored source.
    call = re.search(r"Error in (.*?)\s+:\s*\n?", stderr, re.S)
    expression = re.sub(r"\s+", "", call[1]) if call else ""
    source_lines = [i for i, line in enumerate(code.splitlines(), 1)
                    if expression and expression in re.sub(r"\s+", "", line)]
    # R deparses single-quoted terms with double quotes. Match this known helper's
    # identifier arguments, rather than treating quote style as different code.
    helper_call = r'''research_estimate\(\s*([A-Za-z.][A-Za-z0-9._]*)\s*,\s*(["'])([A-Za-z_][A-Za-z0-9_:]*)\2\s*\)'''
    echoed = re.fullmatch(helper_call, call[1].strip()) if call else None
    if echoed and not source_lines:
        source_lines = [i for i, line in enumerate(code.splitlines(), 1)
                        if any((match[1], match[3]) == (echoed[1], echoed[3])
                               for match in re.finditer(helper_call, line))]
    if 'research_estimate' in expression and source_lines:
        for phrase,category,guidance in (
            ('Requested coefficient is absent from the fitted model','coefficient_name_mismatch',
             'Inspect the exact names returned by coef(fit). Logical predictors receive TRUE suffixes; explicitly convert intended binary predictors to integer before fitting or request their exact fitted coefficient names. Preserve the formula, sample and covariance; do not substitute another coefficient.'),
            ('Requested coefficient or covariance is not estimable','coefficient_not_estimable',
             'Check model identification and whether a main effect is absorbed by strata. An absorbed main effect does not imply an interaction is unestimable. Report unavailable terms honestly; never replace them with zero.')):
            if phrase in stderr:
                return {'category':category,'stage':'execution','message':phrase,
                        'failed_expression':expression,'source_lines':source_lines,'repair_guidance':guidance}
    for phrase in ("replacement has length zero", "subscript out of bounds", "incorrect number of dimensions"):
        if phrase in stderr:
            detail = {"failed_expression": expression, "source_lines": source_lines} if source_lines else {}
            return {"category": "invalid_result_extraction", "stage": "execution", "message": phrase,
                    **detail,
                    "repair_guidance": "Check the exact object/extraction expression. For fitted coefficient/SE use research_estimate(fit, term), backed by coef/vcov; do not replace missing results with zero or silently change inference."}
    # Only return identifiers present in submitted source; never echo arbitrary R values.
    match = re.search(r'''(?:could not find function|object) ["']([A-Za-z.][A-Za-z0-9._]*)["'](?: not found)?''', stderr)
    if match and re.search(r"(?<![A-Za-z0-9._])" + re.escape(match[1]) + r"(?![A-Za-z0-9._])", code):
        symbol = match[1]
        category = "function_not_found" if "could not find function" in match[0] else "object_not_found"
        lines = [index for index, line in enumerate(code.splitlines(), 1)
                 if re.search(r"(?<![A-Za-z0-9._])" + re.escape(symbol) + r"(?![A-Za-z0-9._])", line)]
        return {"category": category, "stage": "execution", "symbol": symbol,
                "source_lines": lines, "message": f"R could not resolve {symbol} used by the submitted code.",
                "repair_guidance": "Fix this symbol's binding or definition using the installed package/data contract; preserve the scientific specification."}
    if "combn" in stderr and any(message in stderr for message in (
            "vector size cannot be NA", "coercion to integer range", "cannot allocate")):
        return {"category": "combinatorial_enumeration_overflow", "stage": "execution",
                "message": "combn attempted to materialize a combination set that exceeded R's vector/allocation capacity.",
                "repair_guidance": "Replace exhaustive combination enumeration with an appropriate package-supported likelihood or prediction/scoring method; preserve the intended comparison, sample and uncertainty target."}
    if "there is no package called" in stderr:
        return {"category": "package_unavailable", "stage": "execution",
                "message": "A package requested by the diagnostic is unavailable in the sandbox.",
                "repair_guidance": "Use the documented installed packages or report the missing dependency explicitly; do not silently substitute a different scientific test."}
    if "is not TRUE" in stderr:
        return {"category": "assertion_failed", "stage": "execution",
                "message": "An assertion in the submitted diagnostic failed.",
                "repair_guidance": "Review the submitted code's assertions against the data/model assumptions; do not delete the assertion solely to obtain a result."}
    if "unexpected" in stderr or "Error in parse" in stderr:
        return {"category": "syntax_error", "stage": "execution",
                "message": "R could not parse part of the submitted diagnostic.",
                "repair_guidance": "Correct the submitted R syntax while retaining the intended analysis."}
    return {"category": "unclassified_execution_error", "stage": "execution",
            "returncode": returncode, "repairable_from_receipt": False,
            "message": "R failed but no safe, actionable error detail was extracted; the full error is retained locally.",
            "repair_guidance": "Return this unresolved execution failure to the supervisor for error inspection; do not guess a code repair or claim a scientific result."}


def output_error(receipt, message):
    failure = {**receipt, "reason": message, "error": {
        "category": "invalid_output_contract", "stage": "output_validation", "message": message,
        "repair_guidance": "Repair the aggregate metric,value CSV output contract; preserve the scientific analysis and do not output individual observations."}}
    return DiagnosticOutputError(message, failure)


def run_agent_r(session, code, data=DATA, cpu_seconds=None):
    if not os.environ.get("JOB_ID"):
        raise RuntimeError("Scheduler allocation required")
    if not isinstance(code, str) or not code.strip() or len(code.encode()) > 16000:
        raise ValueError("R diagnostic must be nonempty and at most 16 KB")
    folder = Path(session) / "agent-r-diagnostic"
    folder.mkdir(mode=0o700)
    source = folder / "diagnostic.R"
    source.write_text(code)
    source.chmod(0o600)
    setup = folder / "runner.R"
    setup.write_text(R_DIAGNOSTIC_SETUP + 'source("/work/diagnostic.R", keep.source=TRUE)\n')
    setup.chmod(0o600)
    command = sandbox(folder)[:-1] + ["--ro-bind", R_HOME, R_HOME,
        "--ro-bind", R_LIBRARY, "/r-library", "--setenv", "R_LIBS_USER", "/r-library",
        "--ro-bind", str(data), "/data/styles.csv", "--setenv", "R_HOME", R_HOME,
        "--setenv", "LD_LIBRARY_PATH", R_HOME + "/lib", "--",
        R_HOME + "/bin/Rscript", "--vanilla", "/work/runner.R"]
    env = {"PATH": "/usr/bin:/bin", "HOME": "/work", "LANG": "C.UTF-8",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    # No captured process output goes into a model prompt; scheduler limits bound execution time.
    def bound_file_size():
        resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 * 1024, 2 * 1024 * 1024))
        if cpu_seconds is not None:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    with (folder / "stdout.log").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
        proc = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                              env=env, preexec_fn=bound_file_size)
    code_hash = hashlib.sha256(code.encode()).hexdigest()
    stderr_path = folder / "stderr.log"
    stderr_bytes = stderr_path.read_bytes()
    receipt = {"status": "failed", "returncode": proc.returncode,
               "code": code, "code_sha256": code_hash,
               "source": {"path": str(source.relative_to(session)), "sha256": code_hash},
               "stderr_source": {"path": str(stderr_path.relative_to(session)),
                                 "sha256": hashlib.sha256(stderr_bytes).hexdigest()},
               "runner_source": {"path": str(setup.relative_to(session)),
                                 "sha256": hashlib.sha256(setup.read_bytes()).hexdigest()},
               "scope": "Agent-written diagnostic, not independently validated; aggregate output only."}
    output = folder / "diagnostics.csv"
    if proc.returncode or output.is_symlink() or not output.is_file() or output.stat().st_size > 16384:
        stderr = stderr_bytes.decode(errors="replace")
        receipt["reason"] = ("R assertion failed; check diagnostic assumptions" if "is not TRUE" in stderr else
                             "R package unavailable" if "there is no package called" in stderr else
                             "R execution failed" if proc.returncode else "diagnostic output missing or oversized")
        receipt["error"] = (execution_error(stderr, code, proc.returncode) if proc.returncode else {
            "category": "missing_or_invalid_output", "stage": "output_validation",
            "message": "R exited without an admissible diagnostics.csv artifact.",
            "repair_guidance": "Write the aggregate metric,value CSV to /work/diagnostics.csv within the existing sandbox output contract; do not use a symlink or output individual observations."})
        return receipt
    with output.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["metric", "value"]:
            raise output_error(receipt, "Output needs exactly metric,value columns")
        rows = list(reader)
    if not 1 <= len(rows) <= 200:
        raise output_error(receipt, "Output requires 1–200 aggregate metrics")
    values = {}
    for row in rows:
        if set(row) != {"metric", "value"} or any(value is None for value in row.values()):
            raise output_error(receipt, "Every output row needs exactly metric,value fields")
        name = row["metric"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name) or name in values:
            raise output_error(receipt, "Invalid or duplicate diagnostic metric name")
        raw = row["value"].strip()
        if raw in ("", "NA", "NaN"):
            values[name] = None
            continue
        try:
            value = float(raw)
        except ValueError as error:
            raise output_error(receipt, "Diagnostic values must be numeric or NA") from error
        if not math.isfinite(value):
            raise output_error(receipt, "Diagnostic values must be finite or NA")
        values[name] = value
    receipt.update(status="completed", metrics=values, output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    return receipt
