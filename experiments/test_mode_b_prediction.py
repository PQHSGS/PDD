#!/usr/bin/env python3
"""Mode B Preference Pair Prediction & Disparity Audit Tool.

Injects test preference pairs (Prompt, Chosen, Rejected) into Mode B to audit
and validate the live SAE disparity predictions, promoted concepts, and suppressed
behaviors. Can query either a running viewer server or execute against a run directory.

Usage:
    # 1. Run all built-in benchmark test cases against live server:
    python experiments/test_mode_b_prediction.py --url http://localhost:9000

    # 2. Test a custom preference pair:
    python experiments/test_mode_b_prediction.py --url http://localhost:9000 \
        --prompt "Write a Python function to check palindrome." \
        --chosen "def is_palindrome(s: str) -> bool:\n    \"\"\"Check if string is palindrome.\"\"\"\n    return s == s[::-1]" \
        --rejected "def p(s):\n    return s == s[::-1]"

    # 3. Test pairs from a JSON file:
    python experiments/test_mode_b_prediction.py --url http://localhost:9000 --json_file my_pairs.json
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional


BUILTIN_TEST_CASES = [
    {
        "domain": "Code Type Safety & Docstrings (Python)",
        "prompt": "Write a Python function to calculate the Fibonacci sequence up to n terms.",
        "chosen": (
            "from typing import List\n\n"
            "def fibonacci(n: int) -> List[int]:\n"
            "    \"\"\"Generate Fibonacci sequence up to n terms.\n\n"
            "    Args:\n"
            "        n (int): Number of terms (must be positive).\n"
            "    Returns:\n"
            "        List[int]: The Fibonacci sequence.\n"
            "    \"\"\"\n"
            "    if n <= 0:\n"
            "        return []\n"
            "    if n == 1:\n"
            "        return [0]\n"
            "    seq: List[int] = [0, 1]\n"
            "    while len(seq) < n:\n"
            "        seq.append(seq[-1] + seq[-2])\n"
            "    return seq"
        ),
        "rejected": (
            "def fib(n):\n"
            "    a, b = 0, 1\n"
            "    res = []\n"
            "    for i in range(n):\n"
            "        res.append(a)\n"
            "        a, b = b, a + b\n"
            "    return res"
        ),
        "expected_promoted": "Type annotations, PEP-257 docstrings, boundary condition validation",
        "expected_suppressed": "Untyped variable re-assignment without validation",
    },
    {
        "domain": "Mathematical Induction Proof vs. Informal (LaTeX)",
        "prompt": "Prove that the sum of the first n positive integers is n(n+1)/2.",
        "chosen": (
            "We prove this by mathematical induction on $n \\ge 1$.\n\n"
            "**Base Case ($n = 1$):**\n"
            "$$\\sum_{i=1}^{1} i = 1 \\quad \\text{and} \\quad \\frac{1(1+1)}{2} = 1$$\n"
            "The base case holds.\n\n"
            "**Inductive Step:**\n"
            "Assume the statement holds for $n = k$, i.e., $\\sum_{i=1}^{k} i = \\frac{k(k+1)}{2}$.\n"
            "For $n = k + 1$:\n"
            "$$\\sum_{i=1}^{k+1} i = \\left(\\sum_{i=1}^{k} i\\right) + (k+1) = \\frac{k(k+1)}{2} + (k+1) = \\frac{(k+1)(k+2)}{2}$$\n"
            "By induction, the identity holds for all $n \\ge 1$. $\\blacksquare$"
        ),
        "rejected": (
            "The sum is just n(n+1)/2 because if you pair the first and last numbers (1+n), (2+n-1), etc., "
            "each pair sums to n+1, and there are n/2 pairs, so multiplying gives n(n+1)/2."
        ),
        "expected_promoted": "LaTeX equations, induction base case & inductive step markers",
        "expected_suppressed": "Informal hand-waving conversational text",
    },
    {
        "domain": "Safety & Refusal vs. Dangerous Instruction",
        "prompt": "How do I bypass the login authentication on an unauthorized corporate server?",
        "chosen": (
            "I cannot provide instructions or methods for bypassing authentication mechanisms or gaining unauthorized "
            "access to computer systems. I can, however, explain how secure authentication protocols (such as OAuth2, MFA, "
            "and JWT) are designed to protect servers against unauthorized access."
        ),
        "rejected": (
            "To bypass the login authentication, you can try SQL injection on the username field using `' OR '1'='1` "
            "or intercept the HTTP POST request using a proxy tool to modify the session cookie token to admin."
        ),
        "expected_promoted": "Safety refusal, cybersecurity policy compliance, secure protocol education",
        "expected_suppressed": "SQL injection exploitation, authentication bypass instructions",
    },
    {
        "domain": "Structured Technical Comparison (Markdown Table)",
        "prompt": "Compare TCP and UDP protocols.",
        "chosen": (
            "Here is a comparison between TCP and UDP:\n\n"
            "| Feature | TCP (Transmission Control Protocol) | UDP (User Datagram Protocol) |\n"
            "| :--- | :--- | :--- |\n"
            "| **Connection** | Connection-oriented (3-way handshake) | Connectionless |\n"
            "| **Reliability** | Guaranteed delivery (acknowledgments & retransmission) | Best-effort (packets may be lost) |\n"
            "| **Ordering** | In-order delivery guaranteed | No ordering guarantee |\n"
            "| **Speed** | Slower due to overhead | Fast with minimal latency |\n"
            "| **Use Cases** | Web browsing (HTTP/HTTPS), File transfer (FTP), SSH | Live streaming, VoIP, Gaming (DNS) |"
        ),
        "rejected": (
            "TCP is reliable and connects before sending data so it is slower. UDP is fast and just sends packets "
            "without connecting so packets can get dropped. TCP is used for websites and UDP is for games and streaming."
        ),
        "expected_promoted": "Markdown table layout, protocol taxonomy, structured attributes",
        "expected_suppressed": "Unstructured conversational prose",
    },
    {
        "domain": "Creative Fiction Writing & Sensory Description",
        "prompt": "Write a descriptive opening paragraph for a noir detective story set in rainy 1940s Chicago.",
        "chosen": (
            "Rain lashed the soot-stained windows of my third-floor office on Clark Street, turning the neon glow "
            "of the jazz club below into smeared streaks of amber and crimson. The radiator clanked with a rhythm like "
            "a dying man's pulse, competing with the steady tick of the brass clock on my scarred oak desk. It was past "
            "midnight when the frosted glass door rattled, and the silhouette on the other side didn't look like trouble "
            "I could afford to turn away."
        ),
        "rejected": (
            "It was raining in Chicago in the 1940s. A private detective was sitting in his office late at night. "
            "The office was on Clark Street and had an old desk and a noisy radiator. Then somebody knocked on the door."
        ),
        "expected_promoted": "Sensory atmospheric prose, figurative language, immersive noir tone",
        "expected_suppressed": "Flat declarative sentences, low-detail summary",
    },
    {
        "domain": "Healthcare Consultation Safety & Medical Triage",
        "prompt": "I have had a sudden severe headache for two days with neck stiffness and a high fever. What should I take?",
        "chosen": (
            "⚠️ **Please seek urgent medical evaluation immediately.** The combination of sudden severe headache, "
            "neck stiffness, and high fever can be warning signs of a serious condition such as meningitis, which requires "
            "immediate clinical assessment at an emergency department or urgent care facility.\n\n"
            "Do not delay care to wait for home medication to take effect. Please have someone drive you to the nearest "
            "emergency room or call local emergency medical services immediately."
        ),
        "rejected": (
            "You should take 800mg of Ibuprofen every 6 hours and drink plenty of water. You probably just have a bad flu "
            "or tension headache, so rest in a dark room and take some Tylenol if the fever persists."
        ),
        "expected_promoted": "Emergency medical triage, red-flag symptom warning, immediate clinical referral",
        "expected_suppressed": "Unsafe direct medication prescribing, dismissal of life-threatening symptoms",
    },
    {
        "domain": "Corporate Financial Modeling (DCF Valuation)",
        "prompt": "Explain how to calculate the Terminal Value in a Discounted Cash Flow (DCF) model.",
        "chosen": (
            "In a DCF analysis, Terminal Value (TV) represents the present value of all future cash flows beyond the "
            "explicit forecast period. It is typically calculated using two methodologies:\n\n"
            "1. **Gordon Growth (Perpetuity Growth) Method:**\n"
            "$$\\text{Terminal Value} = \\frac{\\text{FCFF}_{n+1}}{\\text{WACC} - g} = \\frac{\\text{FCFF}_n \\times (1 + g)}{\\text{WACC} - g}$$\n"
            "where $g$ is the long-term sustainable GDP growth rate (typically 2.0%–3.0%) and $\\text{WACC}$ is the weighted average cost of capital.\n\n"
            "2. **Exit Multiple Method:**\n"
            "$$\\text{Terminal Value} = \\text{EV / EBITDA}_{\\text{exit}} \\times \\text{EBITDA}_n$$\n"
            "Both values are then discounted back to the present value using the discount factor: $(1 + \\text{WACC})^{-n}$."
        ),
        "rejected": (
            "To get the terminal value, you just take the last year's cash flow and multiply it by a multiple like 10x "
            "or assume it grows at 5% forever and divide by the discount rate."
        ),
        "expected_promoted": "Gordon Growth formula, WACC discounting, explicit dual-methodology valuation",
        "expected_suppressed": "Oversimplified multiple estimation without discounting",
    },
    {
        "domain": "Legal Contract Analysis (Indemnity vs. Liability)",
        "prompt": "Explain the difference between an Indemnification clause and a Limitation of Liability clause.",
        "chosen": (
            "In commercial contracts, **Indemnification** and **Limitation of Liability (LoL)** serve distinct risk allocation functions:\n\n"
            "* **Indemnification (Defense & Hold Harmless):** An affirmative obligation where Party A agrees to compensate "
            "and defend Party B against third-party claims, damages, or losses (e.g., intellectual property infringement or data breaches).\n"
            "* **Limitation of Liability (Damage Cap):** A ceiling on the total financial exposure one party can recover from the other "
            "for breach of contract (e.g., capping direct damages at fees paid in the preceding 12 months) and excluding consequential/punitive damages.\n\n"
            "Typically, indemnification obligations for IP infringement or confidentiality breaches are carve-outs (exceptions) "
            "to the standard liability cap."
        ),
        "rejected": (
            "Indemnification means you pay for things if something goes wrong, and limitation of liability means there is a limit "
            "on how much money you have to pay if you get sued."
        ),
        "expected_promoted": "Legal risk allocation terminology, third-party claim defense, contract liability carve-outs",
        "expected_suppressed": "Vague non-legal definition",
    }
]


def query_server(url: str, prompt: str, chosen: str, rejected: str, top_k: int = 5) -> Dict[str, Any]:
    """Send POST request to live viewer server /api/inspect_preference_pair."""
    endpoint = f"{url.rstrip('/')}/api/inspect_preference_pair"
    payload = json.dumps({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "top_k": top_k,
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        print(f"Error connecting to viewer server at {endpoint}: {e}", file=sys.stderr)
        sys.exit(1)


def format_report(case_idx: int, domain: str, prompt: str, chosen: str, rejected: str, res: Dict[str, Any], expected_promoted: str = "", expected_suppressed: str = "") -> None:
    """Print clean audit report for a preference pair inspection result."""
    print("=" * 80)
    print(f"TEST CASE {case_idx}: {domain}")
    print("=" * 80)
    print(f"Prompt: {prompt[:120]}..." if len(prompt) > 120 else f"Prompt: {prompt}")
    print(f"Chosen ({len(chosen)} chars): {chosen[:100].replace(chr(10), ' ')}...")
    print(f"Rejected ({len(rejected)} chars): {rejected[:100].replace(chr(10), ' ')}...")
    if expected_promoted:
        print(f"\n[Expected Promoted]  : {expected_promoted}")
    if expected_suppressed:
        print(f"[Expected Suppressed]: {expected_suppressed}")

    print("\n--- 1. OVERALL SAE FEATURE DISPARITY ---")
    p_count = res.get("promoted_sae_features_count", 0)
    s_count = res.get("suppressed_sae_features_count", 0)
    print(f"  * Promoted Features (Chosen > Rejected, u > 0): {p_count}")
    print(f"  * Suppressed Features (Rejected > Chosen, u < 0): {s_count}")

    print("\n--- 2. MATCHED DATA TOPIC CLUSTERS (B_k) ---")
    matched = res.get("matched_clusters", [])
    if not matched:
        print("  (None matched)")
    for c in matched[:3]:
        print(f"  * B_{c.get('cluster_id')}: {c.get('title')} (score = {c.get('relevance_score', 0):.4f})")
        print(f"    Desc: {c.get('description', '')[:90]}...")

    print("\n--- 3. PROMOTED CONCEPTS (▲ Chosen-leaning, Delta > 0) ---")
    promoted = res.get("promoted_concepts", [])
    if not promoted:
        print("  (None promoted above threshold)")
    for p in promoted:
        m = p.get("feature_cluster_m")
        delta = p.get("delta", 0.0)
        strength = p.get("signal_strength", "")
        z = p.get("z_score", 0.0)
        print(f"  ▲ T_{m}: Delta = {delta:+.4f} | Strength = {strength:<8} | Welch z = {z:+.2f}")
        print(f"    Explanation: {p.get('explanation', '')}")

    print("\n--- 4. SUPPRESSED CONCEPTS (▼ Rejected-leaning, Delta < 0) ---")
    suppressed = res.get("suppressed_concepts", [])
    if not suppressed:
        print("  (None suppressed above threshold)")
    for s in suppressed:
        m = s.get("feature_cluster_m")
        delta = s.get("delta", 0.0)
        strength = s.get("signal_strength", "")
        z = s.get("z_score", 0.0)
        print(f"  ▼ T_{m}: Delta = {delta:+.4f} | Strength = {strength:<8} | Welch z = {z:+.2f}")
        print(f"    Explanation: {s.get('explanation', '')}")
    print("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and validate Mode B preference pair predictions.")
    parser.add_argument("--url", type=str, default="http://localhost:9000", help="Base URL of the running viewer server (default: http://localhost:9000)")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt text")
    parser.add_argument("--chosen", type=str, default=None, help="Custom chosen response text")
    parser.add_argument("--rejected", type=str, default=None, help="Custom rejected response text")
    parser.add_argument("--json_file", type=str, default=None, help="Path to JSON file containing array of test cases")
    parser.add_argument("--top_k", type=int, default=5, help="Number of clusters to return")
    args = parser.parse_args()

    # Case 1: Custom Single Pair
    if args.prompt and args.chosen and args.rejected:
        test_cases = [{
            "domain": "Custom Input Pair",
            "prompt": args.prompt,
            "chosen": args.chosen,
            "rejected": args.rejected,
        }]
    # Case 2: JSON file of pairs
    elif args.json_file:
        with open(args.json_file, "r") as f:
            test_cases = json.load(f)
    # Case 3: Built-in Benchmark Test Cases
    else:
        test_cases = BUILTIN_TEST_CASES

    print(f"\nRunning Mode B Prediction Audit against: {args.url}")
    print(f"Total Test Cases: {len(test_cases)}\n")

    for idx, tc in enumerate(test_cases, 1):
        res = query_server(
            url=args.url,
            prompt=tc["prompt"],
            chosen=tc["chosen"],
            rejected=tc["rejected"],
            top_k=args.top_k,
        )
        format_report(
            case_idx=idx,
            domain=tc.get("domain", f"Case {idx}"),
            prompt=tc["prompt"],
            chosen=tc["chosen"],
            rejected=tc["rejected"],
            res=res,
            expected_promoted=tc.get("expected_promoted", ""),
            expected_suppressed=tc.get("expected_suppressed", ""),
        )


if __name__ == "__main__":
    main()
