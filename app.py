import re, logging, gradio as gr, pandas as pd
from pathlib import Path
from pipeline import analyze_policy
from config import EXAMPLE_POLICY_TXT

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")

# example policy 
EXAMPLE_POLICY = ""
if Path(EXAMPLE_POLICY_TXT).exists():
    EXAMPLE_POLICY = Path(EXAMPLE_POLICY_TXT).read_text(encoding="utf-8")

# colour helpers 
VERDICT_COLOURS = {
    "Covered":      "#16a34a",   # green-600
    "Partial":      "#d97706",   # amber-600
    "Not Observed": "#dc2626",   # red-600
}

def _badge(verdict: str) -> str:
    colour = VERDICT_COLOURS.get(verdict, "#6b7280")
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 10px;'
        f'border-radius:12px;font-weight:600;font-size:0.85em;">'
        f'{verdict}</span>'
    )

# main callback
def run_analysis(policy_text: str, progress=gr.Progress()):
    if not policy_text.strip():
        raise gr.Error("Please paste a privacy policy first.")

    progress(0.05, desc="Loading embedding model …")
    results = analyze_policy(policy_text)

    if not results:
        raise gr.Error("No GDPR articles matched – is this a valid privacy policy?")

    # build summary table
    rows = []
    for r in results:
        art_match = re.search(r"\d+", str(r.get("article", "")))
        art_label = f"Art. {art_match.group()}" if art_match else str(r.get("article", ""))
        rows.append({
            "Article":   art_label,
            "Title":     r.get("title", ""),
            "Verdict":   _badge(r["coverage"]),
            "Rationale": r.get("rationale", ""),
            "Policy Citations": "; ".join(r.get("policy_citations", [])[:3]),
            "GDPR Refs": ", ".join(r.get("gdpr_citations", [])[:4]),
        })

    df = pd.DataFrame(rows)

    # summary stats
    counts = {"Covered": 0, "Partial": 0, "Not Observed": 0}
    for r in results:
        v = r.get("coverage", "Not Observed")
        counts[v] = counts.get(v, 0) + 1

    total = len(results)
    score_pct = round(
        (counts["Covered"] * 100 + counts["Partial"] * 50) / (total * 100) * 100
    ) if total else 0

    summary_md = (
        f"### Analysis Complete — {total} articles evaluated\n\n"
        f"🟢 **Covered:** {counts['Covered']}  &ensp;"
        f"🟡 **Partial:** {counts['Partial']}  &ensp;"
        f"🔴 **Not Observed:** {counts['Not Observed']}  &ensp;"
        f"📊 **Score: {score_pct}%**"
    )

    return summary_md, df


# Gradio UI

DESCRIPTION = """
Paste any privacy policy below and click **Analyze**.
The tool matches the text against 12 key GDPR articles using semantic search,
then sends each match to **Gemini Flash** for a structured compliance verdict.
"""

with gr.Blocks(
    title="GDPR Policy Analyzer",
    theme=gr.themes.Soft(primary_hue="emerald"),
    css="""
        .verdict-table td:nth-child(3) { text-align:center; }
        footer { display:none !important; }
    """,
) as app:

    gr.Markdown("# 🛡️ GDPR Policy Analyzer")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            policy_input = gr.Textbox(
                label="Privacy Policy Text",
                placeholder="Paste the full privacy policy here …",
                lines=14,
                max_lines=30,
            )
        with gr.Column(scale=1, min_width=180):
            analyze_btn = gr.Button("🔍  Analyze", variant="primary", size="lg")
            gr.Markdown("#### Quick-load example")
            example_btn = gr.Button("📄 Notion Privacy Policy", size="sm")

    summary_output = gr.Markdown(label="Summary")

    results_table = gr.Dataframe(
        label="Article-by-Article Results",
        headers=["Article", "Title", "Verdict", "Rationale", "Policy Citations", "GDPR Refs"],
        datatype=["str", "str", "html", "str", "str", "str"],
        interactive=False,
        wrap=True,
    )

    # wiring
    analyze_btn.click(
        fn=run_analysis,
        inputs=policy_input,
        outputs=[summary_output, results_table],
    )
    example_btn.click(fn=lambda: EXAMPLE_POLICY, outputs=policy_input)


if __name__ == "__main__":
    app.launch()
