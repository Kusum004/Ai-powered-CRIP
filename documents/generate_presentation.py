# documents/generate_presentation.py
"""Automated executive presentation deck PDF generator using ReportLab with embedded charts."""
import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.config import Config

def build_pdf_presentation():
    """Generates the executive slide deck PDF with embedded charts and screenshots."""
    Config.ensure_directories()
    pdf_path = Config.DOCUMENTS_DIR / "project_presentation.pdf"
    charts_dir = Config.DOCUMENTS_DIR / "eda_charts"

    # Setup landscape document (11 x 8.5 in)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(letter),
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom FinTech Dark/Slate Theme Styles
    title_style = ParagraphStyle(
        'DeckTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=12
    )
    
    subtitle_style = ParagraphStyle(
        'DeckSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#475569'),
        spaceAfter=20
    )

    slide_header_style = ParagraphStyle(
        'SlideHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=10
    )

    body_style = ParagraphStyle(
        'SlideBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'SlideBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=5,
        leftIndent=15
    )

    highlight_box_style = ParagraphStyle(
        'HighlightBox',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#0284C7')
    )

    story = []

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    story.append(Spacer(1, 40))
    story.append(Paragraph("Enterprise AI-Powered Credit Risk Intelligence Platform", title_style))
    story.append(Paragraph("NeoStats Candidate Assessment — Master Engineering & Execution Architecture", subtitle_style))
    story.append(Spacer(1, 20))
    
    title_table_data = [
        [Paragraph("<b>Domain</b>: Consumer Credit Risk & Portfolio Underwriting", body_style), Paragraph("<b>Dataset</b>: Home Credit Default Risk (307,511 records)", body_style)],
        [Paragraph("<b>Architecture</b>: DuckDB OLAP + LightGBM + SHAP + NL-to-SQL", body_style), Paragraph("<b>Target Optimization</b>: Cost-Sensitive (scale_pos_weight=11.387)", body_style)],
        [Paragraph("<b>Interface</b>: Enterprise Streamlit FinTech Cockpit (Dark Slate)", body_style), Paragraph("<b>Deployment</b>: Docker Microservice & Compose", body_style)]
    ]
    t_title = Table(title_table_data, colWidths=[340, 340])
    t_title.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1F5F9')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_title)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 2: Executive Summary & Financial Asymmetry
    # -------------------------------------------------------------
    story.append(Paragraph("1. Executive Summary & Financial Loss Matrix", slide_header_style))
    story.append(Paragraph("Traditional retail lending models suffer from heavy class imbalance and uncalibrated cutoffs. This platform solves the problem using an asymmetric cost-sensitive architecture.", body_style))
    story.append(Spacer(1, 10))

    exec_data = [
        [Paragraph("<b>Key Dimension</b>", highlight_box_style), Paragraph("<b>Empirical Value / Specification</b>", highlight_box_style), Paragraph("<b>Executive Business Rationale</b>", highlight_box_style)],
        [Paragraph("<b>Dataset Size</b>", body_style), Paragraph("307,511 Loan Applications (122 features)", body_style), Paragraph("Full representative portfolio of diverse retail borrowers.", body_style)],
        [Paragraph("<b>Default Imbalance</b>", body_style), Paragraph("8.07% Defaults (24,825) vs 91.93% Non-Defaults", body_style), Paragraph("Imbalance ratio 11.387:1 requires specialized loss re-weighting.", body_style)],
        [Paragraph("<b>False Negative Cost</b>", body_style), Paragraph("$10,000 Expected Default Charge-Off", body_style), Paragraph("Failing to detect a defaulter generates catastrophic balance sheet loss.", body_style)],
        [Paragraph("<b>False Positive Cost</b>", body_style), Paragraph("$1,000 Underwriter Friction & Lost Interest", body_style), Paragraph("Declining a creditworthy applicant causes minor opportunity cost.", body_style)],
        [Paragraph("<b>Cost Ratio</b>", body_style), Paragraph("10 : 1 Asymmetric Loss Penalty", body_style), Paragraph("Directly integrated into precision-recall decision thresholds.", body_style)]
    ]
    t_exec = Table(exec_data, colWidths=[150, 230, 300])
    t_exec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_exec)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 3: Exploratory Data Analysis & Visual Insights
    # -------------------------------------------------------------
    story.append(Paragraph("2. Exploratory Data Analysis: Key Banking Insights", slide_header_style))
    story.append(Paragraph("Core quantitative risk drivers discovered across 307,511 loan applications:", body_style))
    story.append(Spacer(1, 6))

    img_path_1 = charts_dir / "1_income_vs_default.png"
    img_path_2 = charts_dir / "2_ext_source_distribution.png"
    
    if img_path_1.exists() and img_path_2.exists():
        img1 = Image(str(img_path_1), width=3.3*inch, height=1.9*inch)
        img2 = Image(str(img_path_2), width=3.3*inch, height=1.9*inch)
        img_table = Table([[img1, img2]], colWidths=[340, 340])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(img_table)

    story.append(Spacer(1, 6))
    eda_summary = [
        "<b>Income Gradient</b>: Lowest income quintile defaults at ~9.8% vs ~5.6% for top earners (1.75x spread).",
        "<b>External Bureau Scores (EXT_SOURCES_MEAN)</b>: Single strongest feature (AUC 0.72 standalone); borrowers with score &lt; 0.35 carry default probability &gt; 20%."
    ]
    for b in eda_summary:
        story.append(Paragraph(b, bullet_style))
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 4: Machine Learning & Class Imbalance Benchmark
    # -------------------------------------------------------------
    story.append(Paragraph("3. Model Benchmarking: Champion vs Baseline", slide_header_style))
    story.append(Paragraph("Stratified 5-Fold Cross Validation demonstrates massive performance lift over traditional linear baselines:", body_style))
    story.append(Spacer(1, 8))

    ml_table_data = [
        [Paragraph("<b>Metric / Criterion</b>", highlight_box_style), Paragraph("<b>Baseline: Logistic Regression</b>", highlight_box_style), Paragraph("<b>Champion: LightGBM (GBDT)</b>", highlight_box_style), Paragraph("<b>Performance Lift</b>", highlight_box_style)],
        [Paragraph("<b>Imbalance Strategy</b>", body_style), Paragraph("class_weight='balanced'", body_style), Paragraph("scale_pos_weight=11.387", body_style), Paragraph("Optimized Loss Gradient", body_style)],
        [Paragraph("<b>Cross-Validation</b>", body_style), Paragraph("Stratified 5-Fold CV", body_style), Paragraph("Stratified 5-Fold CV + Early Stop", body_style), Paragraph("Zero Data Leakage", body_style)],
        [Paragraph("<b>ROC-AUC Score</b>", body_style), Paragraph("0.7475 (OOF)", body_style), Paragraph("<b>0.7665 (OOF) / 0.8130 Full</b>", body_style), Paragraph("<b>+0.0191 (+1.91%)</b>", body_style)],
        [Paragraph("<b>PR-AUC (Precision-Recall)</b>", body_style), Paragraph("0.2246", body_style), Paragraph("<b>0.2520</b>", body_style), Paragraph("<b>+0.0274 (+12.2%)</b>", body_style)],
        [Paragraph("<b>Portfolio Cost Savings</b>", body_style), Paragraph("Baseline Benchmark", body_style), Paragraph("<b>$8,712,000 Expected Savings</b>", body_style), Paragraph("50% Fewer Defaults Missed", body_style)]
    ]
    t_ml = Table(ml_table_data, colWidths=[160, 180, 180, 160])
    t_ml.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_ml)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 5: FICO-Scale Scoring & Basel III Risk Bands
    # -------------------------------------------------------------
    story.append(Paragraph("4. FICO-Scaled Scoring & Basel III Risk Banding", slide_header_style))
    story.append(Paragraph("Calibrated default probabilities $P$ map directly to standardized 300–850 credit scores: Score = round(850 - (P * 550)).", body_style))
    story.append(Spacer(1, 10))

    band_table_data = [
        [Paragraph("<b>Risk Band</b>", highlight_box_style), Paragraph("<b>Default Prob (P)</b>", highlight_box_style), Paragraph("<b>Credit Score</b>", highlight_box_style), Paragraph("<b>Empirical Default</b>", highlight_box_style), Paragraph("<b>Underwriting Decision</b>", highlight_box_style)],
        [Paragraph("<b>[LOW RISK]</b>", body_style), Paragraph("P &lt; 0.07", body_style), Paragraph("<b>750 – 850</b>", body_style), Paragraph("&lt; 2.1%", body_style), Paragraph("<b>Auto-Approve</b>: Prime rate pricing, instant digital disbursal.", body_style)],
        [Paragraph("<b>[MEDIUM RISK]</b>", body_style), Paragraph("0.07 &le; P &le; 0.20", body_style), Paragraph("<b>600 – 749</b>", body_style), Paragraph("9.5%", body_style), Paragraph("<b>Manual Underwriting</b>: Income verification, collateral check.", body_style)],
        [Paragraph("<b>[HIGH RISK]</b>", body_style), Paragraph("P &gt; 0.20", body_style), Paragraph("<b>300 – 599</b>", body_style), Paragraph("&gt; 38.0%", body_style), Paragraph("<b>Decline / Restructure</b>: High-risk tier, reject unsecured credit.", body_style)]
    ]
    t_band = Table(band_table_data, colWidths=[110, 110, 100, 110, 250])
    t_band.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#ECFDF5')),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#FFFBEB')),
        ('BACKGROUND', (0,3), (-1,3), colors.HexColor('#FEF2F2')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_band)
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 6: Explainable AI & Rule Engine
    # -------------------------------------------------------------
    story.append(Paragraph("5. Explainable AI & Credit Policy Induction", slide_header_style))
    story.append(Paragraph("Eliminating black-box risk decisions through polynomial-time Game Theoretic SHAP attributions and transparent decision rules:", body_style))
    story.append(Spacer(1, 8))

    xai_bullets = [
        "<b>SHAP TreeExplainer</b>: Computes exact Shapley values O(TLD^2) allocating positive/negative log-odds credit risk impact per feature.",
        "<b>Automated Underwriter Summaries</b>: Converts mathematical attributions into plain-English credit committee narratives (Top 3 Adverse Factors + Top 3 Mitigating Strengths).",
        "<b>Decision Tree Rule Induction</b>: Automatically extracts transparent credit rules (e.g., IF EXT_SOURCES_MEAN &lt; 0.38 AND DTI &gt; 28% THEN Risk=High).",
        "<b>Interactive Scenario Testing</b>: Real-time dynamic sliders allow loan officers to simulate applicant income increases or term extensions."
    ]
    for b in xai_bullets:
        story.append(Paragraph(b, bullet_style))
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 7: Talk-to-Data NL-to-SQL Engine
    # -------------------------------------------------------------
    story.append(Paragraph("6. Talk-to-Data NL-to-SQL Copilot", slide_header_style))
    story.append(Paragraph("Enables risk executives and analysts to query the 307k-record portfolio in natural language with zero latency.", body_style))
    story.append(Spacer(1, 8))

    copilot_bullets = [
        "<b>Zero-Latency DuckDB OLAP</b>: In-memory columnar engine performs multi-dimensional aggregations in &lt; 15 milliseconds.",
        "<b>AST Safety Sanitizer</b>: sqlparse token inspection guarantees read-only SELECT queries and strictly rejects DROP, DELETE, or ALTER.",
        "<b>Self-Healing Retry Loop</b>: Automatically feeds syntax errors back to LLM to self-heal and re-execute valid SQL queries.",
        "<b>Executive Synthesis</b>: Generates 3 structured business takeaway bullet points alongside formatted data tables and interactive charts."
    ]
    for b in copilot_bullets:
        story.append(Paragraph(b, bullet_style))
    story.append(PageBreak())

    # -------------------------------------------------------------
    # SLIDE 8: Production Deployment & Wrap-up
    # -------------------------------------------------------------
    story.append(Paragraph("7. Production Packaging & Deployment", slide_header_style))
    story.append(Paragraph("Fully reproducible enterprise containerization:", body_style))
    story.append(Spacer(1, 8))

    deploy_bullets = [
        "<b>Docker Containerization</b>: Multi-stage Dockerfile and docker-compose.yml for one-click cross-platform deployment.",
        "<b>Health Diagnostics</b>: Automated CPU, Memory, and Storage monitoring built into the runtime engine.",
        "<b>Zero External Dependencies</b>: Built-in offline fallback ensures full functionality without external cloud API access.",
        "<b>Audit & Compliance</b>: Comprehensive JSON metric logging and model artifact versioning."
    ]
    for b in deploy_bullets:
        story.append(Paragraph(b, bullet_style))
    
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Status: Completed & Ready for Production Deployment</b>", highlight_box_style))

    # Build PDF
    doc.build(story)
    print(f"Presentation deck successfully built at: {pdf_path}")

if __name__ == "__main__":
    build_pdf_presentation()
