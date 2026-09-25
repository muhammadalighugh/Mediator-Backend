"""
reporting/pdf_report.py
-----------------------
Build a clean A4 PDF from a MediationReport using ReportLab Platypus.

English-only fields are used throughout — verbatim Urdu/Hindi quotes are
deliberately excluded to avoid font-shaping artefacts with Helvetica.

Public entry point
------------------
    build_pdf(report: MediationReport, speakers: list[Speaker]) -> bytes
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.flowables import KeepTogether

from models.schemas import MediationReport, Speaker

# ---------------------------------------------------------------------------
# Colour palette (close to the dark-mode UI but lightened for white paper)
# ---------------------------------------------------------------------------
_C_HEADER_TEXT   = colors.HexColor("#1a1d27")   # near-black
_C_MUTED         = colors.HexColor("#7b8096")   # gray
_C_DIVIDER       = colors.HexColor("#d0d3de")   # light gray line
_C_ROW_ALT       = colors.HexColor("#f7f8fa")   # alternating table row
_C_SUPPORTED     = colors.HexColor("#16a34a")   # green-600
_C_CONTRADICTED  = colors.HexColor("#dc2626")   # red-600
_C_UNCERTAIN     = colors.HexColor("#d97706")   # amber-600
_C_INSUFF        = colors.HexColor("#7b8096")   # gray
_C_BADGE_FACT    = colors.HexColor("#1d4ed8")   # blue
_C_SECTION_HEAD  = colors.HexColor("#374151")   # dark gray


def _verdict_color(verdict: str) -> colors.HexColor:
    v = (verdict or "").lower()
    if v == "supported":
        return _C_SUPPORTED
    if v == "contradicted":
        return _C_CONTRADICTED
    if v in ("uncertain", "insufficient_evidence"):
        return _C_UNCERTAIN
    return _C_MUTED


def _verdict_label(verdict: str) -> str:
    v = (verdict or "").lower()
    mapping = {
        "supported": "SUPPORTED",
        "contradicted": "CONTRADICTED",
        "uncertain": "UNCERTAIN",
        "insufficient_evidence": "INSUFFICIENT",
    }
    return mapping.get(v, verdict.upper())


# ---------------------------------------------------------------------------
# Style registry
# ---------------------------------------------------------------------------

def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()

    def s(name: str, **kw) -> ParagraphStyle:
        ps = ParagraphStyle(name, parent=base["Normal"], **kw)
        return ps

    return {
        "title": s(
            "title",
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=_C_HEADER_TEXT,
        ),
        "subtitle": s(
            "subtitle",
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            textColor=_C_MUTED,
        ),
        "meta": s(
            "meta",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=_C_MUTED,
            alignment=TA_RIGHT,
        ),
        "section_head": s(
            "section_head",
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=_C_SECTION_HEAD,
            spaceBefore=4,
            spaceAfter=4,
            borderPad=0,
        ),
        "body": s(
            "body",
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=_C_HEADER_TEXT,
            alignment=TA_JUSTIFY,
        ),
        "body_left": s(
            "body_left",
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=_C_HEADER_TEXT,
            alignment=TA_LEFT,
        ),
        "italic": s(
            "italic",
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11,
            textColor=_C_SECTION_HEAD,
        ),
        "small": s(
            "small",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=_C_MUTED,
        ),
        "small_bold": s(
            "small_bold",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=11,
            textColor=_C_SECTION_HEAD,
        ),
        "table_head": s(
            "table_head",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=11,
            textColor=_C_SECTION_HEAD,
        ),
        "table_cell": s(
            "table_cell",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=_C_HEADER_TEXT,
        ),
        "table_cell_italic": s(
            "table_cell_italic",
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11,
            textColor=_C_SECTION_HEAD,
        ),
        "footer": s(
            "footer",
            fontName="Helvetica-Oblique",
            fontSize=7,
            leading=10,
            textColor=_C_MUTED,
            alignment=TA_CENTER,
        ),
        "disclaimer": s(
            "disclaimer",
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11,
            textColor=_C_MUTED,
            alignment=TA_CENTER,
        ),
    }


# ---------------------------------------------------------------------------
# Page template callback (footer on every page)
# ---------------------------------------------------------------------------

class _FooterCanvas:
    """Wrapper that draws the footer line on every page."""

    DISCLAIMER = (
        "This report does not declare a winner — verdicts reflect available evidence only."
    )

    def __init__(self, canvas, doc):  # noqa: ANN001
        self.canvas = canvas
        self.doc = doc
        canvas.saveState()
        self._draw_footer()
        canvas.restoreState()

    def _draw_footer(self) -> None:
        c = self.canvas
        width, height = A4
        margin_x = 20 * mm
        footer_y = 12 * mm

        # Thin divider line
        c.setStrokeColor(_C_DIVIDER)
        c.setLineWidth(0.5)
        c.line(margin_x, footer_y + 4 * mm, width - margin_x, footer_y + 4 * mm)

        # Disclaimer text (left-aligned)
        c.setFont("Helvetica-Oblique", 7)
        c.setFillColor(_C_MUTED)
        c.drawString(margin_x, footer_y, self.DISCLAIMER)

        # Page number (right-aligned)
        page_text = f"Page {self.doc.page}"
        c.drawRightString(width - margin_x, footer_y, page_text)


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def _section_head(text: str, styles: dict) -> list:
    return [
        Spacer(1, 4 * mm),
        Paragraph(text.upper(), styles["section_head"]),
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=_C_DIVIDER,
            spaceAfter=3 * mm,
        ),
    ]


def _safe(text: object) -> str:
    """Return text as a non-None, XML-escaped string safe for Paragraph."""
    if text is None:
        return "—"
    s = str(text)
    # Escape XML special chars that would break Paragraph
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s or "—"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_pdf(
    report: MediationReport,
    speakers: Optional[list[Speaker]] = None,
) -> bytes:
    """
    Render *report* as a PDF and return the raw bytes.

    *speakers* is the list of Speaker objects from the live session; used to
    map speaker_id → display_name.  Falls back to speaker_id strings when
    absent.
    """
    speaker_map: dict[str, str] = {}
    if speakers:
        for sp in speakers:
            speaker_map[sp.id] = sp.display_name

    def _speaker_name(speaker_id: Optional[str]) -> str:
        if not speaker_id:
            return "Unknown"
        return speaker_map.get(speaker_id, speaker_id)

    styles = _build_styles()
    buf = io.BytesIO()

    MARGIN = 20 * mm
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=22 * mm,
        bottomMargin=22 * mm,
        title="Mediation Report",
        author="MediFact",
    )

    story: list = []

    # -----------------------------------------------------------------------
    # HEADER
    # -----------------------------------------------------------------------
    now_str = datetime.now(tz=timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    session_short = report.session_id[:16] + ("…" if len(report.session_id) > 16 else "")

    # Two-column header table: title left, meta right
    header_data = [[
        [
            Paragraph("&#9878; MediFact", styles["title"]),
            Spacer(1, 1 * mm),
            Paragraph("Mediation Report", styles["subtitle"]),
        ],
        [
            Paragraph(now_str, styles["meta"]),
            Paragraph(f"Session: {_safe(session_short)}", styles["meta"]),
        ],
    ]]
    header_table = Table(
        header_data,
        colWidths=["65%", "35%"],
        style=TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]),
    )
    story.append(header_table)
    story.append(Spacer(1, 2 * mm))
    story.append(HRFlowable(width="100%", thickness=1.0, color=_C_HEADER_TEXT))
    story.append(Spacer(1, 3 * mm))

    # Dispute type badge (just a bold paragraph line — no box drawing needed)
    dispute = (report.dispute_type or "UNKNOWN").upper()
    story.append(
        Paragraph(
            f'<font name="Helvetica-Bold" color="{_C_BADGE_FACT.hexval()}">&#9654; {dispute} DISPUTE</font>',
            styles["body_left"],
        )
    )
    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------------------------
    # ASSESSMENT
    # -----------------------------------------------------------------------
    assessments = report.assessments or []
    if assessments:
        story += _section_head("Assessment", styles)

        assess_rows = [[
            Paragraph("Speaker", styles["table_head"]),
            Paragraph("Supported", styles["table_head"]),
            Paragraph("Contradicted", styles["table_head"]),
            Paragraph("Unverified", styles["table_head"]),
        ]]
        for a in assessments:
            name = _speaker_name(a.speaker_id)
            assess_rows.append([
                Paragraph(_safe(name), styles["table_cell"]),
                Paragraph(
                    f'<font color="{_C_SUPPORTED.hexval()}"><b>{a.supported}</b></font>',
                    styles["table_cell"],
                ),
                Paragraph(
                    f'<font color="{_C_CONTRADICTED.hexval()}"><b>{a.contradicted}</b></font>',
                    styles["table_cell"],
                ),
                Paragraph(
                    f'<font color="{_C_MUTED.hexval()}">{a.uncertain}</font>',
                    styles["table_cell"],
                ),
            ])

        col_w = [65 * mm, 30 * mm, 35 * mm, 30 * mm]
        assess_table = Table(assess_rows, colWidths=col_w, repeatRows=1)
        assess_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f1f5")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _C_ROW_ALT]),
            ("GRID", (0, 0), (-1, -1), 0.3, _C_DIVIDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(assess_table)

        # Favorability line
        if len(assessments) >= 2:
            a0, a1 = assessments[0], assessments[1]
            score0 = a0.supported - a0.contradicted
            score1 = a1.supported - a1.contradicted
            n0 = _speaker_name(a0.speaker_id)
            n1 = _speaker_name(a1.speaker_id)
            if score0 - score1 >= 2:
                fav = (
                    f"The evidence favors {n0} — {a0.supported} of "
                    f"{a0.checkable_total} checkable claim"
                    f"{'s' if a0.checkable_total != 1 else ''} supported."
                )
            elif score1 - score0 >= 2:
                fav = (
                    f"The evidence favors {n1} — {a1.supported} of "
                    f"{a1.checkable_total} checkable claim"
                    f"{'s' if a1.checkable_total != 1 else ''} supported."
                )
            else:
                fav = "The evidence does not clearly favor either party."
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(_safe(fav), styles["italic"]))

        story.append(Spacer(1, 3 * mm))

    # -----------------------------------------------------------------------
    # EXECUTIVE SUMMARY
    # -----------------------------------------------------------------------
    story += _section_head("Executive Summary", styles)
    story.append(Paragraph(_safe(report.summary), styles["body"]))
    story.append(Spacer(1, 3 * mm))

    # -----------------------------------------------------------------------
    # CLAIM VERDICTS TABLE
    # -----------------------------------------------------------------------
    story += _section_head("Claim Verdicts", styles)

    # Build claim → verdict map
    verdict_map = {v.claim_id: v for v in report.verdicts}

    # Same filtering logic as the frontend
    verdict_claims = [
        c for c in report.claims
        if not (c.statement_type == "opinion")
        and not (c.statement_type == "evidence_ref" and c.id not in verdict_map)
    ]

    if verdict_claims:
        # Column widths: Claim | Speaker | Verdict | Evidence | Reasoning
        PAGE_W = A4[0] - 2 * MARGIN
        col_widths = [
            PAGE_W * 0.24,   # Claim text
            PAGE_W * 0.11,   # Speaker
            PAGE_W * 0.13,   # Verdict
            PAGE_W * 0.27,   # Evidence quote
            PAGE_W * 0.25,   # Reasoning
        ]

        header_row = [
            Paragraph("Claim", styles["table_head"]),
            Paragraph("Speaker", styles["table_head"]),
            Paragraph("Verdict", styles["table_head"]),
            Paragraph("Evidence", styles["table_head"]),
            Paragraph("Reasoning", styles["table_head"]),
        ]

        rows = [header_row]
        for claim in verdict_claims:
            link = verdict_map.get(claim.id)
            verdict_str = (link.verdict if link else "insufficient_evidence")
            v_color = _verdict_color(verdict_str)
            v_label = _verdict_label(verdict_str)
            spk_name = _speaker_name(claim.speaker_id)

            # Claim: use claim.text (English pronoun-resolved restatement)
            claim_text = _safe(claim.text)
            stmt_type = (claim.statement_type or "").replace("_", " ").title()
            claim_para = Paragraph(
                f"{claim_text}<br/>"
                f'<font size="7" color="{_C_MUTED.hexval()}">{stmt_type}</font>',
                styles["table_cell"],
            )

            spk_para = Paragraph(_safe(spk_name), styles["table_cell"])

            verdict_para = Paragraph(
                f'<font name="Helvetica-Bold" color="{v_color.hexval()}">{v_label}</font>',
                styles["table_cell"],
            )

            # Evidence: English quote from link (verbalization output — already English)
            if link and link.quote:
                ev_para = Paragraph(
                    f'&#8220;{_safe(link.quote)}&#8221;',
                    styles["table_cell_italic"],
                )
            else:
                ev_para = Paragraph("—", styles["small"])

            # Reasoning: already English
            reason_para = Paragraph(_safe(link.reasoning if link else ""), styles["table_cell"])

            rows.append([claim_para, spk_para, verdict_para, ev_para, reason_para])

        verdicts_table = Table(rows, colWidths=col_widths, repeatRows=1)
        verdicts_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f1f5")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _C_ROW_ALT]),
            # Light row separators only — no heavy grid
            ("LINEBELOW", (0, 0), (-1, -1), 0.3, _C_DIVIDER),
            ("LINEAFTER", (0, 0), (-1, -1), 0, _C_DIVIDER),   # no vertical lines
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(verdicts_table)
    else:
        story.append(
            Paragraph("No checkable claims were extracted from this session.", styles["small"])
        )

    story.append(Spacer(1, 4 * mm))

    # -----------------------------------------------------------------------
    # CONTRADICTIONS
    # -----------------------------------------------------------------------
    contradictions = report.contradictions or []
    if contradictions:
        story += _section_head(f"Contradictions ({len(contradictions)})", styles)
        claim_map = {c.id: c for c in report.claims}

        for i, flag in enumerate(contradictions, 1):
            ca = claim_map.get(flag.claim_id_a)
            cb = claim_map.get(flag.claim_id_b)

            block: list = [
                Paragraph(
                    f'<b>{i}. {_safe(flag.description)}</b>',
                    styles["small_bold"],
                ),
                Spacer(1, 1 * mm),
            ]

            # Side-by-side stacked: two mini-paragraphs
            for claim_obj in (ca, cb):
                if claim_obj is None:
                    continue
                spk = _speaker_name(claim_obj.speaker_id)
                block.append(
                    Paragraph(
                        f'<b>{_safe(spk)}:</b> {_safe(claim_obj.text)}',
                        styles["table_cell"],
                    )
                )
            block.append(Spacer(1, 2 * mm))

            story.append(KeepTogether(block))

        story.append(Spacer(1, 3 * mm))

    # -----------------------------------------------------------------------
    # COMMON GROUND
    # -----------------------------------------------------------------------
    agreements = report.agreements or []
    if agreements:
        story += _section_head("Common Ground", styles)
        for agreement in agreements:
            story.append(
                Paragraph(f"&#8226; {_safe(agreement)}", styles["body_left"])
            )
        story.append(Spacer(1, 4 * mm))

    # -----------------------------------------------------------------------
    # DISCLAIMER FOOTER (last flowable — page footers handled by canvas callback)
    # -----------------------------------------------------------------------
    story.append(Spacer(1, 6 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_C_DIVIDER))
    story.append(Spacer(1, 2 * mm))
    story.append(
        Paragraph(
            "This report does not declare a winner — verdicts reflect available evidence only.",
            styles["disclaimer"],
        )
    )

    # -----------------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------------
    doc.build(
        story,
        onFirstPage=_FooterCanvas,
        onLaterPages=_FooterCanvas,
    )

    return buf.getvalue()
