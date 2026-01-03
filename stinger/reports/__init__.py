"""
Stinger V2 - PDF Report Generator
Clinical-grade PDF reports with TrustCat branding

trustcat.ai
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# We'll use reportlab for PDF generation
# pip install reportlab


class PDFReportGenerator:
    """
    Generate professional clinical PDF reports.
    Includes TrustCat branding, findings, and cryptographic proof.
    """
    
    def __init__(
        self,
        template_dir: Optional[Path] = None,
        assets_dir: Optional[Path] = None,
    ):
        self.template_dir = template_dir
        self.assets_dir = assets_dir
        
        # Colors (TrustCat green theme)
        self.colors = {
            "primary": (74/255, 222/255, 128/255),  # #4ade80
            "dark": (2/255, 6/255, 23/255),          # #020617
            "text": (30/255, 41/255, 59/255),        # #1e293b
            "muted": (100/255, 116/255, 139/255),    # #64748b
            "white": (1, 1, 1),
            "black": (0, 0, 0),
        }
    
    async def generate(
        self,
        output_path: Path,
        study_type: str,
        patient_id: str,
        study_date: datetime,
        findings: Dict[str, Any],
        report_text: str,
        images: List[Any] = None,
        job_id: str = "",
        proof: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate a clinical PDF report.
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                Image as RLImage, PageBreak, HRFlowable
            )
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        except ImportError:
            # Fallback to simple text file if reportlab not available
            await self._generate_text_report(output_path, study_type, patient_id, 
                                             study_date, findings, report_text, job_id)
            return
        
        # Create document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
        )
        
        # Styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        styles.add(ParagraphStyle(
            name='TrustCatTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#4ade80'),
            spaceAfter=20,
            alignment=TA_CENTER,
        ))
        
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1e293b'),
            spaceBefore=15,
            spaceAfter=10,
            borderWidth=0,
            borderColor=colors.HexColor('#4ade80'),
            borderPadding=5,
        ))
        
        # Override existing BodyText style
        styles['BodyText'].fontSize = 11
        styles['BodyText'].textColor = colors.HexColor('#334155')
        styles['BodyText'].spaceBefore = 5
        styles['BodyText'].spaceAfter = 5
        styles['BodyText'].leading = 14
        
        styles.add(ParagraphStyle(
            name='Finding',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#1e293b'),
            leftIndent=20,
            bulletIndent=10,
            spaceBefore=3,
            spaceAfter=3,
        ))
        
        styles.add(ParagraphStyle(
            name='Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#64748b'),
            alignment=TA_CENTER,
        ))
        
        styles.add(ParagraphStyle(
            name='Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#94a3b8'),
            spaceBefore=20,
            spaceAfter=10,
            borderWidth=1,
            borderColor=colors.HexColor('#e2e8f0'),
            borderPadding=10,
            backColor=colors.HexColor('#f8fafc'),
        ))
        
        # Build story (content)
        story = []
        
        # =====================================================================
        # HEADER
        # =====================================================================
        story.append(Paragraph("🐱 TRUSTCAT MEDICAL AI REPORT", styles['TrustCatTitle']))
        story.append(Spacer(1, 10))
        
        # Report info table
        info_data = [
            ["Patient ID:", patient_id, "Study Type:", study_type.upper()],
            ["Study Date:", study_date.strftime("%Y-%m-%d %H:%M"), "Job ID:", job_id[:20] if job_id else "N/A"],
            ["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Version:", "Stinger V2"],
        ]
        
        info_table = Table(info_data, colWidths=[1.2*inch, 2*inch, 1.2*inch, 2*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#64748b')),
            ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor('#64748b')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#1e293b')),
            ('TEXTCOLOR', (3, 0), (3, -1), colors.HexColor('#1e293b')),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
            ('FONTNAME', (3, 0), (3, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(info_table)
        
        story.append(HRFlowable(
            width="100%",
            thickness=2,
            color=colors.HexColor('#4ade80'),
            spaceBefore=15,
            spaceAfter=15,
        ))
        
        # =====================================================================
        # FINDINGS SUMMARY
        # =====================================================================
        if findings:
            story.append(Paragraph("FINDINGS SUMMARY", styles['SectionHeader']))
            
            # Summary
            if findings.get("summary"):
                story.append(Paragraph(findings["summary"], styles['BodyText']))
                story.append(Spacer(1, 10))
            
            # Pathologies table
            if findings.get("pathologies"):
                story.append(Paragraph("Detected Pathologies:", styles['BodyText']))
                for pathology in findings["pathologies"]:
                    if isinstance(pathology, dict):
                        text = f"• {pathology.get('name', pathology)}: {pathology.get('severity', 'present')}"
                    else:
                        text = f"• {pathology}"
                    story.append(Paragraph(text, styles['Finding']))
                story.append(Spacer(1, 10))
            
            # Measurements
            if findings.get("measurements"):
                story.append(Paragraph("Measurements:", styles['BodyText']))
                for key, value in findings["measurements"].items():
                    story.append(Paragraph(f"• {key}: {value}", styles['Finding']))
                story.append(Spacer(1, 10))
            
            # Confidence
            if findings.get("confidence"):
                conf = findings["confidence"]
                if isinstance(conf, float):
                    conf_pct = f"{conf * 100:.1f}%" if conf <= 1 else f"{conf:.1f}%"
                else:
                    conf_pct = str(conf)
                story.append(Paragraph(f"AI Confidence: {conf_pct}", styles['BodyText']))
        
        story.append(Spacer(1, 20))
        
        # =====================================================================
        # FULL REPORT TEXT
        # =====================================================================
        story.append(Paragraph("DETAILED REPORT", styles['SectionHeader']))
        
        # Split report into paragraphs and render
        if report_text:
            paragraphs = report_text.split("\n\n")
            for para in paragraphs:
                if para.strip():
                    # Check for headers
                    if para.strip().isupper() and len(para.strip()) < 50:
                        story.append(Paragraph(para.strip(), styles['SectionHeader']))
                    else:
                        story.append(Paragraph(para.strip(), styles['BodyText']))
        else:
            story.append(Paragraph("No detailed report generated.", styles['BodyText']))
        
        story.append(Spacer(1, 20))
        
        # =====================================================================
        # RECOMMENDATIONS
        # =====================================================================
        if findings.get("recommendations"):
            story.append(Paragraph("RECOMMENDATIONS", styles['SectionHeader']))
            for rec in findings["recommendations"]:
                story.append(Paragraph(f"• {rec}", styles['Finding']))
            story.append(Spacer(1, 20))
        
        # =====================================================================
        # CRYPTOGRAPHIC PROOF
        # =====================================================================
        if proof:
            story.append(Paragraph("VERIFICATION", styles['SectionHeader']))
            
            proof_data = [
                ["Merkle Root:", proof.get("merkle_root", "N/A")[:40] + "..."],
                ["IPFS CID:", proof.get("ipfs_cid", "N/A")],
                ["Signer:", proof.get("signer", "N/A")],
                ["Chain ID:", str(proof.get("chain_id", 1))],
                ["Timestamp:", proof.get("timestamp", "N/A")],
            ]
            
            proof_table = Table(proof_data, colWidths=[1.5*inch, 5*inch])
            proof_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Courier'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#64748b')),
                ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#4ade80')),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#020617')),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ]))
            story.append(proof_table)
            story.append(Spacer(1, 20))
        
        # =====================================================================
        # DISCLAIMER
        # =====================================================================
        disclaimer_text = """
        <b>DISCLAIMER:</b> This report was generated by TrustCat Medical AI (Stinger V2) 
        and is intended for clinical decision support only. It does not constitute a medical 
        diagnosis. All findings should be verified by a qualified healthcare professional. 
        This AI system has been validated on real clinical data but is not FDA cleared. 
        Use professional judgment in patient care decisions.
        """
        story.append(Paragraph(disclaimer_text, styles['Disclaimer']))
        
        # =====================================================================
        # FOOTER
        # =====================================================================
        story.append(Spacer(1, 20))
        story.append(HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor('#e2e8f0'),
            spaceBefore=10,
            spaceAfter=10,
        ))
        
        footer_text = """
        Generated by TrustCat — Sovereign Medical AI Infrastructure<br/>
        trustcat.ai | stinger.swarmbee.eth | 💎 Diamond Hands Edition
        """
        story.append(Paragraph(footer_text, styles['Footer']))
        
        # Build PDF
        doc.build(story)
    
    async def _generate_text_report(
        self,
        output_path: Path,
        study_type: str,
        patient_id: str,
        study_date: datetime,
        findings: Dict[str, Any],
        report_text: str,
        job_id: str,
    ):
        """Fallback text report if reportlab not available"""
        lines = [
            "=" * 70,
            "TRUSTCAT MEDICAL AI REPORT",
            "=" * 70,
            "",
            f"Patient ID:  {patient_id}",
            f"Study Type:  {study_type}",
            f"Study Date:  {study_date.strftime('%Y-%m-%d %H:%M')}",
            f"Job ID:      {job_id}",
            f"Generated:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "-" * 70,
            "FINDINGS",
            "-" * 70,
            "",
        ]
        
        if findings.get("summary"):
            lines.append(findings["summary"])
            lines.append("")
        
        if findings.get("pathologies"):
            lines.append("Pathologies:")
            for p in findings["pathologies"]:
                lines.append(f"  • {p}")
            lines.append("")
        
        lines.extend([
            "-" * 70,
            "DETAILED REPORT",
            "-" * 70,
            "",
            report_text or "No detailed report generated.",
            "",
            "-" * 70,
            "DISCLAIMER",
            "-" * 70,
            "This report is for clinical decision support only.",
            "All findings should be verified by a qualified healthcare professional.",
            "",
            "=" * 70,
            "trustcat.ai | stinger.swarmbee.eth",
            "=" * 70,
        ])
        
        # Write as text (could also use fpdf2 to make PDF)
        text_path = output_path.with_suffix('.txt')
        with open(text_path, 'w') as f:
            f.write("\n".join(lines))
        
        # Also try to create PDF with simpler library
        try:
            from fpdf import FPDF
            
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Courier', size=10)
            
            for line in lines:
                pdf.cell(0, 5, txt=line[:90], ln=True)
            
            pdf.output(str(output_path))
        except ImportError:
            # Just keep the text file
            import shutil
            shutil.copy(text_path, output_path)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ['PDFReportGenerator']
