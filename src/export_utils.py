from __future__ import annotations
from io import BytesIO
import json
import pandas as pd

def csv_bytes(df): return df.to_csv(index=False).encode("utf-8")
def excel_bytes(df, sheet_name="results"):
    b=BytesIO()
    with pd.ExcelWriter(b,engine="openpyxl") as w: df.to_excel(w,index=False,sheet_name=sheet_name[:31])
    return b.getvalue()
def json_bytes(obj): return json.dumps(obj,indent=2,default=str).encode("utf-8")
def pdf_bytes(title, sections):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    b=BytesIO(); doc=SimpleDocTemplate(b,pagesize=letter); styles=getSampleStyleSheet(); story=[Paragraph(title,styles["Title"]),Spacer(1,12)]
    for heading,text in sections: story += [Paragraph(heading,styles["Heading2"]),Paragraph(str(text).replace("&","&amp;"),styles["BodyText"]),Spacer(1,8)]
    doc.build(story); return b.getvalue()
