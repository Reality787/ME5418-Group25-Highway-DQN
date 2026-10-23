"""从可编辑英文稿生成恰好两页的报告；只需独立文档工具环境中的 ReportLab。"""
from pathlib import Path
import argparse
from html import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Flowable

class FlowDiagram(Flowable):
    """使用PDF矢量图展示实际学习流程，不依赖图片或第三方作图服务。"""
    def __init__(self,width):
        Flowable.__init__(self);self.width=width;self.height=55
    def draw(self):
        c=self.canv;w=(self.width-24)/3
        labels=['34-D state -> online Q','Mask + epsilon-greedy','Real step + replay','Legal TD target','SmoothL1 + Adam','Periodic target copy']
        for i,label in enumerate(labels):
            row,col=divmod(i,3);x=col*(w+12);y=31-row*27
            c.setFillColor(colors.HexColor('#eef3f8'));c.setStrokeColor(colors.HexColor('#9bb2c7'))
            c.roundRect(x,y,w,21,3,fill=1,stroke=1)
            c.setFillColor(colors.HexColor('#173b57'));c.setFont('ReportSans',8.5);c.drawCentredString(x+w/2,y+7,label)
            if col<2:
                c.line(x+w+2,y+10,x+w+10,y+10);c.line(x+w+7,y+12,x+w+10,y+10);c.line(x+w+7,y+8,x+w+10,y+10)
        c.line(self.width-8,30,self.width-8,26);c.line(self.width-8,26,8,26);c.line(8,26,8,25)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/learning_agent_cr_report.md'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    # 嵌入真实字体，避免查看器替换标准字体造成字距错乱。
    font_dir=Path('/usr/share/fonts/truetype/liberation')
    pdfmetrics.registerFont(TTFont('ReportSans',str(font_dir/'LiberationSans-Regular.ttf')))
    pdfmetrics.registerFont(TTFont('ReportSans-Bold',str(font_dir/'LiberationSans-Bold.ttf')))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportTitle',fontName='ReportSans-Bold',fontSize=18,leading=21,textColor=colors.HexColor('#173b57'),spaceAfter=5))
    styles.add(ParagraphStyle(name='ReportBody',fontName='ReportSans',fontSize=10.1,leading=13.4,spaceAfter=6))
    styles.add(ParagraphStyle(name='ReportHeading',fontName='ReportSans-Bold',fontSize=11,leading=14,spaceBefore=8,spaceAfter=5,textColor=colors.HexColor('#173b57')))
    styles.add(ParagraphStyle(name='ReportSmall',fontName='ReportSans',fontSize=8.0,leading=10,spaceAfter=3))
    styles.add(ParagraphStyle(name='ReportSubtitle',fontName='ReportSans',fontSize=9.3,leading=12,spaceAfter=4,textColor=colors.HexColor('#455b6c')))
    width=A4[0]-92;story=[]
    for block in args.source.read_text().split('\n\n'):
        block=block.strip()
        if not block:continue
        if block=='[PAGEBREAK]':story.append(PageBreak());continue
        if block=='[FLOW]':story.extend([FlowDiagram(width),Spacer(1,5)]);continue
        if block=='[TD]':continue
        if block=='[HYPERPARAMETERS]':
            rows=[['Learning rule','Fixed engineering configuration'],['Optimizer / loss','Adam 1e-4 / SmoothL1Loss; gradient norm <= 10'],['Replay / schedule','Capacity 100,000; batch 64; learn from step 1,000, every 4 steps'],['Target / exploration','Copy every 1,000 updates; epsilon 1.0 to 0.05 over first 60%']]
            table=Table([[Paragraph(escape(x),styles['ReportSmall']) for x in row] for row in rows],colWidths=[105,width-105])
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eef3f8')),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#c6d3df')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
            story.extend([table,Spacer(1,7)]);continue
        if block.startswith('# '):
            lines=block.splitlines();story.append(Paragraph(escape(lines[0][2:]),styles['ReportTitle']))
            for line in lines[1:]:story.append(Paragraph(escape(line),styles['ReportSubtitle']))
        elif block.startswith('## '):
            lines=block.splitlines();story.append(Paragraph(escape(lines[0][3:]),styles['ReportHeading']))
            if len(lines)>1:
                if lines[0]=='## References':
                    for line in lines[1:]:story.append(Paragraph(escape(line),styles['ReportSmall']))
                else:story.append(Paragraph(escape(' '.join(lines[1:])),styles['ReportBody']))
        else:
            style=styles['ReportSmall'] if block.startswith('[1]') else styles['ReportBody']
            story.append(Paragraph(escape(block).replace('\n','<br/>'),style))
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#c6d3df'));canvas.line(46,36,A4[0]-46,36)
        canvas.setFont('ReportSans',8);canvas.setFillColor(colors.HexColor('#455b6c'))
        canvas.drawString(46,24,'ME5418 | Group 25 | Learning Agent C&R');canvas.drawRightString(A4[0]-46,24,f'{doc.page} / 2');canvas.restoreState()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(args.output),pagesize=A4,leftMargin=46,rightMargin=46,topMargin=39,bottomMargin=46,title='ME5418 Group 25 - Learning Agent C&R',author='Group 25').build(story,onFirstPage=footer,onLaterPages=footer)
    print(args.output)
if __name__=='__main__':main()
