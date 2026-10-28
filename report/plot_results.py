"""ReportLab charts from verified validation evidence."""
import json, math
from pathlib import Path
from reportlab.graphics import renderPDF
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Circle, Drawing, String
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT.parent / 'code/evidence/validation_analysis.json').read_text(encoding='utf-8'))
mapping = {row['name']: row for row in data['methods']}
names = ['baseline-majority', 'fusion-embedding-only', 'text-no-arithmetic', 'final-validation']
drawing = Drawing(680, 244)
drawing.add(String(12, 228, '(a) Full and disagreement validation', fontName='Helvetica-Bold', fontSize=11))
bar = HorizontalBarChart()
bar.x, bar.y, bar.width, bar.height = 96, 62, 235, 145
bar.data = [[mapping[name]['accuracy'] for name in names], [mapping[name]['hard_accuracy'] for name in names]]
bar.categoryAxis.categoryNames = ['Majority vote', 'Embedding pair', 'Text features', 'Final fusion']
bar.categoryAxis.labels.fontSize = 8
bar.valueAxis.valueMin, bar.valueAxis.valueMax, bar.valueAxis.valueStep = 0, 100, 25
bar.valueAxis.labels.fontSize = 8
bar.bars[0].fillColor = colors.HexColor('#bfd6e9')
bar.bars[1].fillColor = colors.HexColor('#245b86')
bar.bars.strokeColor = None
bar.barLabelFormat = '%.1f'
bar.barLabels.fontSize = 7
bar.barLabels.nudge = 4
drawing.add(bar)
drawing.add(String(122, 38, 'Selection accuracy (%)', fontSize=9))
drawing.add(String(30, 17, 'Light: full validation n=798; dark: disagreement n=355', fontSize=8))
drawing.add(String(384, 228, '(b) Uncalibrated answer confidence', fontName='Helvetica-Bold', fontSize=11))
line = LinePlot()
line.x, line.y, line.width, line.height = 412, 62, 235, 145
line.data = [[(0, 0), (1, 1)]]
line.xValueAxis.valueMin = line.yValueAxis.valueMin = 0
line.xValueAxis.valueMax = line.yValueAxis.valueMax = 1
line.xValueAxis.valueSteps = line.yValueAxis.valueSteps = [0, .25, .5, .75, 1]
line.xValueAxis.labels.fontSize = line.yValueAxis.labels.fontSize = 8
line.lines[0].strokeColor = colors.HexColor('#87939f')
line.lines[0].strokeDashArray = [3, 3]
drawing.add(line)
for item in data['confidence_hard_subset']['bins']:
    drawing.add(Circle(412 + item['mean_confidence'] * 235, 62 + item['accuracy'] * 145,
                       1.4 + math.sqrt(item['n']) * .48,
                       fillColor=colors.HexColor('#b76539'), strokeColor=colors.white, strokeWidth=.5))
drawing.add(String(442, 38, 'Mean answer-group confidence', fontSize=9))
drawing.add(String(385, 17, 'Disagreement n=355; ECE=0.133; bubble size reflects count', fontSize=8))
drawing.add(String(390, 211, 'Empirical accuracy', fontSize=8))
output = ROOT.parent / 'code/.environment/report-figure.pdf'
output.parent.mkdir(parents=True, exist_ok=True)
renderPDF.drawToFile(drawing, str(output))
print(output)
