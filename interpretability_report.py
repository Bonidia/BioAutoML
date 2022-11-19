import logging
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from itertools import zip_longest
from os.path import join, basename, exists
from sys import stdout

report_handler = logging.StreamHandler(stream=stdout)
report_handler.setLevel(logging.WARNING)
report_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))

report_logger = logging.getLogger(basename(__file__))
report_logger.addHandler(report_handler)

REPORT_MAIN_TITLE_MULTICLASS = "Model Interpretability Report (Multiclass)"
REPORT_SHAP_PREAMBLE = (
	"All of interpretability are based in the SHAP method, in which it calculates what's the importance "
	"level for each feature in the classification process. It uses shapley values, a game theory concept, "
	"as a descriptive metric to create an hierarquical structure between the features."
)
REPORT_SUMMARY_TITLE = "Summary Plots"
REPORT_SHAP_SUMMARY_1 = (
	"The above plot is called Summary plot and it shows, for each class, how low/high values of each feature "
	"contributed for the classification with that class. The features are ranked from most descriptive to "
	"least descriptive. This plot is a summarization of all the entries in the test set."
)
REPORT_SHAP_SUMMARY_2 = (
	"An impact with positive SHAP value means that a high (red dots), medium (purple dots) or low (blue dots) "
	"feature value contributes positively of an entry to be classified with that class. The inverse happens with " 
	"negative SHAP values." 
)
REPORT_WATERFALL_TITLE = "Waterfall Plots"
REPORT_SHAP_WATERFALL = lambda n_samples: (
	"A Waterfall plot shows, for some entries, which of the features contributed more for it to be classified with "
	f"its classified class. In this case, {n_samples} samples for each class were chosen randomly to be analyzed."
)

REPORT_MAIN_TITLE = "Model Interpretability Report (BioAutoML)"
REPORT_SHAP_PREAMBLE = "SHAP: For each sample the SHAP do calculate the feature importance for the classification decision."

REPORT_SHAP_BAR = """
This graph shows the average contribution of each feature, for then highlighting the best features for the model. 
Through this graph it is possible to understand which are the features most important for the problem. 
"""

REPORT_SHAP_BEESWARM= """
Each line in this graph represents a feature and each dot a sample of the trainament conjunction. 
Through this graph it is possible to try to establish a correlation between the value of the sample, being high or low, 
with your contribution to the prediction. 
"""

REPORT_SHAP_WATERFALL = """
Each graph above it is referent to a specific sample, being that the title describes the sample label. 
Each line shows a feature, on the left side can see the sample value for this feature and in the colorful bars can see the contribution value for the classification in this class. 
And can see the limite E[f(x)], values below this number belong one class and values above this same number belong the other class. 
"""


make_bold = lambda s: f"<b>{s}</b>"
make_font_size = lambda s, size: f"<font size={size}>{s}</font>"

class Report:
	styles = None
	story = None
	doc = None
	text_width = None

	def __init__(self, report_name, directory=".", lr_margin=float(0.5*inch), tb_margin=float(0.25*inch)):
		
		"""Create a new PDF report with filename 'report_name'"""
		
		self.styles = getSampleStyleSheet()
		self.story = []
		self.doc = SimpleDocTemplate(
			join(directory, report_name),
			leftMargin=lr_margin,
			rightMargin=lr_margin,
			topMargin=tb_margin,
			bottomMargin=tb_margin,
			pagesize=A4
		)
		self.styles.add(ParagraphStyle(name='Justify', fontName="Helvetica", 
						alignment=TA_JUSTIFY, firstLineIndent=0.3*inch))
		self.styles.add(ParagraphStyle(name='Center', fontName="Helvetica", 
						alignment=TA_CENTER))
		
		page_width, _ = A4
		self.text_width = page_width - 2*lr_margin
		

	def create_paragraph_style(self, style_name, font_name, alignment):
		
		"""Create a new text style into current report's style sheet"""
		
		self.styles.add(
			ParagraphStyle(name=style_name, fontName=font_name, alignment=alignment)
		)


	def insert_doc_header(self, title, font_size=16, logo_fig=None, pre_margin=1, pos_margin=18, bold=True):
		
		"""Insert a header with given title and logo on the file"""

		if not logo_fig:
			self.insert_text_on_doc(title, font_size=font_size, style='Center', pos_margin=pos_margin, bold=bold)
			return

		if pre_margin > 0:
			self.story.append(Spacer(1, pre_margin))
		else:
			report_logger.warning(f"'pre_margin' can't be negative. Ignoring it " +\
							  	  f"and using default value (1). [pre_margin={pre_margin}]")

		assert exists(logo_fig), f"Logo figure in path {logo_fig} does not exist."
		fmt = make_font_size(make_bold(title) if bold else title, font_size)
		self.story.append(Table(
			[
				[Paragraph(fmt, self.styles['Center']), 
			  	Image(logo_fig, width=0.15*self.text_width, height=10*inch, kind='proportional')]
			],
			style=TableStyle([('VALIGN', (0,0), (1,0), 'MIDDLE')]), 
			colWidths=[0.8*self.text_width, 0.2*self.text_width]
		))

		if pos_margin > 0:
			self.story.append(Spacer(1, pos_margin))
		else:
			report_logger.warning(f"'pos_margin' can't be negative. Ignoring it " +\
							 	  f"and using default value (18). [pos_margin={pos_margin}]")


	def insert_text_on_doc(self, text, font_size=12, style='Justify', pre_margin=1, pos_margin=12, bold=False):
		
		"""Insert a new paragraph on report with given text customization"""

		if pre_margin > 0:
			self.story.append(Spacer(1, pre_margin))
		else:
			report_logger.warning(f"'pre_margin' can't be negative. Ignoring it " +\
							  	  f"and using default value (1). [pre_margin={pre_margin}]")

		assert font_size > 0, f"Error: 'font_size' can't be negative. Aborted. [font_size={font_size}]"
		fmt = make_font_size(make_bold(text) if bold else text, font_size)
		self.story.append(Paragraph(fmt, self.styles[style]))

		if pos_margin > 0:
			self.story.append(Spacer(1, pos_margin))
		else:
			report_logger.warning(f"'pos_margin' can't be negative. Ignoring it " +\
							 	  f"and using default value (12). [pos_margin={pos_margin}]")


	def insert_figure_on_doc(self, fig_paths, pre_margin=1, pos_margin=24):
		"""
			Insert a list of figures pairwise into the report
			If the size of the list is odd, the last one will be centered
		"""
		pairwise = lambda iterable: list(zip_longest(*[iter(iterable)] * 2, fillvalue=None))

		for fig, fig2 in pairwise(fig_paths):
			assert exists(fig), f"Figure in path {fig} does not exist."

			if pre_margin > 0:
				self.story.append(Spacer(1, pre_margin))
			else:
				report_logger.warning(f"'pre_margin' can't be negative. Ignoring it " +\
									  f"and using default value (1). [pre_margin={pre_margin}]")

			if not fig2:
				self.story.append(Image(fig, width=0.5*self.text_width, height=10*inch, kind="proportional"))
			else:
				assert exists(fig2), f"Figure in path {fig2} does not exist."
				self.story.append(Table(
					[[Image(fig, width=0.5*self.text_width, height=10*inch, kind="proportional"),\
					  Image(fig2, width=0.5*self.text_width, height=10*inch, kind="proportional")]]
				))

			if pos_margin > 0:
				self.story.append(Spacer(1, pos_margin))
			else:
				report_logger.warning(f"'pos_margin' can't be negative. Ignoring it " +\
									  f"and using default value (12). [pos_margin={pos_margin}]")


	def build(self):

		"""Build report from built story list"""

		self.doc.build(self.story)
