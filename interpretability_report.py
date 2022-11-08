import logging
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from itertools import zip_longest
from os.path import join, basename, exists
from sys import stdout

report_handler = logging.StreamHandler(stream=stdout)
report_handler.setLevel(logging.WARNING)
report_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))

report_logger = logging.getLogger(basename(__file__))
report_logger.addHandler(report_handler)

REPORT_MAIN_TITLE = "Model Interpretability Report (BioAutoML)"
REPORT_SHAP_PREAMBLE = "Método Shap: SHAP calcula para cada amostra qual a importância de cada  feature para a decisão de classificação."
REPORT_SHAP_SUMMARY = "Lorem ipsum"
REPORT_SHAP_WATERFALL = "Lorem ipsum"

class Report:
	def __init__(self, report_name, directory=".", lr_margin=12, tb_margin=12):
		"""
			Create a new PDF report with filename 'report_name'
		"""
		self.styles = getSampleStyleSheet()
		self.story = []
		self.doc = SimpleDocTemplate(
			join(directory, report_name),
			leftMargin=lr_margin,
			rightMargin=lr_margin,
			topMargin=tb_margin,
			bottomMargin=tb_margin
		)
		self.styles.add(ParagraphStyle(name='Justify', fontName="Courier",alignment=TA_JUSTIFY))
		self.styles.add(ParagraphStyle(name='Center', fontName="Courier",alignment=TA_CENTER))
		

	def create_paragraph_style(self, style_name, font_name, alignment):
		
		"""Create a new text style into current report's style sheet"""
		
		self.styles.add(
			ParagraphStyle(name=style_name, fontName=font_name, alignment=alignment)
		)


	def insert_text_on_doc(self, text, font_size=12, style='Justify', pre_margin=1, pos_margin=12):
		
		"""Insert a new paragraph on report with given text customization"""

		if pre_margin > 0:
			self.story.append(Spacer(1, pre_margin))
		else:
			report_logger.warning(f"'pre_margin' can't be negative. Ignoring it " +\
							  f"and using default value (1). [pre_margin={pre_margin}]")

		assert font_size > 0, f"Error: 'font_size' can't be negative. Aborted. [font_size={font_size}]"
		fmt = f"<font size={font_size}>{text}</font>"
		self.story.append(Paragraph(fmt, self.styles[style]))

		if pos_margin > 0:
			self.story.append(Spacer(1, pos_margin))
		else:
			report_logger.warning(f"'pos_margin' can't be negative. Ignoring it " +\
							  f"and using default value (12). [pos_margin={pos_margin}]")


	def insert_figure_on_doc(self, fig_paths, image_width=4*inch, image_height=3*inch, pre_margin=1, pos_margin=24):
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

			assert image_width > 0, f"Error: 'image_width' can't be negative. Aborted. image_width=[{image_width}]"
			assert image_height > 0, f"Error: 'image_height' can't be negative. Aborted. image_height=[{image_height}]"

			if not fig2:
				self.story.append(Image(fig, width=image_width, height=image_height, kind="proportional"))
			else:
				assert exists(fig2), f"Figure in path {fig2} does not exist."
				self.story.append(Table(
					[[Image(fig, width=image_width, height=image_height, kind="proportional"),\
					 Image(fig2, width=image_width, height=image_height, kind="proportional")]]
				))

			if pos_margin > 0:
				self.story.append(Spacer(1, pos_margin))
			else:
				report_logger.warning(f"'pos_margin' can't be negative. Ignoring it " +\
									  f"and using default value (12). [pos_margin={pos_margin}]")


	def build(self):

		"""Build report from built story list"""

		self.doc.build(self.story)