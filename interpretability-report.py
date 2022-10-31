import logging
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from os.path import join, basename
from sys import stdout

mc_handler = logging.StreamHandler(stream=stdout)
mc_handler.setLevel(logging.WARNING)
mc_handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))

mc_logger = logging.getLogger(basename(__file__))
mc_logger.addHandler(mc_handler)

class Report:
	def __init__(self, report_name, directory=".", lr_margin=12, tb_margin=12):
		self.styles = getSampleStyleSheet()
		self.story = []
		self.doc = SimpleDocTemplate(
			join(directory, report_name),
			leftMargin=lr_margin,
			rightMargin=lr_margin,
			topMargin=tb_margin,
			bottomMargin=tb_margin
		)
		""" Default styles """
		self.styles.add(ParagraphStyle(name='Justify', fontName="Courier",alignment=TA_JUSTIFY))
		self.styles.add(ParagraphStyle(name='Center', fontName="Courier",alignment=TA_CENTER))
		

	def create_paragraph_style(self, style_name, font_name, alignment):
		self.styles.add(
			ParagraphStyle(name=style_name, fontName=font_name, alignment=alignment)
		)


	def insert_text_on_doc(self, text, font_size=12, style='Justify', pre_margin=1, pos_margin=12):
		if pre_margin > 0:
			self.story.append(Spacer(1, pre_margin))
		else:
			mc_logger.warning(f"'pre_margin' can't be negative. Ignoring it " +\
							  f"and using default value (1). [pre_margin={pre_margin}]")

		assert font_size > 0, f"Error: 'font_size' can't be negative. Aborted. [font_size={font_size}]"
		fmt = f"<font size={font_size}>{text}</font>"
		self.story.append(Paragraph(fmt, self.styles[style]))

		if pos_margin > 0:
			self.story.append(Spacer(1, pos_margin))
		else:
			mc_logger.warning(f"'pos_margin' can't be negative. Ignoring it " +\
							  f"and using default value (12). [pos_margin={pos_margin}]")


	def insert_figure_on_doc(
			self, fig_path, title, title_size=14, description_size=10, 
			title_style='Center', description_style='Justify', image_width=4*inch, 
			image_height=3*inch, pre_margin=1, pos_margin=24
		):
		self.insert_text_on_doc(title, title_size, title_style)
		
		if pre_margin > 0:
			self.story.append(Spacer(1, pre_margin))
		else:
			mc_logger.warning(f"'pre_margin' can't be negative. Ignoring it " +\
							  f"and using default value (1). [pre_margin={pre_margin}]")
		
		assert image_width > 0, f"Error: 'image_width' can't be negative. Aborted. image_width=[{image_width}]"
		assert image_height > 0, f"Error: 'image_height' can't be negative. Aborted. image_height=[{image_height}]"
		self.story.append(Image(fig_path, width=image_width, height=image_height))
		
		if pos_margin > 0:
			self.story.append(Spacer(1, pos_margin))
		else:
			mc_logger.warning(f"'pos_margin' can't be negative. Ignoring it " +\
							  f"and using default value (12). [pos_margin={pos_margin}]")


	def build(self):
		self.doc.build(self.story)
