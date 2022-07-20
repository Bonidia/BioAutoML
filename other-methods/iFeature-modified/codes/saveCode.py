#!/usr/bin/env python
#_*_coding:utf-8_*_

import sys

def savetsv(encodings, file = 'encoding.csv'):
	with open(file, 'w') as f:
		if encodings == 0:
			f.write('Descriptor calculation failed.')
		else:
			for i in range(len(encodings[0]) - 1):
				f.write(encodings[0][i] + ',')
			f.write(encodings[0][-1] + '\n')
			for i in encodings[1:]:
				f.write(i[0] + ',')
				for j in range(1, len(i) - 1):
					f.write(str(float(i[j])) + ',')
				f.write(str(float(i[len(i)-1])) + '\n')
	return None
