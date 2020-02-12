import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import networkx as nx
import itertools
import pandas as pd


# A, is adjacent matrix for hyper_net


class Take_pic:
	def __init__(self, A, tate=860,Tate = 220, yoko=1600, diameter_net=325, diameter_of_each_node=8, font_setting = False):
		self.net_name_shift = 60
		self.num_node     = A.shape[0]/2
		self.diameter_net = diameter_net
		self.centre1      = np.array( [yoko/4  ,tate/2+Tate] )
		self.centre2      = np.array( [yoko*3/4,tate/2+Tate] )
		self.diameter_of_each_node = diameter_of_each_node
		self.yoko = yoko
		self.tate = tate
		self.Tate = Tate

		# set font
		self.font_setting = font_setting
		if self.font_setting == True:
				path = "/System/Library/Fonts/"
				name = "Arial.ttf"
				self.font_name = path + name
				self.font  = ImageFont.truetype(self.font_name, 55)
				self.font1 = ImageFont.truetype(self.font_name, 25) # for title

		# setting adjacent matrix
		self.A  = A
		self.A1 = A[:self.num_node, :self.num_node]
		self.A2 = A[self.num_node:, self.num_node:]

		# set the position of each node in network A1 and A2
		arr = np.arange(self.num_node)
		X1  = diameter_net * ( np.cos( 2*np.pi*arr/self.num_node - np.pi/2  ) )
		Y1  = diameter_net * ( np.sin( 2*np.pi*arr/self.num_node - np.pi/2  ) )
		X2  = diameter_net * ( np.cos( 2 *np.pi*arr/self.num_node - np.pi/2 ) )
		Y2  = diameter_net * ( np.sin( 2 *np.pi*arr/self.num_node - np.pi/2 ) )

		# position for nodes
		self.x1 = X1 + self.centre1[0]
		self.y1 = Y1 + self.centre1[1]
		self.x2 = X2 + self.centre2[0]
		self.y2 = Y2 + self.centre2[1]
		for i in xrange(self.num_node):
			if i%2 == 0:
				self.x1[i] = 1.1*X1[i] + self.centre1[0]
				self.y1[i] = 1.1*Y1[i] + self.centre1[1]
				self.x2[i] = 1.1*X2[i] + self.centre2[0]
				self.y2[i] = 1.1*Y2[i] + self.centre2[1]



		# position for name
		self.x1_for_name = np.empty(self.num_node)
		self.y1_for_name = np.empty(self.num_node)
		self.x2_for_name = np.empty(self.num_node)
		self.y2_for_name = np.empty(self.num_node)
		for i in range(self.num_node):
			if X1[i] < 0:
				self.x1_for_name[i] = X1[i]*1.1 + self.centre1[0]
			else :
				self.x1_for_name[i] = X1[i]*1.05 + self.centre1[0]
		for i in range(self.num_node):
			if X2[i] < 0:
				self.x2_for_name[i] = X2[i]*1.1 + self.centre2[0]
			else:
				self.x2_for_name[i] = X2[i]*1.05 + self.centre2[0]
		for i in range(self.num_node):
			if Y1[i] < 0:
				self.y1_for_name[i] = Y1[i]*1.1 + self.centre1[1]
			else:
				self.y1_for_name[i] = Y1[i]*1.1 + self.centre1[1]
		for i in range(self.num_node):
			if Y2[i] < 0:
				self.y2_for_name[i] = Y2[i]*1.1 + self.centre2[1]
			else:
				self.y2_for_name[i] = Y2[i]*1.1 + self.centre2[1]

		self.pos1 = np.c_[self.x1 , self.y1]
		self.pos2 = np.c_[self.x2 , self.y2]

		# preparation for getting indices of conneted nodes
		self.count1 = int( np.sum( np.sum(self.A1) ))
		self.count2 = int( np.sum( np.sum(self.A2) ))
		self.count3 = int( np.sum( np.sum( self.A[self.num_node:,:self.num_node] ) ))
		self.ind1   = np.zeros(self.count1*2).reshape(self.count1,2)
		self.ind2   = np.zeros(self.count2*2).reshape(self.count2,2)
		self.ind3   = np.zeros(self.count3*2).reshape(self.count3,2)

		# get the intra connection
		count  = 0
		for i, j in itertools.product( range(self.num_node), range(self.num_node) ):
			if self.A1[i,j] == 1:
				self.ind1[count,0] = int(i)
				self.ind1[count,1] = int(j)
				count             += 1
		count = 0
		for i, j in itertools.product( range(self.num_node), range(self.num_node) ):
			if self.A2[i,j] == 1:
				self.ind2[count,0] = int(i)
				self.ind2[count,1] = int(j)
				count             += 1
		# get the hyper connection
		count = 0
		for i, j in itertools.product( range(self.num_node), range(self.num_node) ):
			if self.A[i+self.num_node,j] == 1:
				self.ind3[count,0] = i
				self.ind3[count,1] = j
				count             += 1

		# set parameters for legend and that of colour
		self.black = (0,0,0)
		self.red   = (200,100,100)
		self.blue  = (100,100,200)

		5 * self.diameter_of_each_node

		self.black_legend_pos = np.array( [5.2*self.yoko/7, 30+20] )
		self.red_legend_pos   = np.array( [5.2*self.yoko/7, 55+30+7 * self.diameter_of_each_node] )
		self.blue_legend_pos  = np.array( [5.2*self.yoko/7, 80+40+14 * self.diameter_of_each_node] )

		self.legend_name_black = "normal node"
		self.legend_name_red   = "sensor node"
		self.legend_name_blue  = "motor node"


	# illustrate intra network
	def Independent_net_pic(self, DIRECTORY, title = "Independent_net_pic"):
		print "def Independent_net_pic(self, title = Independent_net_pic)"
		img  = Image.new("RGB", (self.yoko, self.tate), color=(255, 255, 255))
		draw = ImageDraw.Draw(img)

		# drawing connection
		for i in range(self.count1):
			draw.line( ((self.pos1[int(self.ind1[i,0]),0 ],self.pos1[int(self.ind1[i,0]),1 ]),(self.pos1[int(self.ind1[i,1]),0 ],self.pos1[int(self.ind1[i,1]),1 ])),fill=(0,0,0),width = 3 )
		for i in range(self.count2):
			draw.line( ((self.pos2[int(self.ind2[i,0]),0 ],self.pos2[int(self.ind2[i,0]),1 ]),(self.pos2[int(self.ind2[i,1]),0 ],self.pos2[int(self.ind2[i,1]),1 ])),fill=(0,0,0),width = 3   )

		# drawing nodes
		for i in range(self.num_node):
			draw.ellipse( ((tuple(self.pos1[i] - self.diameter_of_each_node )), ( tuple(self.pos1[i] + self.diameter_of_each_node) )), fill = self.red  , outline=None )
			draw.ellipse( ((tuple(self.pos2[i] - self.diameter_of_each_node )), ( tuple(self.pos2[i] + self.diameter_of_each_node) )), fill = self.blue , outline=None )

			# text
			if self.font_setting == True:
				draw.text( (self.x1_for_name[i], self.y1_for_name[i]) , str(i+1), font = self.font ,fill='#000')
				draw.text( (self.x2_for_name[i], self.y2_for_name[i]) , str(i+1), font = self.font ,fill='#000')
				draw.text( (centre_Left_net_name(self, draw).x , self.centre1[1]+self.diameter_net + self.net_name_shift) , "NET1", font = self.font ,fill='#000')
				draw.text( (centre_Right_net_name(self, draw).x, self.centre2[1]+self.diameter_net + self.net_name_shift) , "NET2", font = self.font ,fill='#000')
				draw.text( ( centre_title(self, draw , title).x , 50) , title, font = self.font1, fill='#000')
			else:
				draw.text( (self.x1_for_name[i], self.y1_for_name[i]) , str(i+1), fill='#000')
				draw.text( (self.x2_for_name[i], self.y2_for_name[i]) , str(i+1), fill='#000')
				draw.text( (centre_Left_net_name(self, draw).x , self.centre1[1]+self.diameter_net + self.net_name_shift) , "NET1", fill='#000')
				draw.text( (centre_Right_net_name(self, draw).x, self.centre2[1]+self.diameter_net + self.net_name_shift) , "NET2", fill='#000')
				draw.text( ( centre_title(self, draw , title).x , 50) , title, fill='#000')

		os.chdir(DIRECTORY)
		img.save( title + "_.png" )

	def for_paper_Independent_net_pic(self, DIRECTORY, title = "Independent_net_pic"):
		print "def Independent_net_pic(self, title = Independent_net_pic)"
		img  = Image.new("RGB", (self.yoko, self.tate + self.Tate), color=(255, 255, 255))
		draw = ImageDraw.Draw(img)

		# drawing connection
		for i in range(self.count1):
			draw.line( ((self.pos1[int(self.ind1[i,0]),0 ],self.pos1[int(self.ind1[i,0]),1 ]),(self.pos1[int(self.ind1[i,1]),0 ],self.pos1[int(self.ind1[i,1]),1 ])),fill=(200,200,200),width = 3 )
		for i in range(self.count2):
			draw.line( ((self.pos2[int(self.ind2[i,0]),0 ],self.pos2[int(self.ind2[i,0]),1 ]),(self.pos2[int(self.ind2[i,1]),0 ],self.pos2[int(self.ind2[i,1]),1 ])),fill=(200,200,200),width = 3   )

		# drawing nodes
		for i in range(self.num_node):
			draw.ellipse( ((tuple(self.pos1[i] - self.diameter_of_each_node )), ( tuple(self.pos1[i] + self.diameter_of_each_node) )), fill = (0,0,0)  , outline=None )
			draw.ellipse( ((tuple(self.pos2[i] - self.diameter_of_each_node )), ( tuple(self.pos2[i] + self.diameter_of_each_node) )), fill = (0,0,0) , outline=None )

			"""
			# text
			if self.font_setting == True:
				draw.text( (self.x1_for_name[i], self.y1_for_name[i]) , str(i+1), font = self.font ,fill='#000')
				draw.text( (self.x2_for_name[i], self.y2_for_name[i]) , str(i+1), font = self.font ,fill='#000')
				draw.text( (centre_Left_net_name(self, draw).x , self.centre1[1]+self.diameter_net + self.net_name_shift) , "NET1", font = self.font ,fill='#000')
				draw.text( (centre_Right_net_name(self, draw).x, self.centre2[1]+self.diameter_net + self.net_name_shift) , "NET2", font = self.font ,fill='#000')
				draw.text( ( centre_title(self, draw , title).x , 50) , title, font = self.font1, fill='#000')
			else:
				draw.text( (self.x1_for_name[i], self.y1_for_name[i]) , str(i+1), fill='#000')
				draw.text( (self.x2_for_name[i], self.y2_for_name[i]) , str(i+1), fill='#000')
				draw.text( (centre_Left_net_name(self, draw).x , self.centre1[1]+self.diameter_net + self.net_name_shift) , "NET1", fill='#000')
				draw.text( (centre_Right_net_name(self, draw).x, self.centre2[1]+self.diameter_net + self.net_name_shift) , "NET2", fill='#000')
				draw.text( ( centre_title(self, draw , title).x , 50) , title, fill='#000')
			"""

		os.chdir(DIRECTORY)
		img.save( title + "_.png" )

	# illustrate hyper network
	def Connected_net_pic(self, DIRECTORY, title = "Connected_net_pic"):
		print "def Connected_net_pic(self, title = Connected_net_pic):"
		img  = Image.new("RGB", (self.yoko, self.tate), color=(255, 255, 255))
		draw = ImageDraw.Draw(img)

		# drawing connection of each node
		for i in range(self.count1):
			draw.line( ((self.pos1[int(self.ind1[i,0]),0 ],self.pos1[int(self.ind1[i,0]),1 ]),(self.pos1[int(self.ind1[i,1]),0 ],self.pos1[int(self.ind1[i,1]),1 ])),fill=(200,200,200),width = 3   )
		for i in range(self.count2):
			draw.line( ((self.pos2[int(self.ind2[i,0]),0 ],self.pos2[int(self.ind2[i,0]),1 ]),(self.pos2[int(self.ind2[i,1]),0 ],self.pos2[int(self.ind2[i,1]),1 ])),fill=(200,200,200),width = 3   )
		for i in range(self.count3):
			draw.line( ((self.pos1[int(self.ind3[i,0]),0 ],self.pos1[int(self.ind3[i,0]),1 ]),(self.pos2[int(self.ind3[i,1]),0 ],self.pos2[int(self.ind3[i,1]),1 ])),fill=(150  ,200  ,0  ),width = 3   )
			draw.line( ((self.pos2[int(self.ind3[i,0]),0 ],self.pos2[int(self.ind3[i,0]),1 ]),(self.pos1[int(self.ind3[i,1]),0 ],self.pos1[int(self.ind3[i,1]),1 ])),fill=(150  ,200  ,0  ),width = 3   )

		# draw each node
		for i in range(self.num_node):
			draw.ellipse( ((tuple(self.pos1[i] - self.diameter_of_each_node )), ( tuple(self.pos1[i] + self.diameter_of_each_node) )), fill = (0,0,0) , outline=None )
			draw.ellipse( ((tuple(self.pos2[i] - self.diameter_of_each_node )), ( tuple(self.pos2[i] + self.diameter_of_each_node) )), fill = (0,0,0) , outline=None )

			# text
			if self.font_setting == True:
				draw.text( (self.x1_for_name[i], self.y1_for_name[i]) , str(i+1), font = self.font ,fill='#000')
				draw.text( (self.x2_for_name[i], self.y2_for_name[i]) , str(i+1), font = self.font ,fill='#000')
				draw.text( (centre_Left_net_name(self, draw).x , self.centre1[1]+self.diameter_net + self.net_name_shift) , "NET1", font = self.font ,fill='#000')
				draw.text( (centre_Right_net_name(self, draw).x, self.centre2[1]+self.diameter_net + self.net_name_shift) , "NET2", font = self.font ,fill='#000')
				draw.text( ( centre_title(self, draw , title).x , 50) , title, font = self.font1, fill='#000')
				draw.text( (self.black_legend_pos[0] + self.diameter_of_each_node*2, self.black_legend_pos[1] - self.diameter_of_each_node) , self.legend_name_black, font = self.font ,fill='#000')
				draw.text( (self.red_legend_pos[0]   + self.diameter_of_each_node*2, self.red_legend_pos[1]   - self.diameter_of_each_node) , self.legend_name_red  , font = self.font ,fill='#000')
				draw.text( (self.blue_legend_pos[0]  + self.diameter_of_each_node*2, self.blue_legend_pos[1]  - self.diameter_of_each_node) , self.legend_name_blue , font = self.font ,fill='#000')
			else:
				draw.text( (self.x1_for_name[i], self.y1_for_name[i]) , str(i+1), fill='#000')
				draw.text( (self.x2_for_name[i], self.y2_for_name[i]) , str(i+1), fill='#000')
				draw.text( (centre_Left_net_name(self, draw).x , self.centre1[1]+self.diameter_net + self.net_name_shift) , "NET1", fill='#000')
				draw.text( (centre_Right_net_name(self, draw).x, self.centre2[1]+self.diameter_net + self.net_name_shift) , "NET2", fill='#000')
				draw.text( ( centre_title(self, draw , title).x , 50) , title, fill='#000')
				draw.text( (self.black_legend_pos[0] + self.diameter_of_each_node*2, self.black_legend_pos[1] - self.diameter_of_each_node) , self.legend_name_black,fill='#000')
				draw.text( (self.red_legend_pos[0]   + self.diameter_of_each_node*2, self.red_legend_pos[1]   - self.diameter_of_each_node) , self.legend_name_red  ,fill='#000')
				draw.text( (self.blue_legend_pos[0]  + self.diameter_of_each_node*2, self.blue_legend_pos[1]  - self.diameter_of_each_node) , self.legend_name_blue ,fill='#000')

		# draw node which has hyper connection.
		# in order to differentiate colour from normal nodes
		for i in range(self.count3):
			draw.ellipse( ((tuple(self.pos1[ int(self.ind3[i,0]) ] - self.diameter_of_each_node )), ( tuple(self.pos1[ int(self.ind3[i,0]) ] + self.diameter_of_each_node) )), fill = self.red  , outline=None )
			draw.ellipse( ((tuple(self.pos1[ int(self.ind3[i,1]) ] - self.diameter_of_each_node )), ( tuple(self.pos1[ int(self.ind3[i,1]) ] + self.diameter_of_each_node) )), fill = self.blue , outline=None )
			draw.ellipse( ((tuple(self.pos2[ int(self.ind3[i,0]) ] - self.diameter_of_each_node )), ( tuple(self.pos2[ int(self.ind3[i,0]) ] + self.diameter_of_each_node) )), fill = self.red  , outline=None )
			draw.ellipse( ((tuple(self.pos2[ int(self.ind3[i,1]) ] - self.diameter_of_each_node )), ( tuple(self.pos2[ int(self.ind3[i,1]) ] + self.diameter_of_each_node) )), fill = self.blue , outline=None )

		# legend node
		draw.ellipse( ((tuple(self.black_legend_pos  - self.diameter_of_each_node )), ( tuple(self.black_legend_pos  + self.diameter_of_each_node) )), fill = self.black  , outline=None )
		draw.ellipse( ((tuple(self.red_legend_pos  - self.diameter_of_each_node )), ( tuple(self.red_legend_pos  + self.diameter_of_each_node) )), fill = self.red  , outline=None )
		draw.ellipse( ((tuple(self.blue_legend_pos - self.diameter_of_each_node )), ( tuple(self.blue_legend_pos + self.diameter_of_each_node) )), fill = self.blue , outline=None )

		os.chdir(DIRECTORY)
		img.save( title + "_.png" )

	# illustrate hyper network
	def for_papers_Connected_net_pic(self, DIRECTORY, title = "Connected_net_pic"):
		print "def Connected_net_pic(self, title = Connected_net_pic):"
		img  = Image.new("RGB", (self.yoko, self.tate+self.Tate), color=(255, 255, 255))
		draw = ImageDraw.Draw(img)

		legend_diameter_each_node = 5 * self.diameter_of_each_node


		# drawing connection of each node
		for i in range(self.count1):
			draw.line( ((self.pos1[int(self.ind1[i,0]),0 ],self.pos1[int(self.ind1[i,0]),1 ]),(self.pos1[int(self.ind1[i,1]),0 ],self.pos1[int(self.ind1[i,1]),1 ])),fill=(200,200,200),width = 3   )
		for i in range(self.count2):
			draw.line( ((self.pos2[int(self.ind2[i,0]),0 ],self.pos2[int(self.ind2[i,0]),1 ]),(self.pos2[int(self.ind2[i,1]),0 ],self.pos2[int(self.ind2[i,1]),1 ])),fill=(200,200,200),width = 3   )
		for i in range(self.count3):
			draw.line( ((self.pos1[int(self.ind3[i,0]),0 ],self.pos1[int(self.ind3[i,0]),1 ]),(self.pos2[int(self.ind3[i,1]),0 ],self.pos2[int(self.ind3[i,1]),1 ])),fill=(150  ,200  ,0  ),width = 3   )
			draw.line( ((self.pos2[int(self.ind3[i,0]),0 ],self.pos2[int(self.ind3[i,0]),1 ]),(self.pos1[int(self.ind3[i,1]),0 ],self.pos1[int(self.ind3[i,1]),1 ])),fill=(150  ,200  ,0  ),width = 3   )

		# draw each node
		for i in range(self.num_node):
			draw.ellipse( ((tuple(self.pos1[i] - self.diameter_of_each_node )), ( tuple(self.pos1[i] + self.diameter_of_each_node) )), fill = (0,0,0) , outline=None )
			draw.ellipse( ((tuple(self.pos2[i] - self.diameter_of_each_node )), ( tuple(self.pos2[i] + self.diameter_of_each_node) )), fill = (0,0,0) , outline=None )

			# text
			if self.font_setting == True:
				"""
				draw.text( (self.x1_for_name[i], self.y1_for_name[i]) , str(i+1), font = self.font ,fill='#000')
				draw.text( (self.x2_for_name[i], self.y2_for_name[i]) , str(i+1), font = self.font ,fill='#000')
				draw.text( (centre_Left_net_name(self, draw).x , self.centre1[1]+self.diameter_net + self.net_name_shift) , "NET1", font = self.font ,fill='#000')
				draw.text( (centre_Right_net_name(self, draw).x, self.centre2[1]+self.diameter_net + self.net_name_shift) , "NET2", font = self.font ,fill='#000')
				draw.text( ( centre_title(self, draw , title).x , 50) , title, font = self.font1, fill='#000')
				"""
				draw.text( (self.black_legend_pos[0] + legend_diameter_each_node*2, self.black_legend_pos[1] - legend_diameter_each_node) , self.legend_name_black, font = self.font ,fill='#000')
				draw.text( (self.red_legend_pos[0]   + legend_diameter_each_node*2, self.red_legend_pos[1]   - legend_diameter_each_node) , self.legend_name_red  , font = self.font ,fill='#000')
				draw.text( (self.blue_legend_pos[0]  + legend_diameter_each_node*2, self.blue_legend_pos[1]  - legend_diameter_each_node) , self.legend_name_blue , font = self.font ,fill='#000')
			else:
				"""
				draw.text( (self.x1_for_name[i], self.y1_for_name[i]) , str(i+1), fill='#000')
				draw.text( (self.x2_for_name[i], self.y2_for_name[i]) , str(i+1), fill='#000')
				draw.text( (centre_Left_net_name(self, draw).x , self.centre1[1]+self.diameter_net + self.net_name_shift) , "NET1", fill='#000')
				draw.text( (centre_Right_net_name(self, draw).x, self.centre2[1]+self.diameter_net + self.net_name_shift) , "NET2", fill='#000')
				draw.text( ( centre_title(self, draw , title).x , 50) , title, fill='#000')
				"""
				draw.text( (self.black_legend_pos[0] + legend_diameter_each_node*2, self.black_legend_pos[1] - legend_diameter_each_node) , self.legend_name_black,fill='#000')
				draw.text( (self.red_legend_pos[0]   + legend_diameter_each_node*2, self.red_legend_pos[1]   - legend_diameter_each_node) , self.legend_name_red  ,fill='#000')
				draw.text( (self.blue_legend_pos[0]  + legend_diameter_each_node*2, self.blue_legend_pos[1]  - legend_diameter_each_node) , self.legend_name_blue ,fill='#000')

		# draw node which has hyper connection.
		# in order to differentiate colour from normal nodes
		for i in range(self.count3):
			draw.ellipse( ((tuple(self.pos1[ int(self.ind3[i,0]) ] - self.diameter_of_each_node )), ( tuple(self.pos1[ int(self.ind3[i,0]) ] + self.diameter_of_each_node) )), fill = self.red  , outline=None )
			draw.ellipse( ((tuple(self.pos1[ int(self.ind3[i,1]) ] - self.diameter_of_each_node )), ( tuple(self.pos1[ int(self.ind3[i,1]) ] + self.diameter_of_each_node) )), fill = self.blue , outline=None )
			draw.ellipse( ((tuple(self.pos2[ int(self.ind3[i,0]) ] - self.diameter_of_each_node )), ( tuple(self.pos2[ int(self.ind3[i,0]) ] + self.diameter_of_each_node) )), fill = self.red  , outline=None )
			draw.ellipse( ((tuple(self.pos2[ int(self.ind3[i,1]) ] - self.diameter_of_each_node )), ( tuple(self.pos2[ int(self.ind3[i,1]) ] + self.diameter_of_each_node) )), fill = self.blue , outline=None )

		# legend node
		draw.ellipse( ((tuple(self.black_legend_pos  - legend_diameter_each_node )), ( tuple(self.black_legend_pos  + legend_diameter_each_node) )), fill = self.black  , outline=None )
		draw.ellipse( ((tuple(self.red_legend_pos  - legend_diameter_each_node )), ( tuple(self.red_legend_pos  + legend_diameter_each_node) )), fill = self.red  , outline=None )
		draw.ellipse( ((tuple(self.blue_legend_pos - legend_diameter_each_node )), ( tuple(self.blue_legend_pos + legend_diameter_each_node) )), fill = self.blue , outline=None )

		os.chdir(DIRECTORY)
		img.save( title + "_.png" )

	def net_feature_pic(self, net_feature1, net_feature2 , title = "Connected_net_feature_value_pic"):
		print "def net_feature_pic(self, net_feature1, net_feature2 , title = Connected_net_feature_value_pic):"
		feature1 = np.array(net_feature1)
		feature2 = np.array(net_feature2)

		if np.amax(feature1) == np.amin(feature1):
			feature1 = self.diameter_of_each_node * 2 * feature1/np.amax(feature1) + 2
		else:
			feature1 = self.diameter_of_each_node * 2 * ( feature1 - np.amin(feature1) )/( np.amax(feature1)-np.amin(feature1) ) + 2
		if np.amax(feature2) == np.amin(feature2):
			feature2 = self.diameter_of_each_node * 2 * feature2/np.amax(feature2) + 2
		else:
			feature2 = self.diameter_of_each_node * 2 * ( feature2 - np.amin(feature2) )/( np.amax(feature2)-np.amin(feature2) ) + 2

		img  = Image.new("RGB", (self.yoko, self.tate), color=(255, 255, 255))
		draw = ImageDraw.Draw(img)

		# drawing connection of each node
		for i in range(self.count1):
			draw.line( ((self.pos1[int(self.ind1[i,0]),0 ],self.pos1[int(self.ind1[i,0]),1 ]),(self.pos1[int(self.ind1[i,1]),0 ],self.pos1[int(self.ind1[i,1]),1 ])),fill=(200,200,200),width = 3   )
		for i in range(self.count2):
			draw.line( ((self.pos2[int(self.ind2[i,0]),0 ],self.pos2[int(self.ind2[i,0]),1 ]),(self.pos2[int(self.ind2[i,1]),0 ],self.pos2[int(self.ind2[i,1]),1 ])),fill=(200,200,200),width = 3   )
		for i in range(self.count3):
			draw.line( ((self.pos1[int(self.ind3[i,0]),0 ],self.pos1[int(self.ind3[i,0]),1 ]),(self.pos2[int(self.ind3[i,1]),0 ],self.pos2[int(self.ind3[i,1]),1 ])),fill=(150  ,200  ,0  ),width = 3   )
			draw.line( ((self.pos2[int(self.ind3[i,0]),0 ],self.pos2[int(self.ind3[i,0]),1 ]),(self.pos1[int(self.ind3[i,1]),0 ],self.pos1[int(self.ind3[i,1]),1 ])),fill=(150  ,200  ,0  ),width = 3   )

		# draw each node changin diamters depending on net feature value
		for i in range(self.num_node):
			draw.ellipse( ((tuple(self.pos1[i] - feature1[i] )), ( tuple(self.pos1[i] + feature1[i]) )), fill = (0,0,0) , outline=None )
			draw.ellipse( ((tuple(self.pos2[i] - feature2[i] )), ( tuple(self.pos2[i] + feature2[i]) )), fill = (0,0,0) , outline=None )

			# text
			if self.font_setting == True:
				draw.text( (self.x1_for_name[i], self.y1_for_name[i]) , str(i+1), font = self.font ,fill='#000')
				draw.text( (self.x2_for_name[i], self.y2_for_name[i]) , str(i+1), font = self.font ,fill='#000')
				draw.text( (centre_Left_net_name(self, draw).x , self.centre1[1]+self.diameter_net + self.net_name_shift) , "NET1", font = self.font ,fill='#000')
				draw.text( (centre_Right_net_name(self, draw).x, self.centre2[1]+self.diameter_net + self.net_name_shift) , "NET2", font = self.font ,fill='#000')
				draw.text( ( centre_title(self, draw , title).x , 50) , title, font = self.font1, fill='#000')
				draw.text( (self.black_legend_pos[0] + self.diameter_of_each_node*2, self.black_legend_pos[1] - self.diameter_of_each_node) , self.legend_name_black, font = self.font ,fill='#000')
				draw.text( (self.red_legend_pos[0]   + self.diameter_of_each_node*2, self.red_legend_pos[1]   - self.diameter_of_each_node) , self.legend_name_red  , font = self.font ,fill='#000')
				draw.text( (self.blue_legend_pos[0]  + self.diameter_of_each_node*2, self.blue_legend_pos[1]  - self.diameter_of_each_node) , self.legend_name_blue , font = self.font ,fill='#000')
			else:
				draw.text( (self.x1_for_name[i], self.y1_for_name[i]) , str(i+1), fill='#000')
				draw.text( (self.x2_for_name[i], self.y2_for_name[i]) , str(i+1), fill='#000')
				draw.text( (centre_Left_net_name(self, draw).x , self.centre1[1]+self.diameter_net + self.net_name_shift) , "NET1", fill='#000')
				draw.text( (centre_Right_net_name(self, draw).x, self.centre2[1]+self.diameter_net + self.net_name_shift) , "NET2", fill='#000')
				draw.text( ( centre_title(self, draw , title).x , 50) , title, fill='#000')
				draw.text( (self.black_legend_pos[0] + self.diameter_of_each_node*2, self.black_legend_pos[1] - self.diameter_of_each_node) , self.legend_name_black,fill='#000')
				draw.text( (self.red_legend_pos[0]   + self.diameter_of_each_node*2, self.red_legend_pos[1]   - self.diameter_of_each_node) , self.legend_name_red  ,fill='#000')
				draw.text( (self.blue_legend_pos[0]  + self.diameter_of_each_node*2, self.blue_legend_pos[1]  - self.diameter_of_each_node) , self.legend_name_blue ,fill='#000')

		# draw node which has hyper connection.
		# in order to differentiate colour from normal nodes
		for i in range(self.count3):
			draw.ellipse( ((tuple(self.pos1[ int(self.ind3[i,0]) ] - feature1[int(self.ind3[i,0])] )), ( tuple(self.pos1[ int(self.ind3[i,0]) ] + feature1[int(self.ind3[i,0])]) )), fill = self.red  , outline=None )
			draw.ellipse( ((tuple(self.pos1[ int(self.ind3[i,1]) ] - feature1[int(self.ind3[i,1])] )), ( tuple(self.pos1[ int(self.ind3[i,1]) ] + feature1[int(self.ind3[i,1])]) )), fill = self.blue , outline=None )
			draw.ellipse( ((tuple(self.pos2[ int(self.ind3[i,0]) ] - feature2[int(self.ind3[i,0])] )), ( tuple(self.pos2[ int(self.ind3[i,0]) ] + feature2[int(self.ind3[i,0])]) )), fill = self.red  , outline=None )
			draw.ellipse( ((tuple(self.pos2[ int(self.ind3[i,1]) ] - feature2[int(self.ind3[i,1])] )), ( tuple(self.pos2[ int(self.ind3[i,1]) ] + feature2[int(self.ind3[i,1])]) )), fill = self.blue , outline=None )

		# legend node
		draw.ellipse( ((tuple(self.black_legend_pos  - self.diameter_of_each_node )), ( tuple(self.black_legend_pos  + self.diameter_of_each_node) )), fill = self.black  , outline=None )
		draw.ellipse( ((tuple(self.red_legend_pos  - self.diameter_of_each_node )), ( tuple(self.red_legend_pos  + self.diameter_of_each_node) )), fill = self.red  , outline=None )
		draw.ellipse( ((tuple(self.blue_legend_pos - self.diameter_of_each_node )), ( tuple(self.blue_legend_pos + self.diameter_of_each_node) )), fill = self.blue , outline=None )

		img.save( title + "_.png" )

	# making gif for phase changes in each node
	def Connected_net_phase_gif(self, title = "Connected_net_phase_gif"):
		print "def Connected_net_phase_gif(self, title = Connected_net_phase_gif):"
		data = self.diameter_of_each_node*np.sin(self.phi) + self.diameter_of_each_node
		imgs = []
		loop = self.phi.shape[1]

		for t in range(loop):
			img  = Image.new("RGB", (self.yoko, self.tate), color=(255, 255, 255))
			draw = ImageDraw.Draw(img)

			# drawing connection
			for i in range(self.count1):
				draw.line( ((self.pos1[int(self.ind1[i,0]),0 ],self.pos1[int(self.ind1[i,0]),1 ]),(self.pos1[int(self.ind1[i,1]),0 ],self.pos1[int(self.ind1[i,1]),1 ])),fill=(200,200,200),width = 3   )
			for i in range(self.count2):
				draw.line( ((self.pos2[int(self.ind2[i,0]),0 ],self.pos2[int(self.ind2[i,0]),1 ]),(self.pos2[int(self.ind2[i,1]),0 ],self.pos2[int(self.ind2[i,1]),1 ])),fill=(200,200,200),width = 3   )
			for i in range(self.count3):
				draw.line( ((self.pos1[int(self.ind3[i,0]),0 ],self.pos1[int(self.ind3[i,0]),1 ]),(self.pos2[int(self.ind3[i,1]),0 ],self.pos2[int(self.ind3[i,1]),1 ])),fill=(150  ,200  ,0  ),width = 3   )
				draw.line( ((self.pos2[int(self.ind3[i,0]),0 ],self.pos2[int(self.ind3[i,0]),1 ]),(self.pos1[int(self.ind3[i,1]),0 ],self.pos1[int(self.ind3[i,1]),1 ])),fill=(150  ,200  ,0  ),width = 3   )

			# drawing node
			for i in range(self.num_node):
				draw.ellipse( ((tuple(self.pos1[i] - data[i,t] ))              , ( tuple(self.pos1[i] + data[i,t] ) ))              , fill = (0,0,0) , outline=None )
				draw.ellipse( ((tuple(self.pos2[i] - data[i+self.num_node,t] )), ( tuple(self.pos2[i] + data[i+self.num_node,t] ) )), fill = (0,0,0) , outline=None )

				# text
				if self.font_setting == True:
					draw.text( (self.x1_for_name[i], self.y1_for_name[i]) , str(i+1), font = self.font ,fill='#000')
					draw.text( (self.x2_for_name[i], self.y2_for_name[i]) , str(i+1), font = self.font ,fill='#000')
					draw.text( (centre_Left_net_name(self, draw).x , self.centre1[1]+self.diameter_net + self.net_name_shift) , "NET1", font = self.font ,fill='#000')
					draw.text( (centre_Right_net_name(self, draw).x, self.centre2[1]+self.diameter_net + self.net_name_shift) , "NET2", font = self.font ,fill='#000')
				else:
					draw.text( (self.x1_for_name[i], self.y1_for_name[i]) , str(i+1), fill='#000')
					draw.text( (self.x2_for_name[i], self.y2_for_name[i]) , str(i+1), fill='#000')
					draw.text( (centre_Left_net_name(self, draw).x , self.centre1[1]+self.diameter_net + self.net_name_shift) , "NET1", fill='#000')
					draw.text( (centre_Right_net_name(self, draw).x, self.centre2[1]+self.diameter_net + self.net_name_shift) , "NET2", fill='#000')

			# drawing node whose are hyper connected between 2 networks
			for i in range(self.count3):
				draw.ellipse( ((tuple(self.pos1[ int(self.ind3[i,0]) ] - data[int(self.ind3[i,0]),t] )), ( tuple(self.pos1[ int(self.ind3[i,0]) ] + data[int(self.ind3[i,0]),t]) )), fill = self.red  , outline=None )
				draw.ellipse( ((tuple(self.pos1[ int(self.ind3[i,1]) ] - data[int(self.ind3[i,1]),t] )), ( tuple(self.pos1[ int(self.ind3[i,1]) ] + data[int(self.ind3[i,1]),t]) )), fill = self.blue , outline=None )
				draw.ellipse( ((tuple(self.pos2[ int(self.ind3[i,0]) ] - data[int(self.ind3[i,0])+self.num_node,t] )), ( tuple(self.pos2[ int(self.ind3[i,0]) ] + data[int(self.ind3[i,0])+self.num_node,t]) )), fill = self.red  , outline=None )
				draw.ellipse( ((tuple(self.pos2[ int(self.ind3[i,1]) ] - data[int(self.ind3[i,1])+self.num_node,t] )), ( tuple(self.pos2[ int(self.ind3[i,1]) ] + data[int(self.ind3[i,1])+self.num_node,t]) )), fill = self.blue , outline=None )

			# drawing node legend
			draw.ellipse( ((tuple(self.black_legend_pos  - self.diameter_of_each_node )), ( tuple(self.black_legend_pos  + self.diameter_of_each_node) )), fill = self.black  , outline=None )
			draw.ellipse( ((tuple(self.red_legend_pos    - self.diameter_of_each_node )), ( tuple(self.red_legend_pos    + self.diameter_of_each_node) )), fill = self.red  , outline=None )
			draw.ellipse( ((tuple(self.blue_legend_pos   - self.diameter_of_each_node )), ( tuple(self.blue_legend_pos   + self.diameter_of_each_node) )), fill = self.blue , outline=None )

			# legend name & title name
			Title = title + "_" + str(t+1) + "(msec)"
			if self.font_setting == True:
				draw.text( ( centre_title(self, draw , Title).x , 50) , Title , font = self.font1, fill='#000')
				draw.text( (self.black_legend_pos[0] + self.diameter_of_each_node*2, self.black_legend_pos[1] - self.diameter_of_each_node) , self.legend_name_black, font = self.font ,fill='#000')
				draw.text( (self.red_legend_pos[0]   + self.diameter_of_each_node*2, self.red_legend_pos[1]   - self.diameter_of_each_node) , self.legend_name_red  , font = self.font ,fill='#000')
				draw.text( (self.blue_legend_pos[0]  + self.diameter_of_each_node*2, self.blue_legend_pos[1]  - self.diameter_of_each_node) , self.legend_name_blue , font = self.font ,fill='#000')
			else:
				draw.text( ( centre_title(self, draw , Title).x , 50) , Title , fill='#000')
				draw.text( (self.black_legend_pos[0] + self.diameter_of_each_node*2, self.black_legend_pos[1] - self.diameter_of_each_node) , self.legend_name_black,fill='#000')
				draw.text( (self.red_legend_pos[0]   + self.diameter_of_each_node*2, self.red_legend_pos[1]   - self.diameter_of_each_node) , self.legend_name_red  ,fill='#000')
				draw.text( (self.blue_legend_pos[0]  + self.diameter_of_each_node*2, self.blue_legend_pos[1]  - self.diameter_of_each_node) , self.legend_name_blue ,fill='#000')

			imgs.append(img)
		imgs[0].save( title +  "_.gif", save_all=True, append_images = imgs[1:], optimize=False, loop=0 )

	# showing each node changes and painting graph cahnges
	def Connected_net_phase_with_params_gif(self, y , ylabel = "params", graph_title = "params_time" , title = "Connected_net_phase_gif"):
		print "def Connected_net_phase_with_params_gif(self, y , ylabel = params, graph_title = params_time , title = Connected_net_phase_gif):"
		TATE      = self.tate + self.Tate
		data      = self.diameter_of_each_node*np.sin(self.phi) + self.diameter_of_each_node
		imgs      = []
		loop      = self.phi.shape[1]
		taxis     = np.arange(loop)

		# setting each axis
		Ylim            = self.Tate*7 / 10
		zero_point      = np.array( [self.centre1[0]-self.diameter_net, TATE -50 ] )
		y_axis_pos      = np.array( [self.centre1[0]-self.diameter_net, TATE -50 - Ylim ] )
		x_axis_pos      = np.array( [self.centre2[0]+self.diameter_net, TATE -50 ] )
		graph_title_pos = np.array( [self.yoko/2, self.tate ] )
		Xlim            = x_axis_pos[0] - zero_point[0]
		step            = float(Xlim) / loop
		xlabel_pos      = np.array( [(x_axis_pos[0] + zero_point[0])/2 - 50, (x_axis_pos[1] + zero_point[1])/2])
		ylabel_pos      = np.array( [(y_axis_pos[0] + zero_point[0])/2 - 70, (y_axis_pos[1] + zero_point[1])/2])

		if y.ndim == 1:
			temp    = np.zeros(1)
			taxis   = np.concatenate( ( temp, taxis ) )
			temp[0] = y[0]
			y       = np.concatenate( ( temp, y ) )

			# setting position of each data
			X = zero_point[0] + taxis * step
			Y = zero_point[1] - y * Ylim

			pos = np.array( [X[0], Y[0]] )

			for t in range(loop):
				img  = Image.new("RGB", (self.yoko, TATE), color=(255, 255, 255))
				draw = ImageDraw.Draw(img)

				# drawing axis for graph
				draw.line( (( zero_point[0], zero_point[1] ),( y_axis_pos[0], y_axis_pos[1] )),fill=(50,50,50),width = 3   ) # graph axis line
				draw.line( (( zero_point[0], zero_point[1] ),( x_axis_pos[0], x_axis_pos[1] )),fill=(50,50,50),width = 3   ) # graph axis line
				draw.line( (( y_axis_pos[0], y_axis_pos[1] ),( x_axis_pos[0], y_axis_pos[1] )),fill=(50,50,50),width = 1   ) # graph axis line
				draw.line( (( zero_point[0], (zero_point[1]+y_axis_pos[1])/2   ),( x_axis_pos[0], (zero_point[1]+y_axis_pos[1])/2    )),fill=(200,200,200),width = 2   ) # y = 1/2 line
				draw.line( (( zero_point[0], (zero_point[1]+y_axis_pos[1])/2 - Ylim/4  ),( x_axis_pos[0], (zero_point[1]+y_axis_pos[1])/2 - Ylim/4 )),fill=(220,220,220),width = 1   ) # y = 3/4 line
				draw.line( (( zero_point[0], (zero_point[1]+y_axis_pos[1])/2 + Ylim/4  ),( x_axis_pos[0], (zero_point[1]+y_axis_pos[1])/2 + Ylim/4 )),fill=(220,220,220),width = 1   ) # y = 1/4 line

				# changes of order parameter
				pos = np.concatenate( (pos, np.array( [X[t+1], Y[t+1]] ) ) )
				draw.line( tuple( pos.tolist() ) ,fill=(0,0,0),width = 3   )

				# draw connection
				for i in range(self.count1):
					draw.line( ((self.pos1[int(self.ind1[i,0]),0 ],self.pos1[int(self.ind1[i,0]),1 ]),(self.pos1[int(self.ind1[i,1]),0 ],self.pos1[int(self.ind1[i,1]),1 ])),fill=(200,200,200),width = 3   )
				for i in range(self.count2):
					draw.line( ((self.pos2[int(self.ind2[i,0]),0 ],self.pos2[int(self.ind2[i,0]),1 ]),(self.pos2[int(self.ind2[i,1]),0 ],self.pos2[int(self.ind2[i,1]),1 ])),fill=(200,200,200),width = 3   )
				for i in range(self.count3):
					draw.line( ((self.pos1[int(self.ind3[i,0]),0 ],self.pos1[int(self.ind3[i,0]),1 ]),(self.pos2[int(self.ind3[i,1]),0 ],self.pos2[int(self.ind3[i,1]),1 ])),fill=(150  ,200  ,0  ),width = 3   )
					draw.line( ((self.pos2[int(self.ind3[i,0]),0 ],self.pos2[int(self.ind3[i,0]),1 ]),(self.pos1[int(self.ind3[i,1]),0 ],self.pos1[int(self.ind3[i,1]),1 ])),fill=(150  ,200  ,0  ),width = 3   )

				# normal nodes changing diamter depending on phi
				for i in range(self.num_node):
					draw.ellipse( ((tuple(self.pos1[i] - data[i,t] ))         , ( tuple(self.pos1[i] + data[i,t] ) )), fill = (0,0,0) , outline=None )
					draw.ellipse( ((tuple(self.pos2[i] - data[i+self.num_node,t] )), ( tuple(self.pos2[i] + data[i+self.num_node,t] ) )), fill = (0,0,0) , outline=None )

					# name of each node
					if self.font_setting == True:
						draw.text( (self.x1_for_name[i], self.y1_for_name[i]) , str(i+1), font = self.font ,fill='#000')
						draw.text( (self.x2_for_name[i], self.y2_for_name[i]) , str(i+1), font = self.font ,fill='#000')
						draw.text( (centre_Left_net_name(self, draw).x , self.centre1[1]+self.diameter_net + self.net_name_shift) , "NET1", font = self.font ,fill='#000')
						draw.text( (centre_Right_net_name(self, draw).x, self.centre2[1]+self.diameter_net + self.net_name_shift) , "NET2", font = self.font ,fill='#000')
					else:
						draw.text( (self.x1_for_name[i], self.y1_for_name[i]) , str(i+1), fill='#000')
						draw.text( (self.x2_for_name[i], self.y2_for_name[i]) , str(i+1), fill='#000')
						draw.text( (centre_Left_net_name(self, draw).x , self.centre1[1]+self.diameter_net + self.net_name_shift) , "NET1", fill='#000')
						draw.text( (centre_Right_net_name(self, draw).x, self.centre2[1]+self.diameter_net + self.net_name_shift) , "NET2", fill='#000')

				# drawing sensor and motor nodes changing diameter
				for i in range(self.count3):
					draw.ellipse( ((tuple(self.pos1[ int(self.ind3[i,0]) ] - data[int(self.ind3[i,0]),t] )), ( tuple(self.pos1[ int(self.ind3[i,0]) ] + data[int(self.ind3[i,0]),t]) )), fill = self.red  , outline=None )
					draw.ellipse( ((tuple(self.pos1[ int(self.ind3[i,1]) ] - data[int(self.ind3[i,1]),t] )), ( tuple(self.pos1[ int(self.ind3[i,1]) ] + data[int(self.ind3[i,1]),t]) )), fill = self.blue , outline=None )
					draw.ellipse( ((tuple(self.pos2[ int(self.ind3[i,0]) ] - data[int(self.ind3[i,0])+self.num_node,t] )), ( tuple(self.pos2[ int(self.ind3[i,0]) ] + data[int(self.ind3[i,0])+self.num_node,t]) )), fill = self.red  , outline=None )
					draw.ellipse( ((tuple(self.pos2[ int(self.ind3[i,1]) ] - data[int(self.ind3[i,1])+self.num_node,t] )), ( tuple(self.pos2[ int(self.ind3[i,1]) ] + data[int(self.ind3[i,1])+self.num_node,t]) )), fill = self.blue , outline=None )

				# legend node
				draw.ellipse( ((tuple(self.black_legend_pos  - self.diameter_of_each_node )), ( tuple(self.black_legend_pos  + self.diameter_of_each_node) )), fill = self.black  , outline=None )
				draw.ellipse( ((tuple(self.red_legend_pos  - self.diameter_of_each_node )), ( tuple(self.red_legend_pos  + self.diameter_of_each_node) )), fill = self.red  , outline=None )
				draw.ellipse( ((tuple(self.blue_legend_pos - self.diameter_of_each_node )), ( tuple(self.blue_legend_pos + self.diameter_of_each_node) )), fill = self.blue , outline=None )

				# title, leged, graph
				Title = title + "_" + str(t+1) + "(msec)"
				if self.font_setting == True:
					draw.text( ( centre_title(self, draw , Title).x , 50) , Title , font = self.font1, fill='#000')
					draw.text( (self.black_legend_pos[0] + self.diameter_of_each_node*2, self.black_legend_pos[1] - self.diameter_of_each_node) , self.legend_name_black, font = self.font ,fill='#000')
					draw.text( (self.red_legend_pos[0]   + self.diameter_of_each_node*2, self.red_legend_pos[1]   - self.diameter_of_each_node) , self.legend_name_red  , font = self.font ,fill='#000')
					draw.text( (self.blue_legend_pos[0]  + self.diameter_of_each_node*2, self.blue_legend_pos[1]  - self.diameter_of_each_node) , self.legend_name_blue , font = self.font ,fill='#000')
					draw.text( (ylabel_pos[0], ylabel_pos[1])  , ylabel            , font = self.font ,fill='#000')
					draw.text( (y_axis_pos[0], y_axis_pos[1])  , "1.0"             , font = self.font ,fill='#000')
					draw.text( (y_axis_pos[0], ylabel_pos[1])  , "0.5"             , font = self.font ,fill='#000')
					draw.text( (zero_point[0], zero_point[1])  , "0.0"             , font = self.font ,fill='#000')
					draw.text( (x_axis_pos[0], x_axis_pos[1])  , str(loop)+" msec" , font = self.font ,fill='#000')
					draw.text( (centre_title(self, draw , "time " + str(t+1) + " msec").x, xlabel_pos[1]       )  , "time " + str(t+1) + " msec" , font = self.font1 ,fill='#000')
					draw.text( (centre_title(self, draw , graph_title).x, graph_title_pos[1]-5)  , graph_title , font = self.font1 ,fill='#000')
				else:
					draw.text( ( centre_title(self, draw , Title).x , 50) , Title , fill='#000')
					draw.text( (self.black_legend_pos[0] + self.diameter_of_each_node*2, self.black_legend_pos[1] - self.diameter_of_each_node) , self.legend_name_black,fill='#000')
					draw.text( (self.red_legend_pos[0]   + self.diameter_of_each_node*2, self.red_legend_pos[1]   - self.diameter_of_each_node) , self.legend_name_red  ,fill='#000')
					draw.text( (self.blue_legend_pos[0]  + self.diameter_of_each_node*2, self.blue_legend_pos[1]  - self.diameter_of_each_node) , self.legend_name_blue ,fill='#000')
					draw.text( (ylabel_pos[0], ylabel_pos[1])  , ylabel            ,fill='#000')
					draw.text( (y_axis_pos[0], y_axis_pos[1])  , "1.0"             ,fill='#000')
					draw.text( (y_axis_pos[0], ylabel_pos[1])  , "0.5"             ,fill='#000')
					draw.text( (zero_point[0], zero_point[1])  , "0.0"             ,fill='#000')
					draw.text( (x_axis_pos[0], x_axis_pos[1])  , str(loop)+" msec" ,fill='#000')
					draw.text( (centre_title(self, draw , "time " + str(t+1) + " msec").x, xlabel_pos[1]       )  , "time " + str(t+1) + " msec" ,fill='#000')
					draw.text( (centre_title(self, draw , graph_title).x, graph_title_pos[1]-5)  , graph_title ,fill='#000')

				imgs.append(img)
			imgs[0].save( title +  "_.gif", save_all=True, append_images = imgs[1:], optimize=False, loop=0 )
		else: # for PLV
			num_slice = 10
			temp      = np.zeros(1)
			taxis     = np.concatenate( ( temp, taxis ) )
			TEMP      = np.empty(y.shape[0]).reshape(y.shape[0],1)
			TEMP[:,0] = y[:,0]
			TP        = np.empty(y.shape[0]*loop).reshape(y.shape[0],loop)
			for i,j in itertools.product( range(num_slice), range(loop/num_slice) ):
				TP[:,i*loop/num_slice+j] = y[:,i]
			y         = np.concatenate( ( TEMP, TP ) ,axis=1)
			arr       = y.shape[0]

			# setting colour of each data
			C = np.empty(arr*3).reshape(arr,3)
			for i in range(arr):
				colour = 200 * np.random.random_sample(3)
				colour = colour.astype(np.int32)
				C[i,:] = colour

			# setting position of each data
			X = zero_point[0] + taxis * step
			Y = zero_point[1] - y * Ylim

			pos = [ [0] * 2 for i in range(arr)  ] # 2-D list
			for i in range(arr):
				pos[i][:] = [X[0],Y[i,0]]

			for t in range(loop):
				img  = Image.new("RGB", (self.yoko, TATE), color=(255, 255, 255))
				draw = ImageDraw.Draw(img)

				# graph
				for i in range(arr):
					pos[i] = np.concatenate( (pos[i], [X[t+1],Y[i,t+1]]) )
					draw.line( tuple(pos[i])  ,fill=tuple(C[i].astype(np.int32)),width = 3   )

				# draw axis
				draw.line( (( zero_point[0], zero_point[1] ),( y_axis_pos[0], y_axis_pos[1] )),fill=(50,50,50),width = 3   ) # graph axis line
				draw.line( (( zero_point[0], zero_point[1] ),( x_axis_pos[0], x_axis_pos[1] )),fill=(50,50,50),width = 3   ) # graph axis line
				draw.line( (( y_axis_pos[0], y_axis_pos[1] ),( x_axis_pos[0], y_axis_pos[1] )),fill=(50,50,50),width = 1   ) # graph axis line
				draw.line( (( zero_point[0], (zero_point[1]+y_axis_pos[1])/2   ),( x_axis_pos[0], (zero_point[1]+y_axis_pos[1])/2    )),fill=(200,200,200),width = 2   ) # y = 1/2 line
				draw.line( (( zero_point[0], (zero_point[1]+y_axis_pos[1])/2 - Ylim/4  ),( x_axis_pos[0], (zero_point[1]+y_axis_pos[1])/2 - Ylim/4 )),fill=(220,220,220),width = 1   ) # y = 3/4 line
				draw.line( (( zero_point[0], (zero_point[1]+y_axis_pos[1])/2 + Ylim/4  ),( x_axis_pos[0], (zero_point[1]+y_axis_pos[1])/2 + Ylim/4 )),fill=(220,220,220),width = 1   ) # y = 1/4 line

				# drawing connection
				for i in range(self.count1):
					draw.line( ((self.pos1[int(self.ind1[i,0]),0 ],self.pos1[int(self.ind1[i,0]),1 ]),(self.pos1[int(self.ind1[i,1]),0 ],self.pos1[int(self.ind1[i,1]),1 ])),fill=(200,200,200),width = 3   )
				for i in range(self.count2):
					draw.line( ((self.pos2[int(self.ind2[i,0]),0 ],self.pos2[int(self.ind2[i,0]),1 ]),(self.pos2[int(self.ind2[i,1]),0 ],self.pos2[int(self.ind2[i,1]),1 ])),fill=(200,200,200),width = 3   )
				for i in range(self.count3):
					draw.line( ((self.pos1[int(self.ind3[i,0]),0 ],self.pos1[int(self.ind3[i,0]),1 ]),(self.pos2[int(self.ind3[i,1]),0 ],self.pos2[int(self.ind3[i,1]),1 ])),fill=(150  ,200  ,0  ),width = 3   )
					draw.line( ((self.pos2[int(self.ind3[i,0]),0 ],self.pos2[int(self.ind3[i,0]),1 ]),(self.pos1[int(self.ind3[i,1]),0 ],self.pos1[int(self.ind3[i,1]),1 ])),fill=(150  ,200  ,0  ),width = 3   )

				# each node changing diamter
				for i in range(self.num_node):
					draw.ellipse( ((tuple(self.pos1[i] - data[i,t] ))         , ( tuple(self.pos1[i] + data[i,t] ) )), fill = (0,0,0) , outline=None )
					draw.ellipse( ((tuple(self.pos2[i] - data[i+self.num_node,t] )), ( tuple(self.pos2[i] + data[i+self.num_node,t] ) )), fill = (0,0,0) , outline=None )

					# each node name
					if self.font_setting == True:
						draw.text( (self.x1_for_name[i], self.y1_for_name[i]) , str(i+1), font = self.font ,fill='#000')
						draw.text( (self.x2_for_name[i], self.y2_for_name[i]) , str(i+1), font = self.font ,fill='#000')
						draw.text( (centre_Left_net_name(self, draw).x , self.centre1[1]+self.diameter_net + self.net_name_shift) , "NET1", font = self.font ,fill='#000')
						draw.text( (centre_Right_net_name(self, draw).x, self.centre2[1]+self.diameter_net + self.net_name_shift) , "NET2", font = self.font ,fill='#000')
					else:
						draw.text( (self.x1_for_name[i], self.y1_for_name[i]) , str(i+1), fill='#000')
						draw.text( (self.x2_for_name[i], self.y2_for_name[i]) , str(i+1), fill='#000')
						draw.text( (centre_Left_net_name(self, draw).x , self.centre1[1]+self.diameter_net + self.net_name_shift) , "NET1", fill='#000')
						draw.text( (centre_Right_net_name(self, draw).x, self.centre2[1]+self.diameter_net + self.net_name_shift) , "NET2", fill='#000')

				# sensor and motor nodes changing diameters
				for i in range(self.count3):
					draw.ellipse( ((tuple(self.pos1[ int(self.ind3[i,0]) ] - data[int(self.ind3[i,0]),t] )), ( tuple(self.pos1[ int(self.ind3[i,0]) ] + data[int(self.ind3[i,0]),t]) )), fill = self.red  , outline=None )
					draw.ellipse( ((tuple(self.pos1[ int(self.ind3[i,1]) ] - data[int(self.ind3[i,1]),t] )), ( tuple(self.pos1[ int(self.ind3[i,1]) ] + data[int(self.ind3[i,1]),t]) )), fill = self.blue , outline=None )
					draw.ellipse( ((tuple(self.pos2[ int(self.ind3[i,0]) ] - data[int(self.ind3[i,0])+self.num_node,t] )), ( tuple(self.pos2[ int(self.ind3[i,0]) ] + data[int(self.ind3[i,0])+self.num_node,t]) )), fill = self.red  , outline=None )
					draw.ellipse( ((tuple(self.pos2[ int(self.ind3[i,1]) ] - data[int(self.ind3[i,1])+self.num_node,t] )), ( tuple(self.pos2[ int(self.ind3[i,1]) ] + data[int(self.ind3[i,1])+self.num_node,t]) )), fill = self.blue , outline=None )

				# legend nodes
				draw.ellipse( ((tuple(self.black_legend_pos  - self.diameter_of_each_node )), ( tuple(self.black_legend_pos  + self.diameter_of_each_node) )), fill = self.black  , outline=None )
				draw.ellipse( ((tuple(self.red_legend_pos  - self.diameter_of_each_node )), ( tuple(self.red_legend_pos  + self.diameter_of_each_node) )), fill = self.red  , outline=None )
				draw.ellipse( ((tuple(self.blue_legend_pos - self.diameter_of_each_node )), ( tuple(self.blue_legend_pos + self.diameter_of_each_node) )), fill = self.blue , outline=None )

				# title, legend & axis name
				Title = title + "_" + str(t+1) + "(msec)"
				if self.font_setting == True:
					draw.text( ( centre_title(self, draw , Title).x , 50) , Title , font = self.font1, fill='#000')
					draw.text( (self.black_legend_pos[0] + self.diameter_of_each_node*2, self.black_legend_pos[1] - self.diameter_of_each_node) , self.legend_name_black, font = self.font ,fill='#000')
					draw.text( (self.red_legend_pos[0]   + self.diameter_of_each_node*2, self.red_legend_pos[1]   - self.diameter_of_each_node) , self.legend_name_red  , font = self.font ,fill='#000')
					draw.text( (self.blue_legend_pos[0]  + self.diameter_of_each_node*2, self.blue_legend_pos[1]  - self.diameter_of_each_node) , self.legend_name_blue , font = self.font ,fill='#000')
					draw.text( (ylabel_pos[0], ylabel_pos[1])  , ylabel            , font = self.font ,fill='#000')
					draw.text( (y_axis_pos[0], y_axis_pos[1])  , "1.0"             , font = self.font ,fill='#000')
					draw.text( (y_axis_pos[0], ylabel_pos[1])  , "0.5"             , font = self.font ,fill='#000')
					draw.text( (zero_point[0], zero_point[1])  , "0.0"             , font = self.font ,fill='#000')
					draw.text( (x_axis_pos[0], x_axis_pos[1])  , str(loop)+" msec" , font = self.font ,fill='#000')
					draw.text( (centre_title(self, draw , "time " + str(t+1) + " msec").x, xlabel_pos[1]       )  , "time " + str(t+1) + " msec" , font = self.font1 ,fill='#000')
					draw.text( (centre_title(self, draw , graph_title).x, graph_title_pos[1]-5)  , graph_title , font = self.font1 ,fill='#000')
				else:
					draw.text( ( centre_title(self, draw , Title).x , 50) , Title , fill='#000')
					draw.text( (self.black_legend_pos[0] + self.diameter_of_each_node*2, self.black_legend_pos[1] - self.diameter_of_each_node) , self.legend_name_black,fill='#000')
					draw.text( (self.red_legend_pos[0]   + self.diameter_of_each_node*2, self.red_legend_pos[1]   - self.diameter_of_each_node) , self.legend_name_red  ,fill='#000')
					draw.text( (self.blue_legend_pos[0]  + self.diameter_of_each_node*2, self.blue_legend_pos[1]  - self.diameter_of_each_node) , self.legend_name_blue ,fill='#000')
					draw.text( (ylabel_pos[0], ylabel_pos[1])  , ylabel            ,fill='#000')
					draw.text( (y_axis_pos[0], y_axis_pos[1])  , "1.0"             ,fill='#000')
					draw.text( (y_axis_pos[0], ylabel_pos[1])  , "0.5"             ,fill='#000')
					draw.text( (zero_point[0], zero_point[1])  , "0.0"             ,fill='#000')
					draw.text( (x_axis_pos[0], x_axis_pos[1])  , str(loop)+" msec" ,fill='#000')
					draw.text( (centre_title(self, draw , "time " + str(t+1) + " msec").x, xlabel_pos[1]       )  , "time " + str(t+1) + " msec" ,fill='#000')
					draw.text( (centre_title(self, draw , graph_title).x, graph_title_pos[1]-5)  , graph_title ,fill='#000')

				imgs.append(img)
			imgs[0].save( title +  "_.gif", save_all=True, append_images = imgs[1:], optimize=False, loop=0 )

	# drawing each node changing and PLV time development
	def Connected_net_phase_with_PLV(self, y , ylabel = "PLV", graph_title = "PLV_time_development" , title = "PLV_time_development_gif"):
		TATE      = self.tate + self.Tate
		DATA      = self.diameter_of_each_node*np.sin(self.phi) + self.diameter_of_each_node
		data      = DATA[:,::5] # each 5 msec data
		imgs      = []
		loop      = self.phi.shape[1]/5
		taxis     = np.arange(loop)

		# setting each axis
		Ylim            = self.Tate*7 / 10
		zero_point      = np.array( [self.centre1[0]-self.diameter_net, TATE -50 ] )
		y_axis_pos      = np.array( [self.centre1[0]-self.diameter_net, TATE -50 - Ylim ] )
		x_axis_pos      = np.array( [self.centre2[0]+self.diameter_net, TATE -50 ] )
		graph_title_pos = np.array( [self.yoko/2, self.tate ] )
		Xlim            = x_axis_pos[0] - zero_point[0]
		step            = float(Xlim) / loop
		xlabel_pos      = np.array( [(x_axis_pos[0] + zero_point[0])/2 - 50, (x_axis_pos[1] + zero_point[1])/2])
		ylabel_pos      = np.array( [(y_axis_pos[0] + zero_point[0])/2 - 70, (y_axis_pos[1] + zero_point[1])/2])

		num_slice = 10
		temp      = np.zeros(1)
		taxis     = np.concatenate( ( temp, taxis ) )
		TEMP      = np.empty(y.shape[0]).reshape(y.shape[0],1)
		TEMP[:,0] = y[:,0]
		TP        = np.empty(y.shape[0]*loop).reshape(y.shape[0],loop)
		for i,j in itertools.product( range(num_slice), range(loop/num_slice) ):
			TP[:,i*loop/num_slice+j] = y[:,i]
		y         = np.concatenate( ( TEMP, TP ) ,axis=1)
		arr       = y.shape[0]

		# setting colour of each data
		C = np.empty(arr*3).reshape(arr,3)
		for i in range(arr):
			colour = 200 * np.random.random_sample(3)
			colour = colour.astype(np.int32)
			C[i,:] = colour

		# setting position of each data
		X = zero_point[0] + taxis * step
		Y = zero_point[1] - y * Ylim

		pos = [ [0] * 2 for i in range(arr)  ] # 2-D list
		for i in range(arr):
			pos[i][:] = [X[0],Y[i,0]]

		for t in range(loop):
			img  = Image.new("RGB", (self.yoko, TATE), color=(255, 255, 255))
			draw = ImageDraw.Draw(img)

			# graph
			for i in range(arr):
				pos[i] = np.concatenate( (pos[i], [X[t+1],Y[i,t+1]]) )
				draw.line( tuple(pos[i])  ,fill=tuple(C[i].astype(np.int32)),width = 3   )

			# draw axis
			draw.line( (( zero_point[0], zero_point[1] ),( y_axis_pos[0], y_axis_pos[1] )),fill=(50,50,50),width = 3   ) # graph axis line
			draw.line( (( zero_point[0], zero_point[1] ),( x_axis_pos[0], x_axis_pos[1] )),fill=(50,50,50),width = 3   ) # graph axis line
			draw.line( (( y_axis_pos[0], y_axis_pos[1] ),( x_axis_pos[0], y_axis_pos[1] )),fill=(50,50,50),width = 1   ) # graph axis line
			draw.line( (( zero_point[0], (zero_point[1]+y_axis_pos[1])/2   ),( x_axis_pos[0], (zero_point[1]+y_axis_pos[1])/2    )),fill=(200,200,200),width = 2   ) # y = 1/2 line
			draw.line( (( zero_point[0], (zero_point[1]+y_axis_pos[1])/2 - Ylim/4  ),( x_axis_pos[0], (zero_point[1]+y_axis_pos[1])/2 - Ylim/4 )),fill=(220,220,220),width = 1   ) # y = 3/4 line
			draw.line( (( zero_point[0], (zero_point[1]+y_axis_pos[1])/2 + Ylim/4  ),( x_axis_pos[0], (zero_point[1]+y_axis_pos[1])/2 + Ylim/4 )),fill=(220,220,220),width = 1   ) # y = 1/4 line

			# drawing connection
			for i in range(self.count1):
				draw.line( ((self.pos1[int(self.ind1[i,0]),0 ],self.pos1[int(self.ind1[i,0]),1 ]),(self.pos1[int(self.ind1[i,1]),0 ],self.pos1[int(self.ind1[i,1]),1 ])),fill=(200,200,200),width = 3   )
			for i in range(self.count2):
				draw.line( ((self.pos2[int(self.ind2[i,0]),0 ],self.pos2[int(self.ind2[i,0]),1 ]),(self.pos2[int(self.ind2[i,1]),0 ],self.pos2[int(self.ind2[i,1]),1 ])),fill=(200,200,200),width = 3   )
			for i in range(self.count3):
				draw.line( ((self.pos1[int(self.ind3[i,0]),0 ],self.pos1[int(self.ind3[i,0]),1 ]),(self.pos2[int(self.ind3[i,1]),0 ],self.pos2[int(self.ind3[i,1]),1 ])),fill=(150  ,200  ,0  ),width = 3   )
				draw.line( ((self.pos2[int(self.ind3[i,0]),0 ],self.pos2[int(self.ind3[i,0]),1 ]),(self.pos1[int(self.ind3[i,1]),0 ],self.pos1[int(self.ind3[i,1]),1 ])),fill=(150  ,200  ,0  ),width = 3   )

			# each node changing diamter
			for i in range(self.num_node):
				draw.ellipse( ((tuple(self.pos1[i] - data[i,t] ))         , ( tuple(self.pos1[i] + data[i,t] ) )), fill = (0,0,0) , outline=None )
				draw.ellipse( ((tuple(self.pos2[i] - data[i+self.num_node,t] )), ( tuple(self.pos2[i] + data[i+self.num_node,t] ) )), fill = (0,0,0) , outline=None )

				# each node name
				if self.font_setting == True:
					draw.text( (self.x1_for_name[i], self.y1_for_name[i]) , str(i+1), font = self.font ,fill='#000')
					draw.text( (self.x2_for_name[i], self.y2_for_name[i]) , str(i+1), font = self.font ,fill='#000')
					draw.text( (centre_Left_net_name(self, draw).x , self.centre1[1]+self.diameter_net + self.net_name_shift) , "NET1", font = self.font ,fill='#000')
					draw.text( (centre_Right_net_name(self, draw).x, self.centre2[1]+self.diameter_net + self.net_name_shift) , "NET2", font = self.font ,fill='#000')
				else:
					draw.text( (self.x1_for_name[i], self.y1_for_name[i]) , str(i+1), fill='#000')
					draw.text( (self.x2_for_name[i], self.y2_for_name[i]) , str(i+1), fill='#000')
					draw.text( (centre_Left_net_name(self, draw).x , self.centre1[1]+self.diameter_net + self.net_name_shift) , "NET1", fill='#000')
					draw.text( (centre_Right_net_name(self, draw).x, self.centre2[1]+self.diameter_net + self.net_name_shift) , "NET2", fill='#000')

			# sensor and motor nodes changing diameters
			for i in range(self.count3):
				draw.ellipse( ((tuple(self.pos1[ int(self.ind3[i,0]) ] - data[int(self.ind3[i,0]),t] )), ( tuple(self.pos1[ int(self.ind3[i,0]) ] + data[int(self.ind3[i,0]),t]) )), fill = self.red  , outline=None )
				draw.ellipse( ((tuple(self.pos1[ int(self.ind3[i,1]) ] - data[int(self.ind3[i,1]),t] )), ( tuple(self.pos1[ int(self.ind3[i,1]) ] + data[int(self.ind3[i,1]),t]) )), fill = self.blue , outline=None )
				draw.ellipse( ((tuple(self.pos2[ int(self.ind3[i,0]) ] - data[int(self.ind3[i,0])+self.num_node,t] )), ( tuple(self.pos2[ int(self.ind3[i,0]) ] + data[int(self.ind3[i,0])+self.num_node,t]) )), fill = self.red  , outline=None )
				draw.ellipse( ((tuple(self.pos2[ int(self.ind3[i,1]) ] - data[int(self.ind3[i,1])+self.num_node,t] )), ( tuple(self.pos2[ int(self.ind3[i,1]) ] + data[int(self.ind3[i,1])+self.num_node,t]) )), fill = self.blue , outline=None )

			# legend nodes
			draw.ellipse( ((tuple(self.black_legend_pos  - self.diameter_of_each_node )), ( tuple(self.black_legend_pos  + self.diameter_of_each_node) )), fill = self.black  , outline=None )
			draw.ellipse( ((tuple(self.red_legend_pos  - self.diameter_of_each_node )), ( tuple(self.red_legend_pos  + self.diameter_of_each_node) )), fill = self.red  , outline=None )
			draw.ellipse( ((tuple(self.blue_legend_pos - self.diameter_of_each_node )), ( tuple(self.blue_legend_pos + self.diameter_of_each_node) )), fill = self.blue , outline=None )

			# title, legend & axis name
			Title = title + "_" + str((t+1)*5) + "(msec)"
			if self.font_setting == True:
				draw.text( ( centre_title(self, draw , Title).x , 50) , Title , font = self.font1, fill='#000')
				draw.text( (self.black_legend_pos[0] + self.diameter_of_each_node*2, self.black_legend_pos[1] - self.diameter_of_each_node) , self.legend_name_black, font = self.font ,fill='#000')
				draw.text( (self.red_legend_pos[0]   + self.diameter_of_each_node*2, self.red_legend_pos[1]   - self.diameter_of_each_node) , self.legend_name_red  , font = self.font ,fill='#000')
				draw.text( (self.blue_legend_pos[0]  + self.diameter_of_each_node*2, self.blue_legend_pos[1]  - self.diameter_of_each_node) , self.legend_name_blue , font = self.font ,fill='#000')
				draw.text( (ylabel_pos[0], ylabel_pos[1])  , ylabel            , font = self.font ,fill='#000')
				draw.text( (y_axis_pos[0], y_axis_pos[1])  , "1.0"             , font = self.font ,fill='#000')
				draw.text( (y_axis_pos[0], ylabel_pos[1])  , "0.5"             , font = self.font ,fill='#000')
				draw.text( (zero_point[0], zero_point[1])  , "0.0"             , font = self.font ,fill='#000')
				draw.text( (x_axis_pos[0], x_axis_pos[1])  , str(loop)+" msec" , font = self.font ,fill='#000')
				draw.text( (centre_title(self, draw , "time " + str((t+1)*5) + " msec").x, xlabel_pos[1]       )  , "time " + str((t+1)*5) + " msec" , font = self.font1 ,fill='#000')
				draw.text( (centre_title(self, draw , graph_title).x, graph_title_pos[1]-5)  , graph_title , font = self.font1 ,fill='#000')
			else:
				draw.text( ( centre_title(self, draw , Title).x , 50) , Title , fill='#000')
				draw.text( (self.black_legend_pos[0] + self.diameter_of_each_node*2, self.black_legend_pos[1] - self.diameter_of_each_node) , self.legend_name_black,fill='#000')
				draw.text( (self.red_legend_pos[0]   + self.diameter_of_each_node*2, self.red_legend_pos[1]   - self.diameter_of_each_node) , self.legend_name_red  ,fill='#000')
				draw.text( (self.blue_legend_pos[0]  + self.diameter_of_each_node*2, self.blue_legend_pos[1]  - self.diameter_of_each_node) , self.legend_name_blue ,fill='#000')
				draw.text( (ylabel_pos[0], ylabel_pos[1])  , ylabel            ,fill='#000')
				draw.text( (y_axis_pos[0], y_axis_pos[1])  , "1.0"             ,fill='#000')
				draw.text( (y_axis_pos[0], ylabel_pos[1])  , "0.5"             ,fill='#000')
				draw.text( (zero_point[0], zero_point[1])  , "0.0"             ,fill='#000')
				draw.text( (x_axis_pos[0], x_axis_pos[1])  , str(loop)+" msec" ,fill='#000')
				draw.text( (centre_title(self, draw , "time " + str((t+1)*5) + " msec").x, xlabel_pos[1]       )  , "time " + str((t+1)*5) + " msec" ,fill='#000')
				draw.text( (centre_title(self, draw , graph_title).x, graph_title_pos[1]-5)  , graph_title ,fill='#000')

			imgs.append(img)
		imgs[0].save( title +  "_.gif", save_all=True, append_images = imgs[1:], optimize=False, loop=0 )


# calculate the centre position for title
class centre_title:
	def __init__(self, Take_pic, draw, title):
		if Take_pic.font_setting == True:
			font1 = Take_pic.font1
			w, h  = draw.textsize(title, font=font1)
		else:
			w, h  = draw.textsize(title)
		self.x  = ( Take_pic.yoko-w )/2

class centre_Left_net_name:
	def __init__(self, Take_pic, draw):
		if Take_pic.font_setting == True:
			font  = Take_pic.font
			w, h  = draw.textsize("NET1", font=font)
		else:
			w, h  = draw.textsize("NET1")
		self.x  = Take_pic.yoko/4 - w/2

class centre_Right_net_name:
	def __init__(self, Take_pic, draw):
		if Take_pic.font_setting == True:
			font  = Take_pic.font
			w, h  = draw.textsize("NET2", font=font)
		else:
			w, h  = draw.textsize("NET2")
		self.x  = 3 * Take_pic.yoko/4 - w/2


if __name__ == "__main__":
	N        = 100
	edge     = 6
	p_ws     = [[0.0,0.0],[0.0,0.1],[0.0,1.0],[0.1,0.1],[0.1,1.0],[1.0,1.0]]
	for p in p_ws:
		p_ws_1   = p[0]
		p_ws_2   = p[1]
		if p_ws_1 == p_ws_2:
			NET1 = nx.watts_strogatz_graph(N, edge, p_ws_1)
			NET2 = NET1
		else:
			NET1 = nx.watts_strogatz_graph(N, edge, p_ws_1)
			NET2 = nx.watts_strogatz_graph(N, edge, p_ws_2)
		adj_mat1 = nx.to_numpy_matrix(NET1)
		adj_mat2 = nx.to_numpy_matrix(NET2)

		num_node = N
		sensor   = int(10*num_node/90)
		motor    = int(8 *num_node/90)

		temp = np.zeros(num_node**2).reshape(num_node,num_node)
		for i, j in itertools.product( xrange(sensor), xrange(motor) ):
			temp[i,j + num_node/2 + 1] = 1

		A1 = np.concatenate( ( adj_mat1,  temp ),axis = 0 )
		A2 = np.concatenate( ( temp,  adj_mat2 ),axis = 0 )
		A  = np.concatenate((A1,A2), axis = 1)

		Take_pic(A, font_setting = True).for_papers_Connected_net_pic(os.getcwd(), title = str(p_ws_1) + ":" + str(p_ws_2) + ":Connected_net_pic")
		if p_ws_2 == 0.1:
			Take_pic(A, font_setting = True).for_paper_Independent_net_pic(os.getcwd(), title = "Independent_net_pic")


	"""

	network  = nx.watts_strogatz_graph(simu_setting.num_node, num_edge, p_ws_1)
	phi = np.loadtxt("phi.txt", delimiter=',')
	A1  = np.loadtxt("adj_matrix1.txt", delimiter=',')
	A2  = np.loadtxt("adj_matrix2.txt", delimiter=',')
	num_node = phi.shape[0]/2
	loop     = phi.shape[1]

	r       = np.abs( np.mean( np.exp(1j*phi), axis=0) )
	r1      = 1 - np.abs( np.mean( np.exp(1j*phi), axis=0) )
	r2      = np.abs( np.mean( np.exp(1j*phi), axis=0) )/2
	exp_phi = np.mean( np.exp(1j*phi), axis=0) / r

	R = np.c_[r,r1]
	R = np.c_[R,r2]
	R = R.T

	Take_pic(phi, A1,A2,font_setting = True).Independent_net_pic(title="intra_net") # 
	Take_pic(phi, A1,A2,font_setting = True).Connected_net_pic(title="inter_net")
# Take_pic(phi, A1,A2,font_setting = True).Connected_net_phase_with_params_gif(R, title = "kuramoto")
	Take_pic(phi, A1,A2,font_setting = True).Connected_net_phase_with_params_gif(r, title = "orderparams")
	"""

#	Take_pic(A, font_setting = True).for_paper_Independent_net_pic(os.getcwd(), title = "Independent_net_pic")
#	Take_pic(A, font_setting = True).for_papers_Connected_net_pic(os.getcwd(), title = "Connected_net_pic")























