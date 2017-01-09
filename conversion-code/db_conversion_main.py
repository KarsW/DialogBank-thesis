### This script is outdated. The new DiAML Converter can be found at https://dialogbank.uvt.nl/representation-formats.

import pandas as pd
import numpy as np
from operator import itemgetter
import xml.etree.ElementTree as ET
import xlrd
from lxml import etree
import re


# GENERAL FUNCTIONS
# These functions are not part of one specific encoding/decoding,
# they are included in one or more other functions.

# Returns string between two characters/symbols.
def between_sym(s, start, end):
	return (s.split(start))[1].split(end)[0]


# Reads level-1 file and returns a list with tuples ('word id/token', 'word').
# Called only if input annotation file is in MultiTab or TabSW format.
def inp_level_one(level_one_file):
	one = []
	with open(level_one_file, 'r') as f:
		for line in f:
			(k, v) = line.rstrip('\n').split(':')
			one.append((k, v.strip()))
	f.close()
	return one


# Reads level-2 file and returns list with tuples ('functional segment id', ['word id/token(s) in list])
# Called only if input annotation file is in MultiTab or TabSW format.
def inp_level_two(level_two_file):
	two = []
	with open(level_two_file, 'r') as f:
		for line in f:
			(k, v) = line.rstrip('\n').split(':')
			two.append((k, v.strip().split(',')))
	f.close()
	return two


# Below are two functions that are used to carry out human sorts.
# In conversion program used to sort word ids/tokens for level-1 file (DiAML-XML to MultiTab/TabSW): (w1, w2, w3, ...),
# and combined with a lambda and another sort argument/key to sort entity structures
# based on their markables: ((fs1, fs2, fs3, ...) and not (f1, f10)).
# source: http://nedbatchelder.com/blog/200712/human_sorting.html
def tryinteger(s):
	try:
		return int(s)  # integer/numeric
	except:
		return s  # string/non-numeric


def alphanum_key(s):
	return [tryinteger(x) for x in re.split('([0-9]+)', s)]  # creates list with chunks of numeric/non-numeric data.


# 1. MULTITAB TO ABSTRACT SYNTAX
# Below are the functions that are part of the MultiTab --> Abstract syntax decoding.

# Reads MultiTab input annotation file (.xlsx) into a pandas DataFrame.
# Extra row(s) created in df if a functional segment has more than one dialogue act.
def mu_pd_data(f):
	pd.options.display.max_colwidth = 500  # allows long strings in DataFrame (Turn transcription, FS text)
	df = pd.read_excel(f, skiprows=3)
	df.columns = ['markable', 'sender', 'addressee', 'other Ps', 'turn transcription', 'fs text', 'task',
				  'autoFeedback', 'alloFeedback', 'turnManagement', 'timeManagement', 'ownCommunicationManagement',
				  'partnerCommunicationManagement', 'discourseStructuring', 'socialObligationsManagement', 'comments']
	dic_df = {}
	for dim in ['task', 'autoFeedback', 'alloFeedback', 'turnManagement', 'timeManagement',
				'ownCommunicationManagement', 'partnerCommunicationManagement', 'discourseStructuring',
				'socialObligationsManagement']:
		non_nan = df[df.apply(lambda x: not pd.isnull(x[dim]), axis=1)]
		dic_df[dim] = non_nan[
			['markable', 'sender', 'addressee', 'other Ps', 'turn transcription', 'fs text', 'comments', dim]]
	new_df = pd.concat([i for i in dic_df.values()])
	df2 = new_df.sort_index(ascending=True)  # ensures order remains (fs1, fs2, ...)
	df2.fillna('NA', inplace=True)
	return df2


# Creates lists from df columns.
# Returns zip_list: entity and link data.
# And returns fs_tt_com: data that is not in the abstract syntax:
# (markable +), FS text, Turn transcription, and Comments columns.
def mu_zipped_list(f):
	df = mu_pd_data(f)
	l1 = list(df['markable'])
	l2 = list(df['sender'])
	l3 = list(df['addressee'])
	l4 = list(df['other Ps'])
	l5 = list(df['turn transcription'])
	l6 = list(df['fs text'])
	l7 = list(df['task'])
	l8 = list(df['autoFeedback'])
	l9 = list(df['alloFeedback'])
	l10 = list(df['turnManagement'])
	l11 = list(df['timeManagement'])
	l12 = list(df['ownCommunicationManagement'])
	l13 = list(df['partnerCommunicationManagement'])
	l14 = list(df['discourseStructuring'])
	l15 = list(df['socialObligationsManagement'])
	l16 = list(df['comments'])
	zip_list = list(zip(l1, l2, l3, l4, l7, l8, l9, l10, l11, l12, l13, l14, l15))
	# fs_tt_com = irrelevant to/outside of abstract syntax, however useful for conversion to TabSW.
	fs_tt_com = list(zip(l1, l6, l5, l16))
	return zip_list, fs_tt_com


# Returns initial list of entity/link structure data.
# Entity and Link structure elements still inside the dimension key/value pairs.
# Inside entity_list there is for each structure/row a dictionary with 13 keys and values (see keys below).
def mu_init_entity(f):
	zip_list = mu_zipped_list(f)[0]
	keys = ('markable', 'sender', 'addressee', 'other Ps', 'task', 'autoFeedback', 'alloFeedback', 'turnManagement',
			'timeManagement', 'ownCommunicationManagement', 'partnerCommunicationManagement', 'discourseStructuring',
			'socialObligationsManagement')
	entity_list = []

	# Creates list with dictionary for each entity structure, see above keys.
	for i in range(len(zip_list)):
		dictionary = dict(zip(keys, zip_list[i]))
		entity_list.append(dictionary)
	return entity_list


# Removes the dimension keys that are empty/'NA' (meaning all but one per dictionary).
# Returns entity list with only one of nine dimension keys left.
def mu_remove_dims(entity_v1):
	entity_list = entity_v1
	for dic in entity_list:
		temp = dict(dic)
		for k, v in temp.items():
			if k == 'other Ps':
				continue
			elif v == 'NA':
				del dic[k]
	return entity_list


# Checks for each dictionary which one of the nine dimension keys is present.
# Accordingly, a 'dimension' key, value pair and a 'communicativeFunction'
# key, value pair is added to each dictionary. Value of communicativeFunction
# key is the old dimension value (e.g. "answer (fu: da4)", so not yet final/clean value).
def mu_func_dim(entity_v2):
	entity_list = entity_v2
	for dic in entity_list:
		if 'task' in dic:
			dic['dimension'] = 'task'
			dic['communicativeFunction'] = dic['task']
		elif 'autoFeedback' in dic:
			dic['dimension'] = 'autoFeedback'
			dic['communicativeFunction'] = dic['autoFeedback']
		elif 'alloFeedback' in dic:
			dic['dimension'] = 'alloFeedback'
			dic['communicativeFunction'] = dic['alloFeedback']
		elif 'turnManagement' in dic:
			dic['dimension'] = 'turnManagement'
			dic['communicativeFunction'] = dic['turnManagement']
		elif 'timeManagement' in dic:
			dic['dimension'] = 'timeManagement'
			dic['communicativeFunction'] = dic['timeManagement']
		elif 'ownCommunicationManagement' in dic:
			dic['dimension'] = 'ownCommunicationManagement'
			dic['communicativeFunction'] = dic['ownCommunicationManagement']
		elif 'partnerCommunicationManagement' in dic:
			dic['dimension'] = 'partnerCommunicationManagement'
			dic['communicativeFunction'] = dic['partnerCommunicationManagement']
		elif 'discourseStructuring' in dic:
			dic['dimension'] = 'discourseStructuring'
			dic['communicativeFunction'] = dic['discourseStructuring']
		elif 'socialObligationsManagement' in dic:
			dic['dimension'] = 'socialObligationsManagement'
			dic['communicativeFunction'] = dic['socialObligationsManagement']

	for dic in entity_list:  # remove the dimension key that is still present.
		dic.pop("task", None)
		dic.pop("autoFeedback", None)
		dic.pop("alloFeedback", None)
		dic.pop("turnManagement", None)
		dic.pop("timeManagement", None)
		dic.pop("ownCommunicationManagement", None)
		dic.pop("partnerCommunicationManagement", None)
		dic.pop("discourseStructuring", None)
		dic.pop("socialObligationsManagement", None)
	return entity_list


# Creates (initializes) in each dictionary a 'qualifiers' key/value pair and 'dependences' key/value pair.
# For now, their values are lists with NA's.
def mu_init_q_dep(entity_v3):
	entity_list = entity_v3
	for es in entity_list:
		es['qualifiers'] = ['NA', 'NA', 'NA']  	# certainty, conditionality, sentiment
		es['dependences'] = ['NA', 'NA']  		# functional dependence, feedback dependence
	return entity_list


# Replaces 'NA' values of 'qualifiers' keys in case qualifiers are present
# with the qualifiers' actual value (e.g. 'certain' or 'happiness').
def mu_qualifiers(entity_v4):
	entity_list = entity_v4
	for dic in entity_list:
		for k, v in dic.items():
			if '[' in v:
				s = between_sym(v, '[', ']')
				if s == 'certain' or s == 'uncertain':
					dic['qualifiers'][0] = s
				elif s == 'conditional' or s == 'unconditional':
					dic['qualifiers'][1] = s
				else:
					dic['qualifiers'][2] = s
	return entity_list


# Replaces 'NA' values of 'dependences' keys in case dependences are present
# with actual value of the dependences (e.g. "da4")
def mu_dependences(entity_v5):
	entity_list = entity_v5
	for dic in entity_list:
		for k, v in dic.items():
			if '(' in v:
				s = between_sym(v, '(', ')').split(':')
				if s[0] == 'fu' or s[0] == 'Fu':
					dic['dependences'][0] = s[1].strip()
				elif s[0] == 'fe' or s[0] == 'Fe':
					dic['dependences'][1] = s[1].strip()
	return entity_list


# Cleans list of entity structure dictionaries.
# Cleans the values of the 'communicativeFunction' keys.
# e.g. "da5: answer (Fu: da4)" to "answer")
def mu_clean_entity(entity_v6):
	entity_list = entity_v6
	for dic in entity_list:
		for k, v in dic.items():
			if k == 'communicativeFunction':
				s = v.split(':', 1)
				s2 = s[1].split()
				for cf in s2:
					if len(s2) == 1:  			# if e.g. "da1:opening"
						dic['communicativeFunction'] = cf
					else:  						# if e.g. "da11:answer(Fu:da10)" or "da11:answer[uncertain](Fu:da10)"
						dic['communicativeFunction'] = s2[0]
	return entity_list


# Returns list with data required to create link structures.
# e.g. [..., 'da3:inform', 'da4:inform {Elab_Specific da3}', ... ]
# Ultimately, list elements with curly brackets are relevant for constructing the link structures.
def mu_retrieve_link_data(f):
	temp = mu_zipped_list(f)[0]
	link_data = []
	for t in temp:
		for el in t:
			if 'da' in el:
				link_data.append(el)
	return link_data


# Returns list with each link structure inside its own dictionary.
# Each dictionary has a rhetoDact, rhetoRelatum, and rel key/value pair.
def mu_link_structures(link_v1):
	link_data = link_v1
	link_list = []
	for tup in link_data:
		link_dic = {}
		if tup.count('{') == 1:  # if one rhetorical relation.
			s = between_sym(tup, '{', '}')
			link_dic['rhetoDact'] = tup.split(':')[0]
			link_dic['rhetoRelatum'] = s.split()[1:]
			link_dic['rel'] = s.split()[0]
			link_list.append(link_dic)
		elif tup.count('{') > 1:  # if more than one rhetorical relation.
			s = tup.split('{')
			s1 = s[1].strip('}').strip('{')
			s2 = s[2].strip('}').strip('{')
			link_list.append({'rel': s1.split()[0], 'rhetoRelatum': s1.split()[1:], 'rhetoDact': s[0].split(':')[0]})
			link_list.append({'rel': s2.split()[0], 'rhetoRelatum': s2.split()[1:], 'rhetoDact': s[0].split(':')[0]})
	return link_list


# Returns abstract syntax: list of entity structures and list of link structures
def mu_entity_link(entity_v7, link_v2):
	ent = entity_v7
	link = link_v2
	return ent, link


# 2. TABSW TO ABSTRACT SYNTAX
# Below are the functions that are part of the TabSW --> Abstract syntax decoding.

# Reads TabSW annotation file (.xlsx) into a pandas DataFrame.
# splits the dactID and dacts columns/'pandas series' ('Da-ID' and 'Dialogue acts' columns) on ';'
# and adds new rows - with otherwise similar data - to df if necessary
# (i.e. if more than one dialogue act is present in a row).
def sw_pd_data(f):
	df = pd.read_excel(f, skiprows=3)
	df.columns = ['markable', 'dactID', 'dacts', 'sender', 'addressee', 'other Ps', 'fs text', 'turn transcription',
				  'comments']
	pd.options.display.max_colwidth = 500
	df.dropna()
	df2 = df[pd.notnull(df['markable'])]
	series_one = df2.dactID.str.strip("'").str.split(';', expand=True).stack().reset_index(drop=True, level=1)
	series_two = df2.dacts.str.strip("'").str.split(';', expand=True).stack().str.strip(). \
		reset_index(drop=True, level=1)
	df3 = pd.concat([series_one, series_two], keys=('dactID', 'dacts'), axis=1)
	df4 = df.drop(['dactID', 'dacts'], axis=1).join(df3).dropna(subset=['markable']).reset_index(drop=True)
	df4.fillna('NA', inplace=True)
	return df4


# Creates lists from df columns.
# Returns zip_list: entity and link data.
# And returns fs_tt_com: data that is not in the abstract syntax: (markable+) FS text, Turn transcription, Comments.
def sw_zipped_list(f):
	df = sw_pd_data(f)
	l1 = list(df['markable'])
	l2 = list(df['dactID'])
	l3 = list(df['dacts'])
	l4 = list(df['sender'])
	l5 = list(df['addressee'])
	l6 = list(df['other Ps'])
	l7 = list(df['fs text'])
	l8 = list(df['turn transcription'])
	l9 = list(df['comments'])
	zip_list = list(zip(l1, l2, l3, l4, l5, l6))
	# fs_tt_com: irrelevant to/outside of abstract syntax, however useful for conversion to MultiTab.
	fs_tt_com = list(zip(l1, l7, l8, l9))
	return zip_list, fs_tt_com


# Returns initial list of entity/link structure data.
# Entity and Link structure elements still inside the 'dact' ('Dialogue acts') key/value paris.
# Inside entity_list there is for each structure/row a dictionary with 6 keys and values.
def sw_init_entity(f):
	z = sw_zipped_list(f)[0]
	keys = ['markable', 'entityID', 'dact', 'sender', 'addressee', 'other Ps']
	entity_list = []

	# Creates list with dictionary for each entity structure, see above keys.
	for i in range(len(z)):
		dictionary = dict(zip(keys, z[i]))
		entity_list.append(dictionary)
	return entity_list


# For each entity structure/dictionary - based on its 'dact' key/value pair -
# a 'dimension' key and a 'communicativeFunction' key are created.
def sw_func_dim(entity_v1):
	entity_list = entity_v1
	for dic in entity_list:
		temp = dict(dic)
		for k, v in temp.items():
			if k == 'dact':
				d = v.split(':')[0]
				d2 = v.split(':')[1].split()
				dic['dimension'] = d
				dic['communicativeFunction'] = d2[0]
	return entity_list


# Creates (initializes) in each dictionary a 'qualifiers' key/value pair and 'dependences' key/value pair.
# For now, their values are lists with NA's.
def sw_init_q_dep(entity_v2):
	entity_list = entity_v2
	for es in entity_list:
		es['qualifiers'] = ['NA', 'NA', 'NA']  # certainty, conditionality, sentiment
		es['dependences'] = ['NA', 'NA']  # functional dependence, feedback dependence
	return entity_list


# Replaces 'NA' values of 'qualifiers' keys in case  qualifiers are present
# with the qualifiers' actual value (e.g. 'certain' or 'happiness').
def sw_qualifiers(entity_v3):
	entity_list = entity_v3
	for dic in entity_list:
		for k, v in dic.items():
			if '[' in v:
				s = between_sym(v, '[', ']')
				if s == 'certain' or s == 'uncertain':
					dic['qualifiers'][0] = s
				elif s == 'conditional' or s == 'unconditional':
					dic['qualifiers'][1] = s
				else:
					dic['qualifiers'][2] = s
	return entity_list


# Replaces 'NA' values of 'dependences' keys in case  dependences are present
# with actual value of the dependences (e.g. "da4").
def sw_dependences(entity_v4):
	entity_list = entity_v4
	for dic in entity_list:
		for k, v in dic.items():
			if '(' in v:
				s = between_sym(v, '(', ')').split(':')
				if s[0] == 'fu' or s[0] == 'Fu':
					dic['dependences'][0] = s[1].strip()
				elif s[0] == 'fe' or s[0] == 'Fe':
					dic['dependences'][1] = s[1].strip()
	return entity_list


# Returns list with data required to create link structures.
# values = one list of lists: inside lists are "da id's" and "dimension/function/qual/dependence/rhetorical link":
# e.g. [['da1', 'DS:opening'], ['da2', 'AutoF:autoPositive (Fe:da1)'], ...]
def sw_retrieve_link_data(entity_v5):
	entity_list = entity_v5
	values = []
	for dic in entity_list:
		keys = ['entityID', 'dact']
		values.append(list(itemgetter(*keys)(dic)))
	return values


# Returns list with each link structure inside its own dictionary.
# Each dictionary has a rhetoDact, rhetoRelatum, and rel key/value pair.
def sw_link_structures(link_v1):
	link_data = link_v1
	link_list = []
	for tup in link_data:
		for v in tup:
			link_dict = {}
			if v.count('{') == 1:  # if one rhetorical relation
				s = between_sym(tup[1], '{', '}')
				link_dict['rhetoRelatum'] = s.split()[1:]
				link_dict['rel'] = s.split()[0]
				link_dict['rhetoDact'] = tup[0]
				link_list.append(link_dict)
			elif v.count('{') > 1:  # if more than one rhetorical relation
				s = v.split('{')
				s1 = s[1].strip('}').strip('{')
				s2 = s[2].strip('}').strip('{')
				link_list.append({'rel': s1.split()[0], 'rhetoRelatum': s1.split()[1:], 'rhetoDact': tup[0]})
				link_list.append({'rel': s2.split()[0], 'rhetoRelatum': s2.split()[1:], 'rhetoDact': tup[0]})
	return link_list


# Clean list of entity structure dictionaries.
# Deletes 'dact', 'entityID' keys.
# Also, updates the abbreviated dimension keys.
def sw_clean_entity(entity_v5):
	entity_list = entity_v5
	for dic in entity_list:
		dic.pop("dact", None)  		# delete 'dact' key and values
		dic.pop("entityID", None)  	# delete 'entityID' keys and values.
		for k, v in dic.items():  	# update 'dimension' values
			if k == 'dimension':
				if v.lower() == 'ta':
					dic[k] = 'task'
				elif v.lower() == 'auf' or v.lower() == 'autof':
					dic[k] = 'autoFeedback'
				elif v.lower() == 'alf' or v.lower() == 'allof':
					dic[k] = 'alloFeedback'
				elif v.lower() == 'tum':
					dic[k] = 'turnManagement'
				elif v.lower() == 'tim':
					dic[k] = 'timeManagement'
				elif v.lower() == 'tum':
					dic[k] = 'turnManagement'
				elif v.lower() == 'ocm':
					dic[k] = 'ownCommunicationManagement'
				elif v.lower() == 'pcm':
					dic[k] = 'partnerCommunicationManagement'
				elif v.lower() == 'ds':
					dic[k] = 'discourseStructuring'
				elif v.lower() == 'som':
					dic[k] = 'socialObligationsManagement'
	return entity_list


# returns abstract syntax: list of entity structures and list of link structures.
def sw_entity_link(entity_v6, link_v2):
	entity = entity_v6
	link = link_v2
	return entity, link


# 3. XML TO ABSTRACT SYNTAX
# Below are the functions that are part of the DiAML-XML --> Abstract syntax decoding.

# Retrieves all data from DiAML-XML input annotation file.
# Returns list with entity data and list with link data.
# dialogueAct elements to entity list.
# rhetoricalLink elements to link list.
def xml_data(f):
	tree = ET.parse(f)
	root = tree.getroot()
	text = root.getchildren()[1]  			# <text> element
	div = text.getchildren()  				# <div> element
	entity_data = div[2]  					# 2nd <div> element = dialogueAct and rhetoricalLink XML-elements
	default_uri = '{http://www.w3.org/XML/1998/namespace}id'
	link_list = []
	entity_list = []

	for dact in entity_data:
		attr = dact.get(default_uri)
		if default_uri in dact.attrib:
			del dact.attrib[default_uri]
			dact.set('entityID', attr)
		if 'dialogueAct' in dact.tag:  		# Adds data from 'dialogueAct' XML-elements to entity list.
			entity_list.append(dact.attrib)
		else:
			link_list.append(dact.attrib)  	# Adds data from 'rhetoricalLink' XML-elements to link list.

	for dic in entity_list:
		for k, v in dic.items():
			dic[k] = v.replace('#', '')  	# removes '#' symbol

	for dic in link_list:
		for k, v in dic.items():
			dic[k] = v.replace('#', '')
	return entity_list, link_list


# Retrieve level-1 and level-2 data for conversion to FS text and
# Turn transcription columns in MultiTab and TabSW formats
# (returns 'segments' and 'word_id' variables for this purpose).
# Also, level-1 file and level-2 file are created, since output format is TabSW or MultiTab.
def fs_words(f, name):
	default_uri = '{http://www.w3.org/XML/1998/namespace}id'
	tree = ET.parse(f)
	for word in tree.iter(tag="spanGrp"):
		attr = word.get(default_uri)
		if default_uri in word.attrib:
			del word.attrib[default_uri]
			word.set('spanGrp:id', attr)

	# sp_list = list with 1: dicts with 'spanGrp:id' and 'type' attributes,
	# and 2: dicts with 'xml:id' and 'from' attributes.
	sp_list = []
	for parent in tree.getiterator():
		if 'spanGrp:id' in parent.keys():
			sp_list.append(parent.attrib)
		elif 'from' in parent.keys():
			sp_list.append(parent.attrib)
		else:
			continue

	# fs_seg = list with functional segments ids: [name-fs1,name-fs2, ... ]
	fs_seg = []
	for element in tree.iter(tag="fs"):
		fs_seg.append(element.attrib[default_uri])

	# Iterates over sp_list.
	# Adds lists to word_nums, inside lists are word IDs/tokens ('from' values) belonging to functional segments.
	word_nums = []
	temp = []
	for x in sp_list:
		if 'from' in x:
			temp.append(x['from'][1:])
		else:
			if temp:
				word_nums.append(temp)
				temp = []
	word_nums.append([])
	word_nums[-1].append(sp_list[-1]['from'][1:])

	segments = list(zip(fs_seg, word_nums))  # List with tuples: (functional segment, [words ID(s)])

	words = []
	id_list = []
	for e in tree.iter(tag="w"):
		attr = e.get(default_uri)
		id_list.append(attr)  # word IDs/tokens
		words.append(e.text)  # words
	word_id = dict(zip(id_list, words))  # Dictionary of all word ids/tokens and words: {wordid: word, ...}

	# Create level-1 and level-2 file for MultiTab and TabSW formats.
	# level-1 tokenization
	words_sorted = sorted(word_id, key=lambda x: alphanum_key(x))  # human sort on word id
	word_ids_sorted = [(w, word_id.get(w)) for w in words_sorted]  # retrieve corresponding word from 'word_id'
	with open(name + "_tokenization.txt", 'w') as file:  		   # level-1 file: "w1: hello"
		for id, w in word_ids_sorted:
			lines = str('{}: {}\n'.format(id, w))
			file.write(lines)
	file.close()

	# level-2 segmentation
	with open(name + "_segmentation.txt", 'w') as file:  		   # level-2 file: "fs1: w1, w2, w3"
		for i in range(len(word_nums)):
			file.write(str(fs_seg[i] + ': ' + ','.join(map(repr, (word_nums[i]))).replace("'", '') + '\n'))
	file.close()
	return segments, word_id


# Creates and returns list with entity data and link data in dictionaries.
# Six entity structure keys already present after this function.
# Also, 'qualifiers' and 'dependences' key/value pairs are created.
def xml_entity_data(all_data):
	data = all_data
	entity_data = data[0]

	# id_list = []            	# entityID
	sen_list = []  				# sender
	add_list = []  				# addressee
	op_list = []  				# other Ps
	target_list = []  			# functional segment/markable/target
	dim_list = []  				# dimension
	comm_func_list = []  		# communicative function
	cert_list = []  			# certainty
	cond_list = []  			# conditionality
	sent_list = []  			# sentiment
	fu_dep_list = []  			# functional dependences
	fb_dep_list = []  			# feedback dependences

	# Fill above lists.
	for x in entity_data:
		sen_list.append(x.get('sender'))
		if 'addressee' in entity_data[0]:
			add_list.append(x.get('addressee'))
		else:
			add_list.append(x.get('addresse'))  # sometimes misspelled in original annotations.
		op_list.append(x.get('otherParticipant'))
		target_list.append(x.get('target'))
		dim_list.append(x.get('dimension'))
		comm_func_list.append(x.get('communicativeFunction'))
		cert_list.append(x.get('certainty'))
		cond_list.append(x.get('conditionality'))
		sent_list.append(x.get('sentiment'))
		fb_dep_list.append(x.get('feedbackDependence'))
		fu_dep_list.append(x.get('functionalDependence'))

	# Set value 'NA' if not present.
	sen_list = ['NA' if v is None else v for v in sen_list]
	add_list = ['NA' if v is None else v for v in add_list]
	op_list = ['NA' if v is None else v for v in op_list]
	target_list = ['NA' if v is None else v for v in target_list]
	dim_list = ['NA' if v is None else v for v in dim_list]
	comm_func_list = ['NA' if v is None else v for v in comm_func_list]
	cert_list = ['NA' if v is None else v for v in cert_list]
	cond_list = ['NA' if v is None else v for v in cond_list]
	sent_list = ['NA' if v is None else v for v in sent_list]
	fb_dep_list = ['NA' if v is None else v for v in fb_dep_list]
	fu_dep_list = ['NA' if v is None else v for v in fu_dep_list]

	keys = ['sender', 'addressee', 'other Ps', 'markable', 'dimension', 'communicativeFunction']
	entity_data = list(zip(sen_list, add_list, op_list, target_list, dim_list, comm_func_list))
	entity_list = []

	# Creates dictionaries for entity structures with above keys.
	for i in range(len(entity_data)):
		dictionary = dict(zip(keys, entity_data[i]))
		entity_list.append(dictionary)

	# Adds 'qualifiers' and 'dependences' keys (values are lists) to dictionaries.
	for i in range(len(entity_list)):
		entity_list[i]['qualifiers'] = [cert_list[i], cond_list[i], sent_list[i]]
		entity_list[i]['dependences'] = [fu_dep_list[i], fb_dep_list[i]]
	return entity_list


# Creates and returns list with link structure data inside dictionaries.
def xml_link_data(all_data):
	data = all_data
	link_data = data[1]
	link_list = []
	for link in link_data:
		link_list.append(
			{'rel': link.get('rhetoRel'), 'rhetoRelatum': link.get('rhetoAntecedent').split(), 'rhetoDact': link.get('dact')})
	return link_list


# Returns abstract syntax: list of entity structures and list of link structures.
def xml_entity_link(entity_v1, link_v1):
	entity = entity_v1
	link = link_v1
	return entity, link


# 4. ABSTRACT SYNTAX TO MULTITAB
# Below are the functions that are part of the Abstract syntax --> MultiTab encoding.

# Sorts entity structures according to their markables. If there are similar markables, sorts on their dimension.
# Assigns identifier to entity structures.
# Extracts entity elements from pd df columns: communicative function, qualifiers and dependences.
# Also (re)-arranges/orders DataFrame columns.
def extract_ent_mu(abs_syn):
	ent = abs_syn[0]  # list of entity structure dictionaries.

	order = {'task': 1, 'autoFeedback': 2, 'alloFeedback': 3, 'turnManagement': 4, 'timeManagement': 5,
			 'ownCommunicationManagement': 6, 'partnerCommunicationManagement': 7, 'discourseStructuring': 8,
			 'socialObligationsManagement': 9}
	ent_sorted = sorted(ent, key=lambda x: (alphanum_key(x['markable']), order[x['dimension']]))

	count = 0
	for dic in ent_sorted:
		count += 1
		dic['entityID'] = 'da' + str(count)

	df_ent = pd.DataFrame(ent_sorted)    # df of sorted entity structures
	df_link = pd.DataFrame(abs_syn[1])   # df of link structures

	for c, li in enumerate(df_ent['qualifiers']):  # adds qualifiers after comm function
		cert = li[0]
		cond = li[1]
		sent = li[2]
		if cert != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + ' [' + str(cert) + ']'
		elif cond != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + ' [' + str(cond) + ']'
		elif sent != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + ' [' + str(sent) + ']'

	for c, li in enumerate(df_ent['dependences']):  # adds dependences after comm function (and possible qualifiers)
		fu = li[0]
		fe = li[1]
		if fu != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + ' (Fu: ' + str(fu) + ')'
		elif fe != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + ' (Fe: ' + str(fe) + ')'

	# drops 'dependences' and 'qualifiers' columns from df, and re-orders columns.
	df_ent.drop(['dependences', 'qualifiers'], inplace=True, axis=1)
	df_ent = df_ent[['markable', 'sender', 'addressee', 'other Ps', 'entityID', 'dimension', 'communicativeFunction']]
	return df_ent, df_link  # returns df with entity structure data and df with link structure data


# Merges entity df and link df.
# The merge operation is on the entity structure identifiers ('entityID') and link structures' 'rhetoDact'.
# The values of the 'rhetoRelatum' and 'rel' keys are added to the values of the 'communicativeFunction' keys.
# So, this value joins the comm function, possible qualifiers, and possible dependence relation.
# Finally, adds nine dimension columns to df. These have no values yes (np.nan).
def extract_link_mu(mu_v1):
	df_ent = mu_v1[0]
	df_link = mu_v1[1]
	df_ent_link = df_ent.merge(df_link, how='left', left_on='entityID', right_on='rhetoDact', sort=False)

	# (re)-arrange link data
	for c, relation in enumerate(df_ent_link['rel']):  			# adds rel to comm function value
		if relation is not np.nan:
			df_ent_link.loc[c, 'communicativeFunction'] = df_ent_link.loc[c, 'communicativeFunction'] \
														  + ' {' + str(relation) + ' '

	for c, relatum in enumerate(df_ent_link['rhetoRelatum']):   # adds relatum to comm function value
		if relatum is not np.nan:
			df_ent_link.loc[c, 'communicativeFunction'] += (str(relatum) + '}').replace('[', '') \
				.replace(']', '').replace("'", '')

	# drop these columns, since necessary data is now added to 'communicativeFunction' values.
	df_ent_link.drop(['rel', 'rhetoDact', 'rhetoRelatum'], inplace=True, axis=1)

	# initialize nine dimension columns required for output annotation.
	df_ent_link['Task'] = np.nan
	df_ent_link['autoFeedback'] = np.nan
	df_ent_link['alloFeedback'] = np.nan
	df_ent_link['turnManagement'] = np.nan
	df_ent_link['timeManagement'] = np.nan
	df_ent_link['ownCommunicationManagement'] = np.nan
	df_ent_link['partnerCommunicationManagement'] = np.nan
	df_ent_link['discourseStructuring'] = np.nan
	df_ent_link['socialObligationsManagement'] = np.nan

	# returns df containing - additionally - rhetorical relations, and empty dimension columns.
	return df_ent_link


# Nine dimension columns are empty. Based on the value in the 'dimension' column the actual nine dimensions are filled
# with (1) the entity structure identifier and (2) the value of the 'communicativeFunction' key.
# A cell in one of these dimension columns may now look as follows:
# 'Task' column: "da4:answer [certain] {cause:reason da3}" or 'autoFeedback' column "da6:autoPositive(Fe:da5)".
# 'entityID', 'dimension', and 'communicativeFunction' columns are dropped from the df.
def dim_cols_mu(mu_v2):
	df1 = mu_v2
	dim = df1['dimension']
	for c, el in enumerate(dim):
		if el.lower() == 'task':
			df1.loc[c, 'Task'] = df1.loc[c, 'entityID'] + ': ' + df1.loc[c, 'communicativeFunction']
		elif el.lower() == 'autofeedback':
			df1.loc[c, 'autoFeedback'] = df1.loc[c, 'entityID'] + ': ' + df1.loc[c, 'communicativeFunction']
		elif el.lower() == 'allofeedback':
			df1.loc[c, 'alloFeedback'] = df1.loc[c, 'entityID'] + ': ' + df1.loc[c, 'communicativeFunction']
		elif el.lower() == 'turnmanagement':
			df1.loc[c, 'turnManagement'] = df1.loc[c, 'entityID'] + ': ' + df1.loc[c, 'communicativeFunction']
		elif el.lower() == 'timemanagement':
			df1.loc[c, 'timeManagement'] = df1.loc[c, 'entityID'] + ': ' + df1.loc[c, 'communicativeFunction']
		elif el.lower() == 'owncommunicationmanagement':
			df1.loc[c, 'ownCommunicationManagement'] = df1.loc[c, 'entityID'] + ': ' + \
													   df1.loc[c, 'communicativeFunction']
		elif el.lower() == 'partnercommunicationmanagement':
			df1.loc[c, 'partnerCommunicationManagement'] = df1.loc[c, 'entityID'] + ': ' + \
														   df1.loc[c, 'communicativeFunction']
		elif el.lower() == 'discoursestructuring':
			df1.loc[c, 'discourseStructuring'] = df1.loc[c, 'entityID'] + ': ' + df1.loc[c, 'communicativeFunction']
		elif el.lower() == 'socialobligationsmanagement':
			df1.loc[c, 'socialObligationsManagement'] = df1.loc[c, 'entityID'] + ': ' + \
														df1.loc[c, 'communicativeFunction']

	df1.drop(['dimension', 'communicativeFunction', 'entityID'], inplace=True, axis=1)
	return df1


# Function that adds to df the FS text, Turn transcription, and Comments columns.
# If input annotation format is TabSW (num = 1) these columns are
# 'copy pasted', outside of the abstract syntax.
# If input annotation format is DiAML-XML (num = 2) these are retrieved from/transformed based on
# the output of the zipped_list function (fs_tt_com: Markable, FS text, TT, Comments).
# In case there are duplicate markables in the df (more than one row with a similar markable)
# their rows are merged/combined, meaning that multiple dialogue acts may now be now represented in the same row.
def fs_tt_com_mu(mu_v3, cols, num):
	df1 = mu_v3
	df1['FS text'] = np.nan
	df1['Turn transcription'] = np.nan
	df1['Comments'] = np.nan
	df2 = df1.fillna('NA')

	if num == 1:  							# 1 if input is TabSW - columns are 'in memory'.
		new_cols = cols  					# Add FS text, Turn transcription, comments to DataFrame.
		for c, el in enumerate(new_cols):
			df1.loc[c, 'FS text'] = el[1]
			df1.loc[c, 'Turn transcription'] = el[2]
			df1.loc[c, 'Comments'] = el[3]

		df2 = df1.groupby(['markable'], sort=False).first().reset_index()  # merge/combine transformation on markables
		df3 = df2[
			['markable', 'sender', 'addressee', 'other Ps', 'Turn transcription', 'FS text', 'Task', 'autoFeedback',
			 'alloFeedback', 'turnManagement', 'timeManagement', 'ownCommunicationManagement',
			 'partnerCommunicationManagement', 'discourseStructuring', 'socialObligationsManagement', 'Comments']]
		df3.columns = ['Markables', 'Sender', 'Addressee', 'other Ps', 'Turn transcription', 'FS text', 'Task',
					   'autoFeedback', 'alloFeedback', 'turnManagement', 'timeManagement', 'ownCommunicationManagement',
					   'partnerCommunicationManagement', 'discourseStructuring', 'socialObligationsManagement',
					   'Comments']
		df_final = df3.replace('NA', np.nan)

	# 2 if input is XML. There are no pre-existing tabular formatted FS text, Turn transcription, and Comments columns.
	elif num == 2:

		df3 = df2.groupby(['markable'], sort=False).first().reset_index()  # groups similar markables
		df4 = df3[
			['markable', 'sender', 'addressee', 'other Ps', 'Turn transcription', 'FS text', 'Task', 'autoFeedback',
			 'alloFeedback', 'turnManagement', 'timeManagement', 'ownCommunicationManagement',
			 'partnerCommunicationManagement', 'discourseStructuring', 'socialObligationsManagement', 'Comments']]
		df4.columns = ['Markables', 'Sender', 'Addressee', 'other Ps', 'Turn transcription', 'FS text', 'Task',
					   'autoFeedback', 'alloFeedback', 'turnManagement', 'timeManagement', 'ownCommunicationManagement',
					   'partnerCommunicationManagement', 'discourseStructuring', 'socialObligationsManagement',
					   'Comments']

		# Fill FS text column.
		fs = cols[0]
		words = cols[1]
		tt_list = []
		for c, el in enumerate(df4['Markables']):  	# count, fs1/fs2/fs3
			for f in fs:  							# (fs1, [w1,w2,w3]
				if el == f[0]:  					# find words corresponding with word numbers
					items = [x for x in f[1]] 	 	# and add words to correct FS text row in df
					tt_list.append({el: {x: y for x, y in words.items() if x in items}})
					for w in items:
						content = str(words[w]) + ' '
						df4.loc[c, 'FS text'] += content

		for c, el in enumerate(df4['FS text']):  # deletes 'NA' in front of content in FS text rows.
			if str(el).startswith('NA'):
				df4.loc[c, 'FS text'] = str(el)[2:]

		# Fill Turn transcription column.
		fs2 = []  								 # fs2 = List of tuples [((segment id, [list of words]), sender), ...]
		for c, el in enumerate(df4['Sender']):
			tt_list[c]['s'] = el
			fs2.append((fs[c], el))

		tt_two = []										# iterates over fs2. Keeps track of who is the current sender.
		sender = None									# If sender/turn does not change adds words to the existing list of
		for s, tup in [(s, tup) for (tup, s) in fs2]:   # words. If sender changes create a new list and add the words
			if s != sender:								# to that list, etc. To each list the first functional segment id
				func_seg = tup[0]						# of that turn is added (to know where in the df to add the words).
				tt_two.append({func_seg: []})			# This way a Turn transcription list is created.
			tt_two[-1][func_seg].extend(tup[1])
			sender = s

		for fs in tt_two:										# Iterates over tt_two. Checks segment id, adds to that
			for k, v in fs.items():  # k = fs1, v = [w1, w2]	# Turn transcription row in the df, belonging to that segment,
				for w in v:										# the words corresponding with the word ids in 'words' dictionary.
					for c, seg in enumerate(df4['Markables']):
						if seg == k:
							df4.loc[c, 'Turn transcription'] += str(words.get(w)) + ' '

		for c, el in enumerate(df4['Turn transcription']):  	# delete 'NA' in front of content in FS text rows.
			if str(el).startswith('NA'):
				df4.loc[c, 'Turn transcription'] = str(el)[2:]

		df_final = df4.replace('NA', np.nan)
	return df_final


# Creates .xlsx file based on above DataFrame.
# Formats the file. Start row, Title, Bold, Wrap etc.
def to_multitab(mu_v4, name):
	final_df = mu_v4

	writer = pd.ExcelWriter(str(name) + '_DiAML-MultiTab.xlsx', engine='xlsxwriter')
	final_df.to_excel(writer, sheet_name='Sheet1', index=False, startrow=3)

	workbook = writer.book
	worksheet = writer.sheets['Sheet1']

	bold = workbook.add_format({'bold': True})
	wrap = workbook.add_format({'text_wrap': True})

	worksheet.write('A1', 'Dialogue "{}", Gold standard ISO 24617-2 annotation represented in DiAML-MultiTab format.'
					.format(name), bold)
	worksheet.set_row(3, 30, wrap)
	worksheet.set_column(0, 0, 15, wrap)
	worksheet.set_column(1, 2, 10, wrap)
	worksheet.set_column(4, 5, 25, wrap)
	worksheet.set_column(6, 16, 20, wrap)
	writer.close()
	return "The conversion has been successfully executed. The annotation file '" + name + \
		   "_DiAML-MultiTab.xlsx' has been created.\n"


# 5. ABSTRACT SYNTAX TO TABSW
# Below are the functions that are part of the Abstract syntax --> TabSW encoding.


# Sorts entity structures. First, sorts on markables.
# If the markables of two entity structures are similar, sorts on dimension.
# Adds identifier to each entity structure.
# Creates entity df from sorted entity structures. Also creates link df.
# Adds qualifiers and dependences to cells in 'communicativeFunction' column. Drops qualifiers and dependences columns.
def extract_ent_sw(abs_syn):
	ent = abs_syn[0]

	order = {'task': 1, 'autoFeedback': 2, 'alloFeedback': 3, 'turnManagement': 4, 'timeManagement': 5,
			 'ownCommunicationManagement': 6, 'partnerCommunicationManagement': 7, 'discourseStructuring': 8,
			 'socialObligationsManagement': 9}
	ent_sorted = sorted(ent, key=lambda x: (alphanum_key(x['markable']), order[x['dimension']]))

	count = 0  # entity structure identifier
	for dic in ent_sorted:
		count += 1
		dic['entityID'] = 'da' + str(count)

	df_ent = pd.DataFrame(ent_sorted)
	link = abs_syn[1]
	df_link = pd.DataFrame(link)

	for c, li in enumerate(df_ent['qualifiers']):  # Adds qualifiers to communicativeFunction column.
		cert = li[0]  # certainty
		cond = li[1]  # conditionality
		sent = li[2]  # sentiment
		if cert != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + ' [' + str(cert) + '] '
		elif cond != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + ' [' + str(cond) + '] '
		elif sent != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + ' [' + str(sent) + '] '

	for c, li in enumerate(df_ent['dependences']):  # Adds dependences to communicativeFunction column.
		fu = li[0]  # functional dependence
		fe = li[1]  # feedback dependence
		if fu != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + ' (Fu:' + str(fu) + ')'
		elif fe != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + ' (Fe:' + str(fe) + ')'

	df_ent.drop(['dependences', 'qualifiers'], inplace=True, axis=1)  # Drops 'dependences' and 'qualifiers' columns.
	df_ent = df_ent[
		['markable', 'sender', 'addressee', 'other Ps', 'entityID', 'dimension', 'communicativeFunction']]  # reorder.
	return df_ent, df_link


# Merges entity df and link df.
# The merge operation is on the entity structure ids and link structures' rhetoDact.
# The values of the rhetoRelatum and rel keys are added to the communicativeFunction column.
# So, this value joins the comm function, possible qualifiers and dependence relation.
# Finally, empty 'Dialogue acts' column is added to df.
def extract_link_sw(sw_v1):
	df_ent = sw_v1[0]
	df_link = sw_v1[1]
	df_ent_link = df_ent.merge(df_link, how='left', left_on='entityID', right_on='rhetoDact', sort=False)

	# (re)-arrange link data
	for c, relation in enumerate(df_ent_link['rel']):  # adds rhetorical relation
		if relation is not np.nan:
			df_ent_link.loc[c, 'communicativeFunction'] = df_ent_link.loc[c, 'communicativeFunction'] + ' {' + str(
				relation) + ' '

	for c, relatum in enumerate(df_ent_link['rhetoRelatum']):  # adds relatum
		if relatum is not np.nan:
			df_ent_link.loc[c, 'communicativeFunction'] += (str(relatum) + '}').replace('[', '').replace(']', '') \
				.replace("'", '')

	df_ent_link.drop(['rel', 'rhetoDact', 'rhetoRelatum'], inplace=True, axis=1)
	df_ent_link['Dialogue acts'] = np.nan
	return df_ent_link  # returns df containing qualifiers, dependences, and rhetorical relations.


# Adds dimension abbreviation and communicativeFunction value to (each row in) the 'Dialogue acts' column.
# Also, the dimension and communicativeFunction columns are dropped from the df.
def dact_col_sw(sw_v2):
	df1 = sw_v2
	dim = df1['dimension']  # Pandas Series: dimension column
	for c, el in enumerate(dim):
		if el.lower() == 'task':
			df1.loc[c, 'Dialogue acts'] = 'Ta:' + df1.loc[c, 'communicativeFunction']
		elif el.lower() == 'autofeedback':
			df1.loc[c, 'Dialogue acts'] = 'AutoF:' + df1.loc[c, 'communicativeFunction']
		elif el.lower() == 'allofeedback':
			df1.loc[c, 'Dialogue acts'] = 'alloF:' + df1.loc[c, 'communicativeFunction']
		elif el.lower() == 'turnmanagement':
			df1.loc[c, 'Dialogue acts'] = 'TuM:' + df1.loc[c, 'communicativeFunction']
		elif el.lower() == 'timemanagement':
			df1.loc[c, 'Dialogue acts'] = 'TiM:' + df1.loc[c, 'communicativeFunction']
		elif el.lower() == 'owncommunicationmanagement':
			df1.loc[c, 'Dialogue acts'] = 'OCM:' + df1.loc[c, 'communicativeFunction']
		elif el.lower() == 'partnercommunicationmanagement':
			df1.loc[c, 'Dialogue acts'] = 'PCM:' + df1.loc[c, 'communicativeFunction']
		elif el.lower() == 'discoursestructuring':
			df1.loc[c, 'Dialogue acts'] = 'DS:' + df1.loc[c, 'communicativeFunction']
		elif el.lower() == 'socialobligationsmanagement':
			df1.loc[c, 'Dialogue acts'] = 'SOM:' + df1.loc[c, 'communicativeFunction']
	df1.drop(['dimension', 'communicativeFunction'], inplace=True, axis=1)
	return df1


# Function that adds to df the FS text, Turn transcription, and Comments columns.
# If input annotation format is MultiTab (num = 1) these columns are
#  'copy pasted', outside of the abstract syntax.
# If input annotation format is DiAML-XML (num = 2) these are retrieved from/transformed based on
# the output of the zipped_list function (fs_tt_com: Markable, FS text, TT, Comments).
# In case there are duplicate markables in the df (i.e. more than one row with a similar markable)
# their rows are merged/combined, meaning that multiple dialogue acts may now be now represented in the same row.
def fs_tt_com_sw(sw_v3, cols, num):
	df1 = sw_v3
	df1['FS text'] = np.nan
	df1['Turn transcription'] = np.nan
	df1['Comments'] = np.nan
	df2 = df1.fillna('NA')

	if num == 1:  				# only if input is MultiTab
		new_cols = cols  		# add (copy/taste) FS text, Turn transcription, Comments columns to df.
		for c, el in enumerate(new_cols):
			df2.loc[c, 'FS text'] = el[1]
			df2.loc[c, 'Turn transcription'] = el[2]
			df2.loc[c, 'Comments'] = el[3]

		df3 = df2[
			['markable', 'entityID', 'Dialogue acts', 'sender', 'addressee', 'other Ps', 'FS text',
			 'Turn transcription',
			 'Comments']]
		df4 = df3.groupby(['markable', 'sender', 'addressee', 'other Ps', 'FS text', 'Turn transcription', 'Comments'],
						  sort=False).agg('; '.join).reset_index()
		df4.columns = ['Markables', 'Sender', 'Addressee', 'Other Ps', 'FS text', 'Turn transcription', 'Comments',
					   'Da-ID', 'Dialogue acts']
		df5 = df4[['Markables', 'Da-ID', 'Dialogue acts', 'Sender', 'Addressee', 'Other Ps', 'FS text',
				   'Turn transcription', 'Comments']]
		df_final = df5.replace('NA', np.nan)

	# 2 if input is XML. There are no pre-existing tabular formatted FS text, Turn transcription, and Comments columns.
	elif num == 2:
		df3 = df2[
			['markable', 'entityID', 'Dialogue acts', 'sender', 'addressee', 'other Ps', 'FS text',
			 'Turn transcription', 'Comments']]
		df4 = df3.groupby(['markable', 'sender', 'addressee', 'other Ps', 'FS text', 'Turn transcription', 'Comments'],
						  sort=False).agg('; '.join).reset_index()
		df4.columns = ['Markables', 'Sender', 'Addressee', 'Other Ps', 'FS text', 'Turn transcription', 'Comments',
					   'Da-ID', 'Dialogue acts']

		# Fill FS text column.
		fs = cols[0]
		words = cols[1]
		tt_list = []
		for c, el in enumerate(df4['Markables']):  	# count, fs1/fs2/fs3
			for f in fs:  							# (fs1, [w1,w2,w3]
				if el == f[0]:  					# find words corresponding to word numbers
					items = [x for x in f[1]]  		# and add words to correct FS text row in df
					tt_list.append({el: {x: y for x, y in words.items() if x in items}})
					for w in items:
						content = str(words[w]) + ' '
						df4.loc[c, 'FS text'] += content

		for c, el in enumerate(df4['FS text']):  	# deletes 'NA' in front of content in FS text rows.
			if str(el).startswith('NA'):
				df4.loc[c, 'FS text'] = str(el)[2:]

		# Fill Turn transcription column.
		fs2 = []  									# fs2 = List of tuples [((segment id, list of words), sender), ...]
		for c, el in enumerate(df4['Sender']):
			tt_list[c]['s'] = el
			fs2.append((fs[c], el))

		tt_two = []  									# tt_two is a list of dicts: [{fs_id: [word nums], ...]
		sender = None  									# This is one of the senders in the loop.
		for s, tup in [(s, tup) for (tup, s) in fs2]:  	# s = sender, tup = (fs_id, [word nums])
			if s != sender:  							# If turn changes.
				func_seg = tup[0]  						# fs_id when turn/sender changes
				tt_two.append({func_seg: []})
			tt_two[-1][func_seg].extend(tup[1])  		# adds word nums if turn/sender is similar
			sender = s									# See also: fs_tt_com_mu function

		for fs in tt_two:								# retrieve and add to df words corresponding to word numbers
			for k, v in fs.items():  					# k = fs1, v = [w1, w2]
				for w in v:
					for c, el in enumerate(df4['Markables']):
						if el == k:
							df4.loc[c, 'Turn transcription'] += str(words.get(w)) + ' '

		for c, el in enumerate(df4['Turn transcription']):  # deletes 'NA' in front of content in FS text rows.
			if str(el).startswith('NA'):
				df4.loc[c, 'Turn transcription'] = str(el)[2:]

		df5 = df4[
			['Markables', 'Da-ID', 'Dialogue acts', 'Sender', 'Addressee', 'Other Ps', 'FS text', 'Turn transcription',
			 'Comments']]
		df_final = df5.replace('NA', np.nan)
	return df_final


# Creates .xlsx file based on above DataFrame.
# Formats the file. Start row, Title, Bold, Wrap etc.
def to_tabsw(sw_v4, name):
	final_df = sw_v4

	writer = pd.ExcelWriter(str(name) + '_DiAML-TabSW.xlsx', engine='xlsxwriter')
	final_df.to_excel(writer, sheet_name='Sheet1', index=False, startrow=3)

	workbook = writer.book
	worksheet = writer.sheets['Sheet1']

	bold = workbook.add_format({'bold': True})
	wrap = workbook.add_format({'text_wrap': True})

	worksheet.write('A1', 'Dialogue "{}", Gold standard ISO 24617-2 annotation represented in DiAML-TabSW format.'
					.format(name), bold)
	worksheet.set_row(3, 30, wrap)
	worksheet.set_column(0, 0, 20)
	worksheet.set_column(1, 1, 10)
	worksheet.set_column(2, 2, 30, wrap)
	worksheet.set_column(6, 7, 30, wrap)
	writer.close()
	return "The conversion has been successfully executed. The annotation file '" + name + \
		   "_DiAML-TabSW.xlsx' has been created.\n"


# 6. ABSTRACT SYNTAX TO XML
# Below are the functions that are part of the Abstract syntax --> DiAML-XML encoding.

# Below is one 'large' function that does all the work.
# This includes XML Namespaces, creation of all the XML elements, the addition of the level-1 and level-2 data,
# the addition of level-3 entity and link structure data to the <dialogueAct> and <rhetoricalLink> elements,
# and the creation/writing of the output file.
def create_xml(level_one, level_two, abs_syn, name):
	# entity structures and link structures
	ent = abs_syn[0]
	link = abs_syn[1]

	# All XML namespaces in the DiAML-XML format.
	tei_ns = 'http://www.tei-c.org/ns/1.0'
	diaml_ns = 'http://www.iso.org/diaml'
	xml_ns = 'http://www.w3.org/XML/1998/namespace'
	ns_map = {"tei": tei_ns,
			  "diaml": diaml_ns, "xml": xml_ns}

	# Create XML root element and sub elements.
	root = etree.Element('TEI', nsmap=ns_map)
	tree = etree.ElementTree(root)

	# participants
	participants = etree.SubElement(root, 'profileDescr')
	ps_list = []
	for dic in ent:  						# all senders
		for k, v in dic.items():
			if k == 'sender':
				ps_list.append(v)
	temp_list = []
	for dic in ent:  						# all 'other participants'
		for k, v in dic.items():
			if k == 'other Ps':
				if v != 'NA':
					s = v.split(',')
					temp_list.append(s)
	for li in temp_list:  					# all participants
		for p in li:
			ps_list.append(p.strip())
	all_ps = list(set(ps_list))  			# remove duplicate participants
	for p in all_ps:  						# create element for each unique participant
		etree.SubElement(participants, 'particDescr',
						 attrib={'{http://www.w3.org/XML/1998/namespace}id': p})

	# Create Sub elements
	text = etree.SubElement(root, 'text')
	div1 = etree.SubElement(text, 'div')
	words = etree.SubElement(div1, 'u')
	div2 = etree.SubElement(text, 'div')
	diaml = etree.SubElement(text, 'diaml', nsmap={None: diaml_ns})

	# add level-1 words from 'level_one'
	for (id, w) in level_one:
		word = etree.SubElement(words, 'w')
		word.set('{http://www.w3.org/XML/1998/namespace}id', id)
		word.text = w

	# add level-2 functional segments from 'level_two'
	count = 0
	for segment in level_two:
		count += 1
		spanGrp = etree.SubElement(div2, 'spanGrp',
								   attrib={'{http://www.w3.org/XML/1998/namespace}id': 'ves' + str(count)})
		spanGrp.set('type', 'functionalVerbalSegment')
		for span in segment[1]:  # word ids from list
			sp = etree.SubElement(spanGrp, 'span')
			sp.set('{http://www.w3.org/XML/1998/namespace}id', 'ts' + str(count))
			sp.set('from', '#'+span)
		fs = etree.SubElement(div2, 'fs', attrib={'type': 'functionalSegment'})
		fs.set('{http://www.w3.org/XML/1998/namespace}id', segment[0])
		f = etree.SubElement(fs, 'f', attrib={'name': 'verbalComponent'})
		f.set('fVal', '#ves' + str(count))

	# add level-3 data from entity structures
	count = 0
	for dic in ent:
		count += 1
		dact = etree.SubElement(diaml, 'dialogueAct')
		dact.set('{http://www.w3.org/XML/1998/namespace}id', 'da' + str(count))
		dact.set('target', '#' + dic['markable'])
		dact.set('sender', '#' + dic['sender'])
		dact.set('addressee', '#' + dic['addressee'])
		dact.set('dimension', dic['dimension'])
		dact.set('communicativeFunction', dic['communicativeFunction'])
		if dic['dependences'][0] != 'NA':
			dact.set('functionalDependence', '#' + str(dic['dependences'][0]).replace(',', ' '))
		if dic['dependences'][1] != 'NA':
			dact.set('feedbackDependence', '#' + str(dic['dependences'][1]).replace(',', ' '))
		if dic['qualifiers'][0] != 'NA':
			dact.set('certainty', dic['qualifiers'][0])
		if dic['qualifiers'][1] != 'NA':
			dact.set('conditionality', dic['qualifiers'][1])
		if dic['qualifiers'][2] != 'NA':
			dact.set('sentiment', dic['qualifiers'][2])
		if dic['other Ps'] != 'NA':
			dact.set('otherParticipant', '#' + str(dic['other Ps']).replace(' ', '').replace(',', ' #'))

	# add level-3 data from link structures
	for dic in link:
		rheto = etree.SubElement(diaml, 'rhetoricalLink')
		rheto.set('dact', '#' + dic['rhetoDact'])
		rheto.set('rhetoAntecedent', '#' +
				  str([x for x in dic['rhetoRelatum']]).replace("'", '').replace('[', '').replace(']', '')
				  .replace(',', ' #').strip())
		rheto.set('rhetoRel', dic['rel'])

	# write to file
	output_file = open(str(name) + '_DiAML-XML.xml', 'wb')
	tree.write(output_file)

	# Use (i.e. uncomment) below print statement to 'pretty print' the XML
	# print(etree.tostring(tree, pretty_print=True, encoding='unicode'))
	return "The conversion has been successfully executed. The annotation file '" + name \
		   + "_DiAML-XML.xlsx' has been created."


# MAIN FUNCTION
# This function is called whenever this .py file/script is executed/run.
# First, the user chooses one of six conversions by entering 1, 2, 3, 4, 5, or 6.
# Secondly, the user enters the file path to the annotation file.
# Thirdly - depending on which conversion - the user is asked for the
# paths to the level-1 file, level-2 file, and/or name of the dialogue.
# Delete/comment out try/except (and un-indent the body of the main function)
# if you'd like to know the exact source of a potential error.

def main():
		try:
			conversion = int(input("Indicate which conversion you want to execute.\n"
								   "Press 1 for input: 'DiAML-MultiTab' and output: 'DiAML-TabSW'.\n"
								   "Press 2 for input: 'DiAML-MultiTab' and output: 'DiAML-XML'.\n"
								   "Press 3 for input: 'DiAML-TabSW' and output: 'DiAML-MultiTab'.\n"
								   "Press 4 for input: 'DiAML-TabSW' and output: 'DiAML-XML'.\n"
								   "Press 5 for input: 'DiAML-XML' and output: 'DiAML-MultiTab'.\n"
								   "Press 6 for input: 'DiAML-XML' and output: 'DiAML-TabSW'."))

			conversions = [1, 2, 3, 4, 5, 6]
			if conversion in conversions:

				inp_name = input("Enter the path to the location of your input annotation file. "
								 "Do not forget to add the file extension.\n"
								 "For instance: C:/user/docs/dialogue1_MultiTab.xlsx.\n")

				if conversion == 1:  # input = MultiTab

					# Paths to level-1 and level-2 files are not needed
					# for current conversion of MultiTab to TabSW.

					# Name dialogue
					name = input("Please enter the name of the dialogue.")

					# MultiTab to Abstract syntax
					entity_v1 = mu_init_entity(inp_name)
					entity_v2 = mu_remove_dims(entity_v1)
					entity_v3 = mu_func_dim(entity_v2)
					entity_v4 = mu_init_q_dep(entity_v3)
					entity_v5 = mu_qualifiers(entity_v4)
					entity_v6 = mu_dependences(entity_v5)
					entity_v7 = mu_clean_entity(entity_v6)
					link_v1 = mu_retrieve_link_data(inp_name)
					link_v2 = mu_link_structures(link_v1)
					abs_syn = mu_entity_link(entity_v7, link_v2)  # abstract syntax
					# print(abs_syn)

					# Abstract syntax to TabSW
					fs_tt_com = mu_zipped_list(inp_name)[1]  # FS text, Turn transcription, Comments.
					sw_v1 = extract_ent_sw(abs_syn)
					sw_v2 = extract_link_sw(sw_v1)
					sw_v3 = dact_col_sw(sw_v2)
					sw_v4 = fs_tt_com_sw(sw_v3, fs_tt_com, 1)  # 1 because input = MultiTab
					sw_v5 = to_tabsw(sw_v4, name)
					print(sw_v5)

				elif conversion == 2:  # input = MultiTab

					# Paths to level-1 and level-2 files.
					inp_lvl_1 = input(
						"Add the path to the level-1 (primary data/tokenization) file. "
						"Do not forget to add the file extension.\n"
						"For instance: C:/USER/DOCS/DIALOGUE1_LVL1.TXT")
					inp_lvl_2 = input(
						"Add the path to the level-2 (functional segment) file. "
						"Do not forget to add the file extension.\n"
						"For instance: C:/USER/DOCS/DIALOGUE1_LVL2.TXT")

					# Name dialogue
					name = input("Please enter the name of the dialogue.")

					# Parses level-1 and level-2 files.
					# Required for conversion to DiAML-XML.
					level_one = inp_level_one(inp_lvl_1)
					level_two = inp_level_two(inp_lvl_2)

					# MultiTab to Abstract Syntax
					entity_v1 = mu_init_entity(inp_name)
					entity_v2 = mu_remove_dims(entity_v1)
					entity_v3 = mu_func_dim(entity_v2)
					entity_v4 = mu_init_q_dep(entity_v3)
					entity_v5 = mu_qualifiers(entity_v4)
					entity_v6 = mu_dependences(entity_v5)
					entity_v7 = mu_clean_entity(entity_v6)
					link_v1 = mu_retrieve_link_data(inp_name)
					link_v2 = mu_link_structures(link_v1)
					abs_syn = mu_entity_link(entity_v7, link_v2)  # abstract syntax
					# print(abs_syn)

					# Abstract Syntax to DiAML-XML
					# level_one to create XML list representation of the words
					# level_two to create XML representation of all functional segments
					# abs_syn to create <dialogueAct> and <rhetoricalLink> XML elements.
					xml_v1 = create_xml(level_one, level_two, abs_syn, name)
					print(xml_v1)

				elif conversion == 3:  # input = TabSW

					# Name dialogue
					name = input("Please enter the name of the dialogue.")

					# TabSW to Abstract Syntax
					entity_v1 = sw_init_entity(inp_name)
					entity_v2 = sw_func_dim(entity_v1)
					entity_v3 = sw_init_q_dep(entity_v2)
					entity_v4 = sw_qualifiers(entity_v3)
					entity_v5 = sw_dependences(entity_v4)
					link_v1 = sw_retrieve_link_data(entity_v5)
					link_v2 = sw_link_structures(link_v1)
					entity_v6 = sw_clean_entity(entity_v5)
					abs_syn = sw_entity_link(entity_v6, link_v2)  # abstract syntax
					# print(abs_syn)

					# Abstract Syntax to MultiTab
					fs_tt_com = sw_zipped_list(inp_name)[1]
					mu_v1 = extract_ent_mu(abs_syn)
					mu_v2 = extract_link_mu(mu_v1)
					mu_v3 = dim_cols_mu(mu_v2)
					mu_v4 = fs_tt_com_mu(mu_v3, fs_tt_com, 1)  # 1 because input is TabSW
					mu_v5 = to_multitab(mu_v4, name)
					print(mu_v5)

				elif conversion == 4:  # input = TabSW

					# Paths to level-1 and level-2 file
					inp_lvl_1 = input("Add the path to the level-1 (primary data/tokenization) file. "
									  "Do not forget to add the file extension.\n"
									  "For instance: C:/USER/DOCS/DIALOGUE1_LVL1.TXT")
					inp_lvl_2 = input("Add the path to the level-2 (functional segment/segmentation) file. "
									  "Do not forget to add the file extension.\n"
									  "For instance: C:/USER/DOCS/DIALOGUE1_LVL2.TXT")

					# Name dialogue
					name = input("Please enter the name of the dialogue.")

					# Parses level-1 and level-2 files
					level_one = inp_level_one(inp_lvl_1)
					level_two = inp_level_two(inp_lvl_2)

					# TabSW to Abstract Syntax
					entity_v1 = sw_init_entity(inp_name)
					entity_v2 = sw_func_dim(entity_v1)
					entity_v3 = sw_init_q_dep(entity_v2)
					entity_v4 = sw_qualifiers(entity_v3)
					entity_v5 = sw_dependences(entity_v4)
					link_v1 = sw_retrieve_link_data(entity_v5)
					link_v2 = sw_link_structures(link_v1)
					entity_v6 = sw_clean_entity(entity_v5)
					abs_syn = sw_entity_link(entity_v6, link_v2)  # abstract syntax
					# print(abs_syn)

					# Abstract Syntax to DiAML-XML
					# level_one to create XML list representation of the words
					# level_two to create XML representation of all functional segments
					# abs_syn to create <dialogueAct> and <rhetoricalLink> XML elements.
					xml_v1 = create_xml(level_one, level_two, abs_syn, name)
					print(xml_v1)

				elif conversion == 5:  # input = XML

					# Name dialogue
					name = input("Please enter the name of the dialogue.")

					# XML to Abstract Syntax
					all_data = xml_data(inp_name)
					fs_ws = fs_words(inp_name, name)
					entity_v1 = xml_entity_data(all_data)
					link_v1 = xml_link_data(all_data)
					abs_syn = xml_entity_link(entity_v1, link_v1)  # abstract syntax
					print("The tokenization file '", name + "_tokenization.txt' and the segmentation file '",
						  name + "_segmentation.txt' have been created.")
					# print(abs_syn)

					# Abstract Syntax to MultiTab
					# fs_tt_com = 'empty'
					mu_v1 = extract_ent_mu(abs_syn)
					mu_v2 = extract_link_mu(mu_v1)
					mu_v3 = dim_cols_mu(mu_v2)  # 2 because input is not TabSW
					mu_v4 = fs_tt_com_mu(mu_v3, fs_ws, 2)
					mu_v4 = to_multitab(mu_v4, name)
					print(mu_v4)

				elif conversion == 6:  # input = XML

					# Name dialogue
					name = input("Please enter the name of the dialogue.")

					# XML to Abstract Syntax
					all_data = xml_data(inp_name)
					fs_ws = fs_words(inp_name, name)
					entity_v1 = xml_entity_data(all_data)
					link_v1 = xml_link_data(all_data)

					abs_syn = xml_entity_link(entity_v1, link_v1)  # abstract syntax
					print("The tokenization file", name + "_tokenization.txt and the segmentation file",
						  name + "_segmentation.txt have been created")
					# print(abs_syn)

					# Abstract Syntax to TabSW
					# fs_tt_com = 'empty'
					sw_v1 = extract_ent_sw(abs_syn)
					sw_v2 = extract_link_sw(sw_v1)
					sw_v3 = dact_col_sw(sw_v2)  # 2 because input is not MultiTab (FS, TT, Comments columns).
					sw_v4 = fs_tt_com_sw(sw_v3, fs_ws, 2)
					sw_v5 = to_tabsw(sw_v4, name)
					print(sw_v5)
		except FileNotFoundError as fnf:
			print(fnf)
			print("The file(s) could not be found. See the above error message.\n"
				  "Make sure to add the correct filepath. Do not forget to add the file extension (e.g. .xlsx, .xml, or .txt)")
		except ET.ParseError as etp:
			print(etp)
			print("There seems to be an error in the DiAML-XML annotation. See the above error message.")
		except (KeyError, ValueError) as kv:
			print(kv)
			print("There seems to be an error in (one of) the file(s).")
		except xlrd.biffh.XLRDError:
			print("There seems to be an error in the Excel annotation file.")
		except:
			print('Oops! Something went wrong. Make sure the file(s) meet(s) all conditions.')

if __name__ == '__main__':
	main()
