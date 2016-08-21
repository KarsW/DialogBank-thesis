import pandas as pd
import numpy as np
from operator import itemgetter
import xml.etree.ElementTree as ET
import re
from collections import OrderedDict


# GENERAL FUNCTIONS
# These functions are not part of on of the encodings/decoding,
# they are included in one or more other functions.

# Returns string between two characters/symbols.
def between_sym(s, start, end):
	return (s.split(start))[1].split(end)[0]


# Function used to human/natural sort the word ids/tokens in the XML to abs_syn decoding
# before construction of the FS text column.
def nat_sort_one(li, k):  # list, key
	change = lambda text: int(text) if text.isdigit() else text  # integer or not
	chunks = lambda item: [change(c) for c in re.split('([0-9]+)', k(item))]  # chunks the strings
	return sorted(li, key=chunks)


# Removes 'na' and various other characters/symbols from rows in DataFrame.
def replace_na(r):
	new_r = str(r).replace('NA;', '').replace('NA ', ' ').replace(';NA', '').replace('  ', ' ')
	return new_r


# DiAML-XML
# Output format is MultiTab or TabSW. Therefore, a level-1 tokenization file
# should be constructed from the DiAML-XML file. DBOX and Map Task annotations
# contain words, timestamps, and word tokens that include specification of sender:
# e.g. wp11 (word 1 from speaker 1). Timestamps, original word tokens, words,
# and new word tokens (w1, w2, ...) are added to the tokenization file as follows:
# "new word token (original word token): word    start time    end time"
def level_one(file, name):
	default_uri = '{http://www.w3.org/XML/1998/namespace}id'
	tree = ET.parse(file)
	id_start_end = []  	# one list with dicts with 'end': 'end#', and 'start': 'start#' key, value pairs.
	words = []  		# one list that contains all words
	for e in tree.iter(tag="w"):
		attr = e.get(default_uri)
		if default_uri in e.attrib:
			del e.attrib[default_uri]
			e.set('word_id', attr)
		id_start_end.append(e.attrib)
		words.append(e.text)

	# add for each dict corresponding actual word; with 'id', word' as key,value pair.
	# changes id_start_end
	for w1, w2 in zip(id_start_end, words):
		w1['word'] = w2

	id_start_end2 = []  # list with lists that consist of word, word ID, and corresponding start and end #'s.
	for x in id_start_end:
		y = [x['word'], x['word_id'], x['start'], x['end']]
		id_start_end2.append(y)

	id_and_interval = []  # list with dictionaries that consist of 'word_id' and 'time' (and 'since') keys/values.
	for t in tree.iter(tag='when'):
		attr = t.get(default_uri)
		if default_uri in t.attrib:
			del t.attrib[default_uri]
			t.set('word_id', attr)
		id_and_interval.append(t.attrib)

	del id_and_interval[0]  # deletes "<when xml:id="TW0" absolute="00:00:00"/>" from id_interval

	time_dict = {}  # dictionary with 'start#': 'time' and 'end#': 'time' keys/values.
	for x in id_and_interval:
		time_dict.update({x['word_id']: x['interval']})

	words_list = []  # final list including word, word id, and start and end times.
	for word, word_id, start, end in id_start_end2:
		words_list.append([word, word_id, float(time_dict[start[1:]]), float(time_dict[end[1:]])])
	words_list.sort(key=itemgetter(2))  # sort on start time

	new_filename = name + "_tokenization.txt"  # creates level-1 file
	with open(new_filename, 'w') as file:
		count = 0
		for w, w_id, s, e in words_list:  # w = word, s = start time, e = end time, w_id = original word token/id
			count += 1
			lines = str('w{} ({}): {!s:<15} start: {!s:<15} end: {!s:<15}\n'.format(count, w_id, w, s, e))
			file.write(lines)
	file.close()
	return "The tokenization file '" + new_filename + "' has been created."


# Following function creates level-2 segmentation file,
# and returns 'segments' and 'word_id' which are necessary to create Turn transcription
# and FS text columns later on.
def level_two(f, name):  # get spanGrp + span keys, values
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
		id_list.append(attr)  # word tokens/ids
		words.append(e.text)  # words
	word_id = dict(zip(id_list, words))  # Dictionary of all word ids and words: {wordid: word, ...}

	# Create level-2 file for MultiTab and TabSW formats.
	new_filename = name + "_segmentation.txt"
	with open(new_filename, 'w') as file:
		for i in range(len(word_nums)):
			file.write(str(fs_seg[i] + ': ' + ', '.join(map(repr, (word_nums[i]))).replace("'", '') + '\n'))
	file.close()
	print("The segmentation file '" + new_filename + "' has been created.")

	# return for later construction of 'FS text' and 'Turn transcription' column in output representations.
	return segments, word_id


# level-3
# Returns list with entity data and list with link data.
def xml_data(f):
	tree = ET.parse(f)
	root = tree.getroot()
	text = root.getchildren()[1]  	# <text> element
	div = text.getchildren()  		# <div> element
	entity_data = div[3]  			# 3d <div> element = dialogueAct and rhetoricalLink XML-elements
	default_uri = '{http://www.w3.org/XML/1998/namespace}id'
	link_list = []
	entity_list = []

	for dact in entity_data:
		attr = dact.get(default_uri)
		if default_uri in dact.attrib:
			del dact.attrib[default_uri]
			dact.set('entityID', attr)
		if 'dialogueAct' in dact.tag:  		# Adds data from 'dialogueAct' XML-elements to entity list
			entity_list.append(dact.attrib)
		else:
			link_list.append(dact.attrib)  	# Adds data from 'rhetoricalLink' XML-elements to link list

	for dic in entity_list:
		for k, v in dic.items():
			dic[k] = v.replace('#', '')  	# removes '#' symbols

	for dic in link_list:
		for k, v in dic.items():
			dic[k] = v.replace('#', '')		# removes '#' symbols
	return entity_list, link_list


def xml_entity_data(d, two): 	# two = output of 'level_two' function
	data = d					# d = all xml data (from previous function)
	entity_data = data[0]
	fs_segs = two[0]  			# [(fs1, [w1,w2,w3]),..)]
	word_id = two[1]  			# {'w32': 'papers', 'w6': 'get', ... }

	id_list = []  				# entityID
	sen_list = []  				# sender
	add_list = []  				# addressee
	op_list = []  				# other Ps
	target_list = []  			# functional segment/markable/target
	dim_list = []  				# dimension
	comm_func_list = []  		# communicative function
	cert_list = []  			# certainty
	cond_list = []  			# conditionality
	sent_list = []  			# sentiment
	fu_dep_list = [] 			# functional dependences
	fb_dep_list = []  			# feedback dependences

	# Fill above lists.
	for x in entity_data:
		id_list.append(x.get('entityID'))
		sen_list.append(x.get('sender'))
		if 'addressee' in entity_data[0]:
			add_list.append(x.get('addressee'))
		else:
			add_list.append(x.get('addresse'))
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
	id_list = ['NA' if v is None else v for v in id_list]
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

	keys = ['entityID', 'sender', 'addressee', 'other Ps', 'markable', 'dimension', 'communicativeFunction']
	entity_data = list(zip(id_list, sen_list, add_list, op_list, target_list, dim_list, comm_func_list))
	entity_list = []

	# Creates dictionaries for entity structures with above keys.
	for i in range(len(entity_data)):
		dictionary = dict(zip(keys, entity_data[i]))
		entity_list.append(dictionary)

	# Adds 'qualifiers' and 'dependences' keys (values are lists) to dictionaries.
	for i in range(len(entity_list)):
		entity_list[i]['qualifiers'] = [cert_list[i], cond_list[i], sent_list[i]]
		entity_list[i]['dependences'] = [fu_dep_list[i], fb_dep_list[i]]

	# Adds 'words' key to each dictionary.
	# Value is dictionary with a functional segment's 'word IDs' as keys and 'words' as values.
	# ... 'words': {'w6': 'very', 'w7': 'nice'} ...
	for (k, v) in fs_segs:
		for c, dic in enumerate(entity_list):
			if k in dic.values():
				entity_list[c]['words'] = {p: word_id[p] for p in v}
	return entity_list


# Creates and returns list with link structure data in dictionaries.
def xml_link_data(d):
	data = d
	link_data = data[1]
	link_list = []
	for x in link_data:
		link_list.append(
			{'rel': x.get('rhetoRel'), 'rhetoRelatum': x.get('rhetoAntecedent').split(), 'rhetoDact': x.get('dact')})
	return link_list


# Returns abstract syntax: list of entity- and list of link structures.
# Adds start times to word ids by introducing 'times' key.
def xml_entity_link(ent, li, name):
	entity = ent
	link = li

	filename = name + '_tokenization.txt'
	with open(filename, 'r') as file:
		id_start = []
		for line in file:
			# (k, v) = line.rstrip('\n').split(':')
			start = line.split(':')[2].split()[0]
			w_id = between_sym(str(line.split(':')[0]), '(', ')')
			id_start.append({w_id: float(start)})
	file.close()

	# add 'times' key to entity list.
	for dic in entity:
		temp = dict(dic)
		for k, v in temp.items():
			if k == 'words':
				# dic['times'] = [(k2, v2) for x in id_start for k2, v2 in x.items() if k2 in v]
				dic['times'] = {k2: v2 for x in id_start for k2, v2 in x.items() if k2 in v}

	# Sort 'times' on values/start times. Changes value of 'times' keys. Now list of tuples (word id, start time).
	# 'times': [('wp12', 10.017),('wp18', 10.598)]
	# now, all the values of 'times' keys are sorted.
	for dic in entity:
		for k, v in dic.items():
			if k == 'times':
				values = sorted(v.items(), key=itemgetter(1))
				dic[k] = values

	# Now sort entire entity list on the start time of the first/earliest tuple (2nd element of tuple).
	ent_new = sorted(entity, key=lambda x: x['times'][0][1])

	# update 'words' key to list of tuples (like 'times'): (word id, word)
	# Needed for below sort of word ids.
	for dic in ent_new:
		temp = dict(dic)
		for k, v in temp.items():
			dic.pop('times', None)  # remove 'times' keys. Not needed anymore.
			if k == 'words':
				new_v = [(x, y) for x, y in v.items()]
				dic[k] = new_v

	# Finally, sort word id values of 'words' key(s).
	# Requires natural/human sort.
	# Allows correctly ordered implementation of words in a (FS text) column in TabSW or MultiTab DataFrame.
	for dic in entity:
		for k, v in dic.items():
			if k == 'words':
				new_v = nat_sort_one(v, itemgetter(0))
				dic[k] = new_v
	return ent_new, link


# TO MULTITAB
# Returns entity df and link df.
# Also, extracts and (re-)arranges data elements in df: qualifiers, dependences.
def extract_ent_mu(abs_syn):
	pd.options.display.max_colwidth = 500
	df_ent = pd.DataFrame(abs_syn[0])
	link = abs_syn[1]
	df_link = pd.DataFrame(link)

	for c, li in enumerate(df_ent['qualifiers']):  # add qualifiers first
		cert = li[0]
		cond = li[1]
		sent = li[2]
		if cert != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + '[' + str(cert) + ']'
		elif cond != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + '[' + str(cond) + ']'
		elif sent != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + '[' + str(sent) + ']'

	for c, li in enumerate(df_ent['dependences']):  # now add dependences
		fu = li[0]
		fe = li[1]
		if fu != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + '(Fu:' + str(fu) + ')'
		elif fe != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + '(Fe:' + str(fe) + ')'

	# clean 'words' (FS text) column by removing all characters (except the words).
	for c, row in enumerate(df_ent['words']):
		df_ent.loc[c, 'words'] = str(row).replace('[', '').replace(']', '') \
			.replace(',', '').replace("'", '').replace('"', '')

	df_ent.drop(['dependences', 'qualifiers'], inplace=True, axis=1)  # drop 'dependences' and 'qualifiers' columns.
	df_ent['Turn transcription'] = np.nan
	df_ent = df_ent[
		['markable', 'sender', 'addressee', 'other Ps', 'Turn transcription', 'words', 'entityID', 'dimension',
		 'communicativeFunction']]  # reorder.
	return df_ent, df_link  # return df with entity structure data and df with link structure data


# Returns df containing entity data + link data.
# Link structure data is added to communicative functions.
# Link data columns are then dropped from the DataFame.
# Also, empty dimension columns are added.
def extract_link_mu(mu_v1):
	pd.options.display.max_colwidth = 500
	df_ent = mu_v1[0]
	df_link = mu_v1[1]
	df_ent_link = df_ent.merge(df_link, how='left', left_on='entityID', right_on='rhetoDact', sort=False)

	# (re)-arrange link data
	for c, relation in enumerate(df_ent_link['rel']):  # relation
		if relation is not np.nan:
			df_ent_link.loc[c, 'communicativeFunction'] = df_ent_link.loc[c, 'communicativeFunction'] + ' {' + str(
				relation) + ' '

	for c, relatum in enumerate(df_ent_link['rhetoRelatum']):  # relatum
		if relatum is not np.nan:
			df_ent_link.loc[c, 'communicativeFunction'] += (str(relatum) + '}').replace('[', '').replace(']','')\
				.replace("'", '')

	df_ent_link.drop(['rel', 'rhetoDact', 'rhetoRelatum'], inplace=True, axis=1)

	df_ent_link['Task'] = np.nan
	df_ent_link['autoFeedback'] = np.nan
	df_ent_link['alloFeedback'] = np.nan
	df_ent_link['turnManagement'] = np.nan
	df_ent_link['timeManagement'] = np.nan
	df_ent_link['ownCommunicationManagement'] = np.nan
	df_ent_link['partnerCommunicationManagement'] = np.nan
	df_ent_link['discourseStructuring'] = np.nan
	df_ent_link['socialObligationsManagement'] = np.nan
	df_ent_link['other'] = np.nan
	df_ent_link['Comments'] = np.nan
	return df_ent_link  # returns df containing qualifiers, dependences, and rhetorical relations.


# The empty dimension columns are now filled with data:
# entity structure identifiers plus values of the 'communicativeFunction' column.
# dimension, communicativeFunction and entityID columns are dropped, as their data
# is now in the ten dimension columns.
# If more than one functional segment name for the same primary data collection then the segments are placed
# in a similar row/cell separated by a ';'.
# Finally, the FS text (+ Turn transcription, Comments) column is created/cleaned,
# and unwanted characters are deleted from the dimension columns.
def dim_cols_mu(mu_v2):
	pd.options.display.max_colwidth = 500
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
		elif el.lower() == 'socialobligationsmanagement' or el.lower() == 'socialobligationmanagement':
			df1.loc[c, 'socialObligationsManagement'] = df1.loc[c, 'entityID'] + ': ' + \
														df1.loc[c, 'communicativeFunction']
		else:
			df1.loc[c, 'other'] = df1.loc[c, 'entityID'] + ':' + str(el) + ':' + df1.loc[c, 'communicativeFunction']

	df1.drop(['dimension', 'communicativeFunction', 'entityID'], inplace=True, axis=1)

	df1['FS text'] = np.nan
	df1['Turn transcription'] = np.nan
	df1['Comments'] = np.nan
	df2 = df1.fillna('NA')

	df3 = df2.groupby(['sender', 'addressee', 'other Ps', 'words', 'Turn transcription', 'Comments'], sort=False).agg(
		';'.join).reset_index()

	df4 = df3[
		['markable', 'sender', 'addressee', 'other Ps', 'Turn transcription', 'FS text', 'Task', 'autoFeedback',
		 'alloFeedback', 'turnManagement', 'timeManagement', 'ownCommunicationManagement',
		 'partnerCommunicationManagement', 'discourseStructuring', 'socialObligationsManagement', 'other', 'Comments',
		 'words']]

	# Clean 'words' column rows and subsequently add rows to 'FS text' column
	for c, row in enumerate(df4['words']):
		spl = row.split()
		for el in spl:
			if ')' in el:
				new_el = str(el).replace(')', '')
				df4.loc[c, 'FS text'] += ' ' + new_el + ' '

	# remove unwanted characters from 'FS text' and all dimension columns.
	for c, row in enumerate(df4['FS text']):
		df4.loc[c, 'FS text'] = replace_na(row)
	for c, row in enumerate(df4['Task']):
		df4.loc[c, 'Task'] = replace_na(row)
	for c, row in enumerate(df4['autoFeedback']):
		df4.loc[c, 'autoFeedback'] = replace_na(row)
	for c, row in enumerate(df4['alloFeedback']):
		df4.loc[c, 'alloFeedback'] = replace_na(row)
	for c, row in enumerate(df4['turnManagement']):
		df4.loc[c, 'turnManagement'] = replace_na(row)
	for c, row in enumerate(df4['timeManagement']):
		df4.loc[c, 'timeManagement'] = replace_na(row)
	for c, row in enumerate(df4['ownCommunicationManagement']):
		df4.loc[c, 'ownCommunicationManagement'] = replace_na(row)
	for c, row in enumerate(df4['partnerCommunicationManagement']):
		df4.loc[c, 'partnerCommunicationManagement'] = replace_na(row)
	for c, row in enumerate(df4['discourseStructuring']):
		df4.loc[c, 'discourseStructuring'] = replace_na(row)
	for c, row in enumerate(df4['socialObligationsManagement']):
		df4.loc[c, 'socialObligationsManagement'] = replace_na(row)
	for c, row in enumerate(df4['other']):
		df4.loc[c, 'other'] = replace_na(row)

	df5 = df4.replace('NA', np.nan)

	df_final = df5[
		['markable', 'sender', 'addressee', 'other Ps', 'Turn transcription', 'FS text', 'Task', 'autoFeedback',
		 'alloFeedback', 'turnManagement', 'timeManagement', 'ownCommunicationManagement',
		 'partnerCommunicationManagement', 'discourseStructuring', 'socialObligationsManagement', 'other', 'Comments']]
	df_final.columns = ['Markables', 'Sender', 'Addressee', 'other Ps', 'Turn transcription', 'FS text', 'Task',
						'autoFeedback', 'alloFeedback', 'turnManagement', 'timeManagement',
						'ownCommunicationManagement', 'partnerCommunicationManagement', 'discourseStructuring',
						'socialObligationsManagement', 'other', 'Comments']
	return df_final


# Function used to correctly fill the Turn transcription column.
def tt_col_mu(mu_v3, cols):
	df = mu_v3
	fs = cols[0]
	words = cols[1]
	words2 = OrderedDict(words)

	fs_list = []  				 	# list that stores first functional segment from each row.
	for row in df['Markables']:
		if ';' not in row:
			fs_list.append(row)
		elif ';' in row:
			f = row.split(';')
			fs_list.append(f[0])
	sen_list = []					# list that stores for each row the sender.
	for row in df['Sender']:
		sen_list.append(row)

	tt_list = []
	for el in fs_list:			# functional segments
		for f in fs:  			# (fs1, [w1,w2,w3]
			if el == f[0]:		# add words to correct FS text 'row'
				items = [x for x in f[1]]
				tt_list.append([el, [x for x in items if x in words2.keys()]])

	for c, el in enumerate(sen_list):
		tt_list[c].append(el)   # tt_list is now [[fs, [word num(s)], sender], []]

	tt_two = []								# iterates over tt_list. Keeps track of who is the current sender.
	sender = None							# If sender/turn does not change adds words to the existing list of
	for li in tt_list:						# words. If sender changes create a new list and add the words
		if li[2] != sender:					# to that list, etc. To each list the first functional segment id
			func_seg = li[0]				# of that turn is added (to know where in the df to add the words).
			tt_two.append({func_seg: []})	# This way a Turn transcription list is created.
		tt_two[-1][func_seg].extend(li[1])
		sender = li[2]

	df2 = df.fillna('NA')

	for f in tt_two:									 # Iterates over above tt_two. Checks segment id, adds to that
		for k, v in f.items():  # k = fs1, v = [w1, w2]	 # Turn transcription row in the df, belonging to that segment,
			for w in v:									 # the words corresponding with the word ids in 'words' dictionary.
				for c, el in enumerate(fs_list):
					if el == k:
						df2.loc[c, 'Turn transcription'] += str(words.get(w)) + ' '

	for c, el in enumerate(df2['Turn transcription']):  # delete 'NA' in front of content in FS text rows.
		if str(el).startswith('NA'):
			df2.loc[c, 'Turn transcription'] = str(el)[2:]

	df_final = df2.replace('NA', np.nan)
	return df_final


# Creates .xlsx file based on above DataFrame.
# Formats the file. Start row, Title, Bold, Wrap etc.
def to_multitab(mu_tt, name):
	final_df = mu_tt

	writer = pd.ExcelWriter(str(name) + '_DiAML-MultiTab.xlsx', engine='xlsxwriter')
	final_df.to_excel(writer, sheet_name='Sheet1', index=False, startrow=3)

	workbook = writer.book
	worksheet = writer.sheets['Sheet1']

	bold = workbook.add_format({'bold': True})
	wrap = workbook.add_format({'text_wrap': True})

	worksheet.write('A1', 'Dialogue "{}", Gold standard ISO 24617-2 annotation represented in DiAML-MultiTab format.'
					.format(name), bold)
	worksheet.set_row(3, 30, wrap)
	worksheet.set_column(0, 0, 20, wrap)
	worksheet.set_column(1, 2, 10, wrap)
	worksheet.set_column(4, 5, 25, wrap)
	worksheet.set_column(6, 17, 20, wrap)
	writer.close()
	return "The conversion has been successfully executed. The annotation file '" + name + \
		   "_DiAML-MultiTab.xlsx' has been created.\n"


# TO TABSW
# Returns entity DataFrame and link DataFrame.
# Qualifiers and dependences are added to rows in communicativeFunction column,
# qualifiers and dependences columns are dropped from entity df.
# Words/FS text column is cleaned. Empty Turn transcription column is created.
def extract_ent_sw(abs_syn):
	ent = abs_syn[0]
	df_ent = pd.DataFrame(ent)
	link = abs_syn[1]
	df_link = pd.DataFrame(link)

	for c, li in enumerate(df_ent['qualifiers']):  # Adds qualifiers to communicativeFunction column.
		cert = li[0]
		cond = li[1]
		sent = li[2]
		if cert != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + ' [' + str(cert) + '] '
		elif cond != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + ' [' + str(cond) + '] '
		elif sent != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + ' [' + str(sent) + '] '

	for c, li in enumerate(df_ent['dependences']):  # Adds dependences to communicativeFunction column.
		fu = li[0]
		fe = li[1]
		if fu != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + ' (Fu:' + str(fu) + ')'
		elif fe != 'NA':
			df_ent.loc[c, 'communicativeFunction'] = df_ent.loc[c, 'communicativeFunction'] + ' (Fe:' + str(fe) + ')'

	for c, row in enumerate(df_ent['words']):
		df_ent.loc[c, 'words'] = str(row).replace('[', '').replace(']', '') \
			.replace(',', '').replace("'", '').replace('"', '')

	df_ent.drop(['dependences', 'qualifiers'], inplace=True, axis=1)  # Drops 'dependences' and 'qualifiers' columns.
	df_ent['Turn transcription'] = np.nan
	df_ent = df_ent[['markable', 'sender', 'addressee', 'other Ps', 'Turn transcription', 'words', 'entityID',
					 'dimension', 'communicativeFunction']]  # reorder.
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
	for c, relation in enumerate(df_ent_link['rel']):  # rhetorical relation
		if relation is not np.nan:
			df_ent_link.loc[c, 'communicativeFunction'] = df_ent_link.loc[c, 'communicativeFunction'] + ' {' + str(
				relation) + ' '

	for c, relatum in enumerate(df_ent_link['rhetoRelatum']):  # relatum
		if relatum is not np.nan:
			df_ent_link.loc[c, 'communicativeFunction'] += (str(relatum) + '}').replace('[', '').replace(']', '') \
				.replace("'", '')

	df_ent_link.drop(['rel', 'rhetoDact', 'rhetoRelatum'], inplace=True, axis=1)
	df_ent_link['Dialogue acts'] = np.nan
	return df_ent_link  # returns df containing qualifiers, dependences, and rhetorical relations.


# Adds dimension abbreviation and communicativeFunction value to (each row in) the 'Dialogue acts' column.
# The dimension and communicativeFunction columns are dropped from the df.
# Also, identical markables (+ their entityIDs and Dialogue acts) are placed
# in same cells/row and FS text column is filled.
def dact_col_sw(sw_v2):
	pd.options.display.max_colwidth = 500
	df1 = sw_v2
	dim = df1['dimension']
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
		elif el.lower() == 'socialobligationsmanagement' or el.lower() == 'socialobligationmanagement':
			df1.loc[c, 'Dialogue acts'] = 'SOM:' + df1.loc[c, 'communicativeFunction']
		else:
			df1.loc[c, 'Dialogue acts'] = str(el) + ':' + df1.loc[c, 'communicativeFunction']

	df1.drop(['dimension', 'communicativeFunction'], inplace=True, axis=1)

	df1['Turn transcription'] = np.nan
	df1['Comments'] = np.nan

	df2 = df1.fillna('NA')

	df3 = df2[
		['markable', 'entityID', 'Dialogue acts', 'sender', 'addressee', 'other Ps', 'words',
		 'Turn transcription', 'Comments']]

	# if rows in below columns are identical 'compress' these rows.
	# Rows (cell content) in 'markable', 'entityID' and 'Dialogue acts' columns are placed in one row separated by a ';'.
	df4 = df3.groupby(['sender', 'addressee', 'other Ps', 'words', 'Turn transcription', 'Comments'],
					  sort=False).agg('; '.join).reset_index()

	# Clean 'words' column and create column 'FS text'.
	df4['FS text'] = 'NA'
	for c, row in enumerate(df4['words']):
		spl = row.split()
		for el in spl:
			if ')' in el:
				new_el = str(el).replace(')', '')
				df4.loc[c, 'FS text'] += ' ' + new_el + ' '

	for c, row in enumerate(df4['FS text']):
		df4.loc[c, 'FS text'] = replace_na(row)

	df4.columns = ['Sender', 'Addressee', 'other Ps', 'words', 'Turn transcription', 'Comments', 'Markables',
				   'Da-ID', 'Dialogue acts', 'FS text']

	df4.drop(['words'], inplace=True, axis=1)

	df5 = df4[
		['Markables', 'Da-ID', 'Dialogue acts', 'Sender', 'Addressee', 'other Ps', 'FS text',
		 'Turn transcription', 'Comments']]
	df_final = df5.replace('NA', np.nan)
	return df_final


# Turn transcription column is correctly filled.
def tt_col_sw(sw_v3, cols):
	df = sw_v3
	fs = cols[0]
	words = cols[1]
	words2 = OrderedDict(words)

	fs_list = []  					# list that stores first functional segment from each row.
	for row in df['Markables']:
		if ';' not in row:
			fs_list.append(row)
		elif ';' in row:
			f = row.split(';')
			fs_list.append(f[0])
	sen_list = []					# list that stores for each row the sender.
	for row in df['Sender']:
		sen_list.append(row)

	tt_list = []
	for el in fs_list:			# functional segments
		for f in fs:  			# (fs1, [w1,w2,w3]
			if el == f[0]:		# add words to correct FS text 'row'
				items = [x for x in f[1]]
				tt_list.append([el, [x for x in items if x in words2.keys()]])

	for c, el in enumerate(sen_list):
		tt_list[c].append(el)	# tt_list is now [[fs, [word num(s)], sender], []]

	tt_two = []								# iterates over tt_list. Keeps tracker of who is the current sender.
	sender = None							# if sender/turn does not change adds words to the existing list of
	for li in tt_list:						# words. If sender changes create a new list and add the words
		if li[2] != sender:					# to that list, etc. To each list the first functional segment id
			func_seg = li[0]				# of that turn is added (to know where in the df to add the words).
			tt_two.append({func_seg: []})   # This way a Turn transcription list is created.
		tt_two[-1][func_seg].extend(li[1])
		sender = li[2]

	df2 = df.fillna('NA')

	for f in tt_two:										# iterates over above tt_two. Checks segment id, adds to that
		for k, v in f.items():  # k = fs1, v = [w1, w2]		# Turn transcription row in the df, belonging to that segment,
			for w in v:										# the words corresponding with the word ids in 'words' dictionary.
				for c, el in enumerate(fs_list):
					if el == k:
						df2.loc[c, 'Turn transcription'] += str(words.get(w)) + ' '

	for c, el in enumerate(df2['Turn transcription']):  # delete 'NA' in front of content in FS text rows.
		if str(el).startswith('NA'):
			df2.loc[c, 'Turn transcription'] = str(el)[2:]

	df_final = df2.replace('NA', np.nan)
	return df_final


# Creates .xlsx file based on above DataFrame.
# Formats the file. Start row, title, bold, wrap etc.
def to_tabsw(sw_tt, name):
	final_df = sw_tt

	writer = pd.ExcelWriter(str(name) + '_DiAML-TabSW.xlsx', engine='xlsxwriter')
	final_df.to_excel(writer, sheet_name='Sheet1', index=False, startrow=3)

	workbook = writer.book
	worksheet = writer.sheets['Sheet1']

	bold = workbook.add_format({'bold': True})
	wrap = workbook.add_format({'text_wrap': True})

	worksheet.write('A1', 'Dialogue "{}", Gold standard ISO 24617-2 annotation represented in DiAML-TabSW format.'
					.format(name), bold)
	worksheet.set_row(3, 30, wrap)
	worksheet.set_column(0, 0, 20, wrap)
	worksheet.set_column(1, 1, 20, wrap)
	worksheet.set_column(2, 2, 30, wrap)
	worksheet.set_column(6, 7, 30, wrap)
	writer.close()
	return "The conversion has been successfully executed. The annotation file '" + name + \
		   "_DiAML-TabSW.xlsx' has been created.\n"


# MAIN FUNCTION
# This function is called whenever this .py file/script is executed/run.
# First, the user enters the file path to the DiAML-XML annotation file.
# Secondly, the user enters the name of the dialogue.
# Finally, the user chooses one of two conversions by entering 1, or 2.
# 1 = to DiAML-MultiTab, 2 = to DiAML-TabSW.
# Delete/comment out try/except (and un-indent the body of the main function)
# if you'd like to know the exact source of a potential error.
def main():
	try:
		file = input("Enter the  path to the DiAML-XML file. Do not forget to add the file extension.")
		name = input("Enter the name of the dialogue.")
		conversion = int(input("Indicate which conversion you want to execute.\n"
							   "Press 1 for input: 'DiAML-XML' and output: 'DiAML-MultiTab'.\n"
							   "Press 2 for input: 'DiAML-XML' and output: 'DiAML-TabSW'.\n"))

		if conversion == 1:
			level_1 = level_one(file, name)
			print(level_1)  # create level-1 tokenization file
			level_2 = level_two(file, name)

			all_data = xml_data(file)
			entity_v1 = xml_entity_data(all_data, level_2)
			link_v1 = xml_link_data(all_data)
			abs_syn = xml_entity_link(entity_v1, link_v1, name)
			# print(abs_syn)

			# Abstract syntax to MultiTab
			mu_v1 = extract_ent_mu(abs_syn)
			mu_v2 = extract_link_mu(mu_v1)
			mu_v3 = dim_cols_mu(mu_v2)
			mu_tt = tt_col_mu(mu_v3, level_2)  # create level-2 segmentation file
			mu_v4 = to_multitab(mu_tt, name)
			print(mu_v4)

		elif conversion == 2:
			level_1 = level_one(file, name)
			print(level_1)  # create level-1 tokenization file
			level_2 = level_two(file, name)

			all_data = xml_data(file)
			entity_v1 = xml_entity_data(all_data, level_2)
			link_v1 = xml_link_data(all_data)
			abs_syn = xml_entity_link(entity_v1, link_v1, name)
			# print(abs_syn)

			# Abstract syntax to MultiTab
			sw_v1 = extract_ent_sw(abs_syn)
			sw_v2 = extract_link_sw(sw_v1)
			sw_v3 = dact_col_sw(sw_v2)
			sw_tt = tt_col_sw(sw_v3, level_2)  # create level-2 segmentation file
			sw_v4 = to_tabsw(sw_tt, name)
			print(sw_v4)
	except FileNotFoundError as fnf:
		print (fnf)
		print("Your file could not be found. See the above error message.\n"
			  "Make sure to add the correct filepath. Do not forget to add the file extension (e.g. .diaml or .xml)")
	except ET.ParseError as etp:
		print(etp)
		print("There seems to be an error in the DiAML-XML annotation. See the above error message.")
	except (KeyError, ValueError):
		print("There seems to be an error in the DiAML-XML annotation.\n"
			  "Check the annotation for incorrect (or unwanted duplicate) XML elements, attributes, and values.")
	except:
		print('Oops! Something went wrong. Make sure the file(s) meet(s) all conditions.')


if __name__ == '__main__':
	main()
