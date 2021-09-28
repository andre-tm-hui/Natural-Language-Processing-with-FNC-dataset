from __future__ import division
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from nltk.corpus import stopwords
from scipy import sparse
from scipy.sparse import hstack
import nltk
import pandas as pd
import csv
import sys, os
import pickle
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers

import time

try:
	stopwords.words('english')
except:
	nltk.download('stopwords')

FIELDNAMES = ['Headline', 'Body ID', 'Stance']
LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
PUNCTUATION = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
RELATED = LABELS[0:3]

class Dataset():
	def __init__(self, name='train', path='fnc-1', size=1, clean=True):
		self.path = path

		if not os.path.isfile('%s/%s_all_%s.csv' % (path, name, 'processed' if clean else 'unprocessed')):
			print('Generating')
			bodies = name + '_bodies.csv'
			stances = name + '_stances.csv'
			self.articles = self.read(bodies)
			self.stances = self.read(stances)

			self.articles = pd.read_csv('%s/%s' % (path, bodies))
			self.stances = pd.read_csv('%s/%s' % (path, stances))

			pd.options.display.max_columns = None
			self.data = pd.merge(self.articles, self.stances, how='right', on='Body ID')
			for index, d in self.data.iterrows():
				self.data.at[index, 'Body ID'] = int(d['Body ID'])
				if clean:
					self.data.at[index, 'Headline'] = preprocess(d['Headline'])
					self.data.at[index, 'articleBody'] = preprocess(d['articleBody'])
			self.data.to_csv('%s/%s_all_%s.csv' % (path, name, 'processed' if clean else 'unprocessed'), index = False)
		
		self.data = self.read('%s_all_%s.csv' % (name, 'processed' if clean else 'unprocessed'))
		self.data = self.data[:int(len(self.data) * size)]
		all_headlines = [d['Headline'] for d in self.data]
		all_bodies = [d['articleBody'] for d in self.data]

		self.all_text = [' '.join([a,b]) for a,b in zip(all_headlines, all_bodies)]
		

	def read(self, filename):
		rows = []
		with open(self.path + '/' + filename, 'r', encoding='utf-8') as table:
			r = csv.DictReader(table)
			for line in r:
				rows.append(line)
		return rows


class LSTM_cls(nn.Module):
	def __init__(self, embedder_mode, hidden_dim, dropout, output_dim, device, bidirectional = False, layers = 1):
		super(LSTM_cls, self).__init__()
		self.device = device
		self.embedder = embedder(embedder_mode, self.device)
		if embedder_mode == 'bert':
			embedding_dim = 3072
		elif embedder_mode == 'tfidf':
			embedding_dim = 27992
		self.input_dim = 1000
		self.hidden_dim = hidden_dim
		self.lstm = nn.LSTM(self.input_dim, hidden_dim, layers, bidirectional = bidirectional, dropout = dropout)
		self.fc1 = nn.Linear(hidden_dim * 2, output_dim)
		self.fc2 = nn.Linear(hidden_dim, output_dim)
		self.pool = nn.AvgPool2d(3)
		self.dropout = nn.Dropout(dropout)
		self.act = nn.Softmax(dim=-1)

		self.embed_dim = 5 * hidden_dim * layers
		if bidirectional:
			self.embed_dim *= 2

		self.reduction = nn.Linear(embedding_dim, self.input_dim)

		self.classifier = nn.Sequential(
			nn.Linear(self.embed_dim, output_dim),
			#nn.ReLU(),
			#nn.Linear(self.embed_dim // 2, output_dim),
			nn.LogSoftmax(dim=-1)
			)

	def forward(self, headline, body):
		with torch.no_grad():
			h_embed, b_embed = self.embedder.forward(headline, body)
			h_embed, b_embed = self.reduction(h_embed), self.reduction(b_embed)
		#.view(h_embed.shape[1],h_embed.shape[0],h_embed.shape[2])
		h_output, (h_hidden, h_cell) = self.lstm(h_embed.view(h_embed.shape[1],h_embed.shape[0],h_embed.shape[2]))
		b_output, (b_hidden, b_cell) = self.lstm(b_embed.view(h_embed.shape[1],h_embed.shape[0],h_embed.shape[2]))

		# last hidden state is used as sequence/sentence vector
		h_hidden = torch.cat([h for h in h_hidden], -1)
		b_hidden = torch.cat([b for b in b_hidden], -1)

		# compute some similarities
		features = torch.cat((h_hidden, torch.abs(h_hidden-b_hidden), b_hidden, h_hidden * b_hidden, (h_hidden + b_hidden) / 2), 1)

		output = self.classifier(features)
		return output

class embedder():
	def __init__(self, mode, device):
		self.mode = mode
		self.device = device
		if mode == 'bert':
			self.embed = bert_embed
			self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
			self.model = transformers.BertModel.from_pretrained('bert-base-uncased').to(device)
		elif mode == 'tfidf':
			train_dataset = Dataset()
			test_dataset = Dataset(name='competition_test')
			vec = TfidfVectorizer(ngram_range=(1,1), max_df=0.8, min_df=2)
			vec.fit(train_dataset.all_text + test_dataset.all_text)
			self.vocab = vec.vocabulary_
			self.embed = tfidf_embed

	def forward(self, headline, body):
		if self.mode == 'tfidf':
			headline_embed = torch.from_numpy(self.embed(headline, self.vocab)).to(self.device).unsqueeze(0)
			body_embed = torch.from_numpy(self.embed(body, self.vocab)).to(self.device).unsqueeze(0)
		elif self.mode == 'bert':
			headline_embed = self.embed(headline, self.tokenizer, self.model, self.device)
			body_embed = self.embed(body, self.tokenizer, self.model, self.device)

		try:
			max_headline = nn.MaxPool1d(3)(headline_embed)
		except:
			headline_embed = headline_embed.unsqueeze(0)
			body_embed = body_embed.unsqueeze(0)
			max_headline = nn.MaxPool1d(3)(headline_embed)

		#max_headline = nn.MaxPool1d(3)(headline_embed)
		#mean_headline = nn.AvgPool1d(3)(headline_embed)
		#headline_embed = torch.cat((max_headline, mean_headline), dim=-1)

		#max_body = nn.MaxPool1d(3)(body_embed)
		#mean_body = nn.AvgPool1d(3)(body_embed)
		#body_embed = torch.cat((max_body, mean_headline), dim=-1)

		#embedding = torch.cat((headline_embed, body_embed), dim=-1)

		return headline_embed, body_embed

def tfidf_embed(sentence, vocab):
	vec = TfidfVectorizer(ngram_range=(1,1), max_df=0.8, min_df=2, vocabulary=vocab)
	try:
		tfidf = vec.fit_transform(sentence)
	except:
		tfidf = vec.fit_transfrom([sentence])
	return np.float32(tfidf.toarray())

def bert_embed(sentence, tokenizer, model, device):
	encoding = tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True)
	input_id, token_type_id, attention_mask = encoding['input_ids'], encoding['token_type_ids'], encoding['attention_mask']
	
	output = model(input_id.to(device), token_type_ids = token_type_id.to(device), attention_mask = attention_mask.to(device), output_hidden_states=True)
	hidden_states = output[2]
	bert = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]], dim=-1).squeeze()
	return bert

		
def preprocess(data):
	from nltk.tokenize import word_tokenize
	from nltk.stem.porter import PorterStemmer
	import string
	tokens = word_tokenize(data)
	tokens = [token.lower() for token in tokens]
	table = str.maketrans('', '', PUNCTUATION)
	tokens = [token.translate(table) for token in tokens]
	tokens = [token for token in tokens if token.isalpha()]
	stop_words = set(stopwords.words('english'))
	tokens = [token for token in tokens if not token in stop_words]
	porter = PorterStemmer()
	tokens = [porter.stem(token) for token in tokens]
	return ' '.join(tokens)

def cosine_sim(x,y):
	try:
		if type(x) is np.ndarray: x = x.reshape(1, -1) # get rid of the warning
		if type(y) is np.ndarray: y = y.reshape(1, -1)
		d = cosine_similarity(x, y)
		d = d[0][0]
	except:
		print(x)
		print(y)
		d = 0.
	return d

def euclidean_dist(x,y):
	try:
		if type(x) is np.ndarray: x = x.reshape(1, -1) # get rid of the warning
		if type(y) is np.ndarray: y = y.reshape(1, -1)
		d = euclidean_distances(x, y)
		d = d[0][0]
	except:
		print(x)
		print(y)
		d = 0.
	return d

def get_tfIdf(data, body_id):
	i = 0
	exists = False
	for item in data:
		if item['Body ID'] == str(body_id):
			exists = True
			break
		else:
			i += 1

	if not exists:
		error = 'ERROR: Invalid Body ID'
		raise FNCException(error)

	print('%d: %s' % (body_id, data[i]['Headline']))

	body_df = pd.DataFrame(body_tfIdf[i].T.todense(), index = body_vectorizer.get_feature_names(), columns = ['TF-IDF'])
	body_df = body_df.sort_values('TF-IDF', ascending=False)
	print(body_df.head(25))

	headline_df = pd.DataFrame(headline_tfIdf[i].T.todense(), index = headline_vectorizer.get_feature_names(), columns = ['TF-IDF'])
	headline_df = headline_df.sort_values('TF-IDF', ascending=False)
	print(headline_df.head(5))

	return body_df, headline_df


def classify_tfidf(mode, classifier = 'svm'):
	print('Configuration: %s, %s' % (mode, classifier))

	train_dataset = Dataset()
	test_dataset = Dataset(name='competition_test')
	print('Datasets loaded')
	vec = TfidfVectorizer(ngram_range=(1,3), max_df=0.8, min_df=2)
	vec.fit(train_dataset.all_text + test_dataset.all_text)
	vocab = vec.vocabulary_
	print('Vocabulary initialized')

	if classifier == 'randforest':
		model = RandomForestClassifier(n_estimators=1000, random_state=0)
	elif classifier == 'svm':
		model = SGDClassifier(loss='hinge', penalty='l2', random_state=42, max_iter=100, tol=None)
	elif classifier == 'lr':
		model = LogisticRegression(C=2, class_weight = 'balanced')

	if mode == 'sim':
		ext = 'npy'
		load = np.load
		save = np.save
	else:
		ext = 'npz'
		load = sparse.load_npz
		save = sparse.save_npz

	if os.path.isfile('features/tfidf_train_%s.%s' % (mode, ext)):
		X_train = load('features/tfidf_train_%s.%s' % (mode, ext))
		X_test = load('features/tfidf_test_%s.%s' % (mode, ext))
	else:
		if mode == 'combine':
			vecHB = TfidfVectorizer(ngram_range=(1,3), max_df=0.8, min_df=2, vocabulary=vocab)
			tfidfHB = vecHB.fit_transform([' '.join([d['Headline'], d['articleBody']]) for d in train_dataset.data])
			print('tfidfs for headlines + bodies generated ', tfidfHB.shape)
			test_tfidfHB = vecHB.transform([' '.join([d['Headline'], d['articleBody']]) for d in test_dataset.data])
			print('tfidfs for test headlines + bodies generated ', test_tfidfHB.shape)

			X_train = tfidfHB
			X_test = test_tfidfHB
		else:
			vecH = TfidfVectorizer(ngram_range=(1,3), max_df=0.8, min_df=2, vocabulary=vocab)
			tfidfH = vecH.fit_transform([d['Headline'] for d in train_dataset.data])
			print('tfidfs for headlines generated ', tfidfH.shape)
			test_tfidfH = vecH.transform([d['Headline'] for d in test_dataset.data])
			print('tfidfs for test headlines generated ', test_tfidfH.shape)

			vecB = TfidfVectorizer(ngram_range=(1,3), max_df=0.8, min_df=2, vocabulary=vocab)
			tfidfB = vecB.fit_transform([d['articleBody'] for d in train_dataset.data])
			print('tfidfs for bodies generated ', tfidfB.shape)
			test_tfidfB = vecB.transform([d['articleBody'] for d in test_dataset.data])
			print('tfidfs for test bodies generated ', test_tfidfB.shape)

			if mode == 'sim':
				cos_sim = np.asarray(list(map(cosine_sim, tfidfH, tfidfB)))[:, np.newaxis]
				print('cos similarity between header and body tfidfs generated ', cos_sim.shape)
				euclid_dist = np.asarray(list(map(euclidean_dist, tfidfH, tfidfB)))[:, np.newaxis]
				print('euclidean distances between header and body tfidfs generated ', euclid_dist.shape)


				test_cos_sim = np.asarray(list(map(cosine_sim, test_tfidfH, test_tfidfB)))[:, np.newaxis]
				print('cos similarity between test header and body tfidfs generated ', test_cos_sim.shape)
				test_euclid_dist = np.asarray(list(map(euclidean_dist, test_tfidfH, test_tfidfB)))[:, np.newaxis]
				print('euclidean distances between test header and body tfidfs generated ', test_euclid_dist.shape)
				
				X_train = np.concatenate((cos_sim, euclid_dist), axis = 1)
				X_test = np.concatenate((test_cos_sim, test_euclid_dist), axis = 1)
			elif mode == 'cat':
				X_train = sparse.hstack([tfidfH, tfidfB])
				X_test = sparse.hstack([test_tfidfH, test_tfidfB])

		save('features/tfidf_train_%s.%s' % (mode, ext), X_train)
		save('features/tfidf_test_%s.%s' % (mode, ext), X_test)


	y_train = [0 if d['Stance'] == 'unrelated' else 1 for d in train_dataset.data]
	y_test = [0 if d['Stance'] == 'unrelated' else 1 for d in test_dataset.data]


	if os.path.isfile('models/tfidf_%s_%s.pkl' % (classifier, mode)):
		with open('models/tfidf_%s_%s.pkl' % (classifier, mode), 'rb') as file:
			model = pickle.load(file)
		print('Classifier loaded')
	else:
		model.fit(X_train, y_train)
		print('Classifier trained')
		with open('models/tfidf_%s_%s.pkl' % (classifier, mode), 'wb') as file:
			pickle.dump(model, file)

	y_pred = model.predict(X_test)

	correct = 0
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for y_real, y_fake in zip(y_test, y_pred):
		y_fake = round(y_fake)
		if y_real == y_fake:
			correct += 1
			if y_fake == 1:
				tp += 1
			elif y_fake == 0:
				tn += 1
		elif y_fake == 1:
			fp += 1
		elif y_fake == 0:
			fn += 1
	print('\nAccuracy: ', correct / len(y_pred),
		'\nTrue Positive Rate: ', tp / len(y_pred),
		'\nTrue Negative Rate: ', tn / len(y_pred),
		'\nFalse Positive Rate: ', fp / len(y_pred),
		'\nFalse Negative Rate: ', fn / len(y_pred), '\n'
	)

def generate_bert_features(dataset, tokenizer, model, device, split=False, mode='cls', dtype='train', sim=False):
	headlines = [d['Headline'] for d in dataset]
	bodies = [d['articleBody'] for d in dataset]
	feats = []

	if split:
		h_encoding = tokenizer(headlines, return_tensors='pt', padding=True, truncation=True)
		b_encoding = tokenizer(bodies, return_tensors='pt', padding=True, truncation=True)
		print('Dataset encoded')

		for h_inp, h_type, h_att, b_inp, b_type, b_att in zip(
			h_encoding['input_ids'], h_encoding['token_type_ids'], h_encoding['attention_mask'], b_encoding['input_ids'], b_encoding['token_type_ids'], b_encoding['attention_mask']
			):
			h_output = model(h_inp.to(device).unsqueeze(0), token_type_ids = h_type.to(device).unsqueeze(0), attention_mask = h_att.to(device).unsqueeze(0), output_hidden_states=True)
			b_output = model(b_inp.to(device).unsqueeze(0), token_type_ids = b_type.to(device).unsqueeze(0), attention_mask = b_att.to(device).unsqueeze(0), output_hidden_states=True)

			if mode == 'cls':
				h_svec = h_output[0][0][0].squeeze().cpu().detach().numpy()
				b_svec = b_output[0][0][0].squeeze().cpu().detach().numpy()
			elif mode == 'pooled':
				h_svec = h_output[1][0].squeeze().cpu().detach().numpy()
				b_svec = b_output[1][0].squeeze().cpu().detach().numpy()
			elif mode == 'cat':
				h_cat = torch.cat([h_output[2][i] for i in [-1,-2,-3,-4]], dim=-1).squeeze()
				b_cat = torch.cat([b_output[2][i] for i in [-1,-2,-3,-4]], dim=-1).squeeze()
				h_svec = torch.mean(h_cat, dim=0).cpu().detach().numpy()
				b_svec = torch.mean(b_cat, dim=0).cpu().detach().numpy()
			if sim:
				feats += [[cosine_sim(h_svec, b_svec)]]
			else:
				feats += [np.append(h_svec, b_svec)]

	else:
		encoding = tokenizer(headlines, bodies, return_tensors='pt', padding=True, truncation=True)
		input_ids, token_type_ids, attention_masks = encoding['input_ids'], encoding['token_type_ids'], encoding['attention_mask']
		
		for input_id, token_type_id, attention_mask in zip(input_ids, token_type_ids, attention_masks):
			output = model(input_id.to(device).unsqueeze(0), token_type_ids = token_type_id.to(device).unsqueeze(0), attention_mask = attention_mask.to(device).unsqueeze(0), output_hidden_states=True)
		
			cls_feat = output[0].squeeze()[0]
			pooled_output = output[1].squeeze()

			hidden_states = output[2]
			cat = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]], dim=-1).squeeze()

			if mode == 'cls':
				feats += [cls_feat.cpu().detach().numpy()]
			elif mode == 'pooled':
				feats += [pooled_output.cpu().detach().numpy()]
			elif mode == 'cat':
				feats += [cat[0].cpu().detach().numpy()]

	np.save('features/%s_%s_%s_%s.npy' % (dtype, mode, 'sim' if sim else 'cat', 'split' if split else 'pair'), np.array(feats))
	print('Features Generated')
	return feats

def classify_bert(mode, classifier = 'svm', sim = False, split = False):
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	
	print('Configuration: %s, %s, %s, %s' % (mode, classifier, 'cossim' if sim else 'stack', 'split' if split else 'paired'))

	if classifier == 'randforest':
		model = RandomForestClassifier(n_estimators=1000, random_state=0)
	elif classifier == 'svm':
		model = SGDClassifier(loss='hinge', penalty='l2', random_state=42, max_iter=100, tol=None)
	elif classifier == 'lr':
		model = LogisticRegression(C=2, class_weight = 'balanced')

	train_dataset = Dataset(clean=False)
	test_dataset = Dataset(name='competition_test', clean=False)
	print('Datasets loaded')

	tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
	bert = transformers.BertModel.from_pretrained('bert-base-uncased').to(device)
	bert.eval()

	if not os.path.isfile('features/train_%s_%s_%s.npy' % (mode, 'sim' if sim else 'cat', 'split' if split else 'pair')):
		X_train = generate_bert_features(train_dataset.data, tokenizer, bert, device, mode=mode)
	else:
		X_train = np.load('features/train_%s_%s_%s.npy' % (mode, 'sim' if sim else 'cat', 'split' if split else 'pair'))
	y_train = [0 if d['Stance'] == 'unrelated' else 1 for d in train_dataset.data]

	
	if not os.path.isfile('features/test_%s_%s_%s.npy' % (mode, 'sim' if sim else 'cat', 'split' if split else 'pair')):
		X_test = generate_bert_features(test_dataset.data, tokenizer, bert, device, mode=mode, dtype='test')
	else:
		X_test = np.load('features/test_%s_%s_%s.npy' % (mode, 'sim' if sim else 'cat', 'split' if split else 'pair'))
	y_test = [0 if d['Stance'] == 'unrelated' else 1 for d in test_dataset.data]

	if os.path.isfile('models/bert_%s_%s_%s_%s.pkl' % (classifier, mode, 'sim' if sim else 'cat', 'split' if split else 'pair')):
		with open('models/bert_%s_%s_%s_%s.pkl' % (classifier, mode, 'sim' if sim else 'cat', 'split' if split else 'pair'), 'rb') as file:
			model = pickle.load(file)
		print('Classifier loaded')
	else:
		#classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
		model.fit(X_train, y_train)
		print('Classifier trained')
		with open('models/bert_%s_%s_%s_%s.pkl' % (classifier, mode, 'sim' if sim else 'cat', 'split' if split else 'pair'), 'wb') as file:
			pickle.dump(model, file)

	y_pred = model.predict(X_test)

	correct = 0
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for y_real, y_fake in zip(y_test, y_pred):
		y_fake = round(y_fake)
		if y_real == y_fake:
			correct += 1
			if y_fake == 1:
				tp += 1
			elif y_fake == 0:
				tn += 1
		elif y_fake == 1:
			fp += 1
		elif y_fake == 0:
			fn += 1
	print('\nAccuracy: ', correct / len(y_pred),
		'\nTrue Positive Rate: ', tp / len(y_pred),
		'\nTrue Negative Rate: ', tn / len(y_pred),
		'\nFalse Positive Rate: ', fp / len(y_pred),
		'\nFalse Negative Rate: ', fn / len(y_pred), '\n'
	)


if __name__ == '__main__':
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	for dirname in ['features', 'models']:
		if not os.path.isdir(dirname):
			os.mkdir(dirname)

	#train_tfidf()
	classifier = 'randforest'

	classify_tfidf('sim', classifier)

	#for classifier in ['randforest', 'lr', 'svm']:

		#classify_bert('cls', classifier, split=True, sim=True)
		#classify_bert('pooled', classifier, split=True, sim=True)
		#classify_bert('cat', classifier, split=True, sim=True)
		

		#
		#classify_tfidf('combine', classifier)
		#classify_tfidf('cat', classifier)

	mode = 'bert'
	batch_size = 30
	e = 0
	epochs = 5
	model = LSTM_cls(mode, 256, 0.2, 2, device).to(device)
	#model.eval()
	#test = model('asdf', 'yeet')
	#print(test)
	loss_function = nn.NLLLoss()
	optimizer = optim.Adam(model.parameters(), lr = 0.01)


	train_dataset = Dataset(clean = False if mode == 'bert' else True)

	for epoch in range(epochs):
		if os.path.isfile('models/%s_epoch_%d.wgt' % (mode, epoch)) and False:
			model.load_state_dict(torch.load('models/%s_epoch_%d.wgt' % (mode, epoch)))
			e = epoch + 1
	if e < epochs:
		for epoch in range(e, epochs):
			for i in range(0, len(train_dataset.data), batch_size):
				t = time.time()
				batch = train_dataset.data[i:min(i+batch_size, len(train_dataset.data))]
				headlines = [d['Headline'] for d in batch]
				bodies = [d['articleBody'] for d in batch]
				targets = torch.LongTensor([0 if d['Stance'] == 'unrelated' else 1 for d in batch]).to(device)

				model.zero_grad()

				scores = model(headlines, bodies)
				#print('\n', scores, targets)
				loss = loss_function(scores, targets)
				#print(loss)
				loss.backward()
				optimizer.step()

				t = time.time() - t
				t = ((epochs - 1 - epoch) * (((len(train_dataset.data) // batch_size) + 1) * t)) + ((((len(train_dataset.data) - i - batch_size) // batch_size) + 1) * t)

				sys.stdout.write('\rEpoch: %d | Number of Trained Pairs: %d/%d | Avg Loss for Last Batch: %f | Remaining Time: %d:%d' % (
					epoch,
					i + batch_size,
					len(train_dataset.data),
					loss.item(),
					t // 60,
					int(t % 60)
					))

			torch.save(model.state_dict(), 'models/%s_epoch_%d.wgt' % (mode, epoch))


		test_dataset = Dataset(name = 'competition_test', clean = False if mode == 'bert' else True)
		tp = 0
		tn = 0
		fp = 0
		fn = 0

		with torch.no_grad():
			for i in range(0, 1000, batch_size):
				t = time.time()
				batch = test_dataset.data[i:min(i+batch_size, len(test_dataset.data))]
				headlines = [d['Headline'] for d in batch]
				bodies = [d['articleBody'] for d in batch]
				targets = torch.LongTensor([0 if d['Stance'] == 'unrelated' else 1 for d in batch])

				scores = model(headlines, bodies)
				preds = torch.argmax(scores, dim=-1)
				print(scores)

				for pred, target in zip(preds, targets):
					if pred == target:
						if pred == 1:
							tp += 1
						else:
							tn += 1
					else:
						if pred == 1:
							fp += 1
						else:
							fn += 1

		print('\nAccuracy: ', (tp + tn) / (tp + tn + fp + fn),
			'\nTrue Positive Rate: ', tp / (tp + tn + fp + fn),
			'\nTrue Negative Rate: ', tn / (tp + tn + fp + fn),
			'\nFalse Positive Rate: ', fp / (tp + tn + fp + fn),
			'\nFalse Negative Rate: ', fn / (tp + tn + fp + fn), '\n'
		)


	

	

	'''for d in train_dataset.data:
		vecH = TfidfVectorizer(ngram_range=(1,3), max_df=0.8, min_df=2, vocabulary=vocab)
		tfidfH = vecH.fit_transform([d['Headline']])
		vecB = TfidfVectorizer(ngram_range=(1,3), max_df=0.8, min_df=2, vocabulary=vocab)
		tfidfB = vecB.fit_transform([d['articleBody']])

		cos_sim = cosine_sim(tfidfH, tfidfB)

		cos_sims += [cos_sim]
		headlines += [tfidfH]
		bodies += [tfidfB]'''



	'''
	data, bodies, headlines = load_dataset()
	body_vectorizer = TfidfVectorizer(max_df=.65, min_df=1, stop_words=stopwords.words('english'), use_idf=True, norm=None)
	headline_vectorizer = TfidfVectorizer(max_df=.65, min_df=1, stop_words=stopwords.words('english'), use_idf=True, norm=None)

	body_tfIdf = body_vectorizer.fit_transform(bodies)
	headline_tfIdf = headline_vectorizer.fit_transform(headlines)
	
	get_tfIdf(data, 15)

	train_dataset = Dataset()

	text_clf = Pipeline([
		('vect', TfidfVectorizer()),
		('tfidf', TfidfTransformer()),
		('clf-svm', SGDClassifier(loss='hinge', penalty='12', alpha=1e-3, n_iter=5, random_state=42))])



	

	# Max length is 512 tokens
	# With regards to [CLS] and [SEP], split the body into sub-bodies of length 510
	# Add a stride S so that adjacent sub-bodies have an overlap of S tokens/words
	for i in range(len(bodies)):
		body = bodies[i]
		input_ids, attention_mask = get_bert_inputs(body)
		outputs = model(input_ids, attention_mask=attention_mask)
		word_embeddings, sentence_embeddings = get_bert_features(outputs)
		break'''
