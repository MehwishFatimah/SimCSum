from datasets import load_metric
import pandas as pd
import torch
import os
from nltk.tokenize import word_tokenize, sent_tokenize

class Evaluate(object):
	'''----------------------------------------------------------------
	Initialize evaluation object 
	'''
	def __init__(self, folder):

		self.metric = ""
		self.folder = folder
		self.device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	
	'''----------------------------------------------------------------
	Args:
		pred  : str 
		ref   : str 
	Return:
		s: [0,1]float
	'''
	def compute_sari(self, src, pred, ref):

		source = []
		source.append(src)
		prediction = []
		prediction.append(pred) 
		ref_list = []
		ref_list.append(ref) 
		reference = []
		reference.append(ref_list) # list of list
		#compute takes list
		score = self.metric.compute(sources= source, predictions= prediction, references= reference)
		print(score)
		
		
		return score['sari']

	'''----------------------------------------------------------------
	Args:
		pred  : str 
		ref   : str 
	Return:
		s: [0,1]float
	'''
	def compute_mauve(self, pred, ref):

		prediction = []
		prediction.append(pred) 
		reference = []
		reference.append(ref)
		#compute takes list
		score = self.metric.compute(predictions= prediction, references= reference, device_id = 0)
		print(score)
		#print(score.get('precision')[0])
		
		
		return score.mauve, score.frontier_integral, score.divergence_curve, score.p_hist, score.q_hist


	'''----------------------------------------------------------------
	Args:
		pred  : str 
		ref   : str 
	Return:
		s: [0,1]float
	'''
	def compute_meteor(self, pred, ref):

		prediction = []
		prediction.append(pred) 
		reference = []
		reference.append(ref)
		#compute takes list
		score = self.metric.compute(predictions= prediction, references= reference)
		print(score)
		#print(score.get('precision')[0])
		s = round(score['meteor'], 4)
		
		return s


	'''----------------------------------------------------------------
	Args:
		pred  : str 
		ref   : str 
	Return:
		p, r, f: float
	'''
	def compute_bert(self, pred, ref):

		prediction = []
		prediction.append(pred) 
		ref_list = []
		ref_list.append(ref) 
		reference = []
		reference.append(ref_list) #list of list
		#compute takes list
		score = self.metric.compute(predictions= prediction, references= reference, lang="de")
		print(score)
		#print(score.get('precision')[0])
		p = round(score.get('precision')[0], 4)*100
		#print(p)
		r = round(score.get('recall')[0], 4)*100
		f = round(score.get('f1')[0], 4)*100
		
		return p, r, f

	'''----------------------------------------------------------------
	Args:
		pred  : str 
		ref   : str 
	Return:
		bs, p, bp, ratio, r_length, p_length: float except p (list)
	'''
	def compute_bleu(self, pred, ref):

		prediction = [[i for i in word_tokenize(pred)]] #for s in sent_tokenize(pred)]
		
		#prediction = []#list of list
		#prediction.append(pred)#
		print('p: {}'.format(prediction))
		#reference = [[[]]] #list of list of list is required
		reference = [[[i for i in word_tokenize(ref)]]] #for s in sent_tokenize(ref)]
		print('r: {}'.format(reference))
		
		#print(type(reference)) 
		#compute takes list
		score = self.metric.compute(predictions= prediction, references= reference)
		#print(score.keys())
		bs = round(score.get('bleu'), 4)
		p = score.get('precision')
		bp = score.get('brevity_penalty')
		ratio = score.get('length_ratio')
		r_length = score.get('reference_length')
		p_length = score.get('translation_length')

		return bs, p, bp, ratio, r_length, p_length

	'''----------------------------------------------------------------
	Args:
		pred  : str 
		ref   : str 
		r_type: str
	Return:
		r_p, r_r, r_f: float
	'''
	def compute_rouge(self, pred, ref, r_type):

		prediction = []
		prediction.append(pred)
		#print(prediction)
		reference = []
		reference.append(ref) 
		#print(reference)
		#compute takes list
		score = self.metric.compute(predictions= prediction, references= reference, rouge_types=[r_type])[r_type].mid
		print(score)
		r_p = round(score.precision, 4)*100
		r_r = round(score.recall, 4)*100
		r_f = round(score.fmeasure, 4)*100

		return r_p, r_r, r_f

	'''----------------------------------------------------------------
	Args:
		file: file
	Return: None
	'''
	def write_bert(self, file):

		df = pd.read_csv(file)
		#df = df.head(5)
		
		sdf = pd.DataFrame()
		self.metric = load_metric("bertscore")
		print('****************bertscore****************')
		
		sdf['bs_p'], sdf['bs_r'], sdf['bs_f'] \
		= zip(*df.apply(lambda x: self.compute_bert(x['system'], x['reference']), axis=1))
		
		#sdf['bs_p'] = sdf['bs_p'].multiply(100)
		#sdf['bs_r'] = sdf['bs_r'].multiply(100)
		#sdf['bs_f'] = sdf['bs_f'].multiply(100)
		
		out_file = os.path.join(self.folder, 'bert.csv')	
		sdf.to_csv(out_file, index=False, encoding='utf-8')
		print('****************bertscore end')

	'''----------------------------------------------------------------
	Args:
		file: file
	Return: None
	'''
	def write_bleu(self, file):

		df = pd.read_csv(file)
		#df = df.head(5)
		
		sdf = pd.DataFrame()
		self.metric = load_metric("bleu")
		print('****************blue****************')      
		sdf['bleu'], sdf['p'], sdf['bp'], sdf['lr'], sdf['rl'], sdf['pl'] \
		= zip(*df.apply(lambda x: self.compute_bleu(x['system'], x['reference']), axis=1))	

		out_file = os.path.join(self.folder, 'bleu.csv')	
		sdf.to_csv(out_file, index=False, encoding='utf-8')
		print('****************bleu end')

	'''----------------------------------------------------------------
	Args:
		file: file
	Return: None
	'''
	def write_rouge(self, file):

		df = pd.read_csv(file)
		#df = df.head(5)
		#print(df)
		sdf = pd.DataFrame()
		self.metric = load_metric("rouge")
		print('****************rouge****************')
		sdf['r1_p'], sdf['r1_r'], sdf['r1_f'] \
		= zip(*df.apply(lambda x: self.compute_rouge(x['system'], x['reference'], "rouge1"), axis=1))
		
		sdf['r2_p'], sdf['r2_r'], sdf['r2_f'] \
		= zip(*df.apply(lambda x: self.compute_rouge(x['system'], x['reference'], "rouge2"), axis=1))

		sdf['rL_p'], sdf['rL_r'], sdf['rL_f'] \
		= zip(*df.apply(lambda x: self.compute_rouge(x['system'], x['reference'], "rougeL"), axis=1))

		#sdf['rLS_p'], sdf['rLS_r'], sdf['rLS_f'] \
		#= zip(*df.apply(lambda x: self.compute_rouge(x['system'], x['reference'], "rougeLSum"), axis=1))

		out_file = os.path.join(self.folder, 'rouge.csv')
		sdf.to_csv(out_file, index=False, encoding='utf-8')
		print('****************rouge end')


	'''----------------------------------------------------------------
	Args:
		file: file
	Return: None
	'''
	def write_meteor(self, file):

		df = pd.read_csv(file)
		#df = df.head(5)
		#print(df)
		
		sdf = pd.DataFrame()
		self.metric = load_metric("meteor")
		print('****************\n{}\n****************'.format(self.metric))
		print()
		sdf['ms']= df.apply(lambda x: self.compute_meteor(x['system'], x['reference']), axis=1)
		
		out_file = os.path.join(self.folder, 'meteor.csv')
		sdf.to_csv(out_file, index=False, encoding='utf-8')
		print('****************meteor end')

	'''----------------------------------------------------------------
	Args:
		file: file
	Return: None
	'''
	def write_mauve(self, file):

		df = pd.read_csv(file)
		#df = df.head(5)
		#print(df)
		
		sdf = pd.DataFrame()
		self.metric = load_metric("mauve")
		print('****************\n{}\n****************'.format(self.metric))
		print()
		sdf['mauve'], sdf['fi'], sdf['dc'], sdf['ph'], sdf['qh'] \
		= zip(*df.apply(lambda x: self.compute_mauve(x['system'], x['reference']), axis=1))
		
		out_file = os.path.join(self.folder, 'mauve.csv')
		sdf.to_csv(out_file, index=False, encoding='utf-8')
		print('****************mauve end')
	'''----------------------------------------------------------------
	Args:
		file: file
	Return: None
	'''
	def write_sari(self, sfile, file):

		ss = pd.read_csv(sfile)
		df = pd.read_csv(file)
		df['text'] = ss['text'] 
		#df = df.head(5)
		#print(df)

		sdf = pd.DataFrame()
		self.metric = load_metric("sari")
		print('****************\n{}\n****************'.format(self.metric))
		print()
		sdf['sari']= df.apply(lambda x: self.compute_sari(x['text'], x['system'], x['reference']), axis=1)
		
		out_file = os.path.join(self.folder, 'sari.csv')
		sdf.to_csv(out_file, index=False, encoding='utf-8')

		print('****************sari end')


	'''----------------------------------------------------------------
	Args:
		file: file
	Return: None
	'''
	def evaluation(self, sfile, file):
		
		self.write_rouge(file)
		self.write_bleu(file)
		self.write_bert(file)		
		self.write_meteor(file)
		self.write_mauve(file)
		self.write_sari(sfile, file)
		
		