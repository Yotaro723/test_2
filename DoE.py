# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
import statsmodels.formula.api as sm

from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectPercentile
from sklearn import datasets

from pandas.tools.plotting import scatter_matrix
from scipy.stats import f_oneway
from mpl_toolkits.mplot3d import Axes3D
# seaborn（可視化の補助を行う）のインポート
import seaborn as sns


def analysis_0(df_Coredata, dataName):
	"""正規性の確認 """
	#http://www.ie-kau.net/entry/2016/03/10/正規分布かどうかを見極める3つのステップ（Python）
	#http://www4.kke.co.jp/minitab/support/newsletter/mt200912.html

	plt.hist(df_Coredata[dataName])
	plt.show()
	

def analysis_1(df_Coredata, bunbo):
	"""多変量解析による偏回帰係数の比較"""
	#https://pythondatascience.plavox.info/scikit-learn/線形回帰
	#正規化
	#https://miningoo.com/1032
	clf = linear_model.LinearRegression()

	if bunbo == 'minmax' :
		# データフレームの各列を正規化（最大・最小）
		df_Coredata = df_Coredata.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
		print('========min max========')
		print(bunbo)
		print(df_Coredata)
	elif bunbo == 'zscore':
		# データフレームの各列を正規化２(標準偏差を使用)
		df_Coredata = (df_Coredata - df_Coredata.mean()) / df_Coredata.std()
		print('========std========')
		print(bunbo)
		print(df_Coredata)

	else:
		sys.exit()

	# 説明変数に "" を利用
	df_Coredata_except_j = df_Coredata.drop("j", axis=1)
	X = df_Coredata_except_j.as_matrix()
 
	# 目的変数に "j" を利用
	Y = df_Coredata['j'].as_matrix()
 
	# 予測モデルを作成
	clf.fit(X, Y)
 
	# 偏回帰係数
	print(pd.DataFrame({"Name":df_Coredata_except_j.columns,
	                    "Coefficients":np.abs(clf.coef_)}).sort_values(by='Coefficients') )
 
	# 切片 (誤差)
	print(clf.intercept_)


def analysis_2(df_Coredata, setumei, mokuteki):
	"""２変数による回帰直線"""
	#https://pythondatascience.plavox.info/scikit-learn/線形回帰
	clf = linear_model.LinearRegression()

	# 説明変数に setumei を利用
	X = df_Coredata.loc[:,[setumei]].as_matrix()
	# 目的変数に mokuteki を利用
	Y = df_Coredata[mokuteki].as_matrix()

	#print(X)
	#print(Y)

	#相関係数
	print(df_Coredata.corr(method='pearson'))

	# 予測モデルを作成
	clf.fit(X, Y)
 
	# 回帰係数
	print(clf.coef_)
	# 切片 (誤差)
	print(clf.intercept_)
	# 決定係数
	#print(clf.score(X, Y))

	### 散布図
	plt.scatter(X, Y)
	### 回帰直線
	plt.plot(X, clf.predict(X))

	plt.show()

def analysis_3(df_Coredata, setumei, mokuteki):
	"""一元配置分散分析　：　等分散検定　バートレットの検定 """
	# http://lang.sist.chukyo-u.ac.jp/classes/PythonProbStat/Python-statistics6.html
	# https://py3math.hatenablog.com/entry/oneway-anovatests1

	# 使用したデータは『すぐできる生物統計』
	#
	# 一元配置分散分析:
	# 概要:
	# 2つのグループが互いに有意に違っているかどうか。
	#
	# 帰無仮説:
	# 2つのグループが同じ *** をもっている。


	#ユニークな要素のリスト
	u = df_Coredata[setumei].unique()	
	print( "水準数 : " + str(len(u)) )
	
	#listNumb = range(10)
	#print(listNumb)	

	#最小の水準数にあわせる。最大だとnp.sumで問題発生。
	valList = []
	j = 0
	for i in u:
		valList.append( len(df_Coredata[ (df_Coredata[setumei] == i) ] ) )
		

	print( "最小サンプル数 : " + str(min(valList)) )
	minVal = min(valList)

	df_temp = pd.DataFrame(index = range(minVal))
	
	#インデックスを0からの連番としたデータフレームの定義
	for i in u:
		#print(df_Coredata[ (df_Coredata[setumei] == i) ])
		df_temp_2 = df_Coredata[ (df_Coredata[setumei] == i) ]
		df_temp_2.index = range(len(df_temp_2))
		df_temp[i] = df_temp_2[mokuteki]

	print(df_temp)

	GroupAverageMatrix =np.ones(df_temp.shape)
	for i in range(df_temp.shape[1]):
		GroupAverageMatrix[:,i] = df_temp.mean().iloc[i]

	InGroup = np.array(df_temp - GroupAverageMatrix)

	InGroupSquareSum = np.sum(InGroup**2)

	OverallMean = np.sum(df_temp.mean())/len(df_temp.columns)

	InterGroup =GroupAverageMatrix - np.ones(df_temp.shape)*OverallMean

	InterGroupSquareSum = np.sum(InterGroup**2)

	Dividend = InterGroupSquareSum / (len(df_temp.columns) - 1.0)

	Divider = InGroupSquareSum /( ( len(df_temp.index) -1.0)*len(df_temp.columns))

	print(Divider)
	
	
	#### ｆ値の導出
	print( df_temp[ u[0:len(u)] ] )

	res = f_oneway( df_temp[ u[0] ] , df_temp[ u[1] ], df_temp[ u[2] ], df_temp[ u[3] ] )

	print("f_oneway : " + str( res ))

	F_value, p_value = res

	# 帰無仮説が棄却されるかどうか。
	if p_value < 0.05:
		print('p 値: {} < 0.05'.format(p_value))
		print('帰無仮説は棄却される。')
	else:
		print('p 値: {} > 0.05'.format(p_value))
		print('帰無仮説は棄却されない。')

	#バートレット検定
	bt_results = stats.bartlett( df_temp[ u[0] ] , df_temp[ u[1] ], df_temp[ u[2] ], df_temp[ u[3] ] )
	print(bt_results)


def analysis_4(df_Coredata, setumei_1, setumei_2 , mokuteki):
	"""三次元散布図"""
	#http://publicjournal.hatenablog.com/entry/2018/01/28/194101

	# 初期化
	fig = plt.figure()
	ax = Axes3D(fig)

	xs = df_Coredata[setumei_1].values
	ys = df_Coredata[setumei_2].values
	zs = df_Coredata[mokuteki].values

	print(xs)
	print(ys)
	print(zs)

        #相関係数
	print(df_Coredata.corr(method='pearson'))

	#reg = "y ~ x1 + x2"
	#model = sm.ols(formula=reg, data=df.T)

	# 回帰分析を実行する
	#result = model.fit()

	# 描画
	ax.scatter3D(xs, ys, zs)
	plt.show()

def analysis_5(df_Coredata, setumei_1, setumei_2, mokuteki):
	"""contour 高等線図プロット """
	#http://zeroemon00.com/2017/07/17/【python】excel表から等高線グラフを描く方法/

	#setumei_1 と setumei_2 が重複した目的変数を削除 **いずれは平均をとる。
	df_del_juhuku  = df_Coredata[~df_Coredata.duplicated(subset=[setumei_1, setumei_2])]

	print(df_del_juhuku)

        #ユニークな要素のリスト
	PmName_1 = df_del_juhuku[setumei_1].unique()
	PmName_2 = df_del_juhuku[setumei_2].unique()

	PmName_1.sort()
	PmName_2.sort()


	for i in PmName_1:
		for j in PmName_2:
			df_temp  = df_del_juhuku[ ( df_del_juhuku[setumei_1] == i) & (df_del_juhuku[setumei_2] == j) ]
			
			print('i : ' + str(i) + ' j : ' + str(j))
			print( '配列の要素数 : ' + str ( len(df_temp[mokuteki].values) ) )
			print( df_temp[mokuteki].values )
			if len( df_temp[mokuteki] == 0 ) :
				print('エラー　：　数字が存在しない')
				sys.exit()
			
	

	#xs = df_Coredata[setumei_1].values
	#ys = df_Coredata[setumei_2].values
	#zs = df_Coredata[mokuteki].values

	#X,Y = np.meshgrid(xs, ys)

	#arr = [[0 for i in range(len(xs))] for j in range(len(ys))]

	#for i in range(len(xs)):
	#	for j in range(len(ys)):
	#		arr[j][i] = df_Coredata.iloc([j][i])


def analysis_7(df_Coredata):
	""" 多次元多項式モデル """

	#https://www.jeremyjordan.me/polynomial-regression/

	X = df_Coredata[['d','e','f','g','i']]
	y = df_Coredata['j']

	# グラフのスタイルを指定
	sns.set(style = 'whitegrid', context = 'notebook')
	# 変数のペアの関係をプロット
	#sns.pairplot(df_Coredata)
	#plt.show()


	#X_train, X_test, y_train, y_test  =  train_test_split(X,y,random_state = 0)
	#lr = linear_model.LinearRegression().fit(X_train, y_train)
	#print("Trainng set score: {:.2f}".format(lr.score(X_train, y_train)))
	#print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

	### データのスケール変換
	# 標準化
	std_Scaler = StandardScaler()
	data_std = std_Scaler.fit_transform(X)

	mmx_Scaler =MinMaxScaler()
	X_scaled = mmx_Scaler.fit_transform(X)
	#X_test_scaled = scaler.transform(X_test)

	#print(X_train_scaled)

	poly = PolynomialFeatures(degree = 2).fit(data_std)
	print(poly.get_feature_names())
	

	#X_train_poly = poly.transform(X_train_scaled)
	#X_test_poly = poly.transform(X_test_scaled)

	#select = SelectPercentile(percentile = 10)
	#select.fit(X_train_poly, y_train)
	
	#mask = select.get_support()
	#print(mask)
	

	#lr_2 = linear_model.LinearRegression().fit(X_train_poly, y_train)
	
	#print("poly Trainng set score: {:.2f}".format(lr_2.score(X_train_poly, y_train)))
	#print("Test set score: {:.2f}".format(lr_2.score(X_test_poly, y_test)))

	#print("X_train.shape :{} ".format(X_train_poly.shape))
	#print("polynominal feature names : \n{}".format(poly.get_feature_names()))
	#print(df_Coredata.columns)
	#print(X_train_poly[:,0])

def analysis_8(df_Coredata):
	'''次元圧縮_PCA'''
	#scikit-learn で主成分分析 (PCA) してみる
	#http://blog.amedama.jp/entry/2017/04/02/130530
	#http://i.cla.kobe-u.ac.jp/murao/class/2015-SeminarB3/05_Python_de_PCA.pdf

	dataset = datasets.load_iris()
	print(dataset)
	#データの列数取得
	dataArrayName = df_Coredata.columns.values
	#print(dataArrayName)
	dataArrayNum = len( df_Coredata.columns )


	#標準化
	scaler = StandardScaler()
	scaler.fit(df_Coredata)
	scaled_data = scaler.transform(df_Coredata)

	#print(scaled_data)

	###ここ一般化
	X_scaled, Y_scaled  =np.split(scaled_data, [dataArrayNum - 1], axis=1)

	colors = [plt.cm.hsv(0.1 * i, 1) for i in range(dataArrayNum)]
	
	pca = PCA(n_components = 2)
	pca.fit(scaled_data)

	#寄与率の確認
	print('寄与率' + str(pca.explained_variance_ratio_))
	#因子負荷量の確認
	print('因子負荷量' + str(pca.components_))

	x_pca = pca.transform(scaled_data)

	#print(x_pca)
	for i in range(dataArrayNum):
		plt.scatter(x_pca[i,0], x_pca[i,1], c=colors[i], label = dataArrayName[i])

	plt.show()


def arrange_0(df_Coredata, numLevel, setumei):
	'''複数の値をとる（ばらつきのある）変数をnumLevelにて指定された水準数に書き換える'''
	# ロジック
	# 最大と最少値・・・ばらつきめんどくさい。
	
	df_temp = df_Coredata[setumei]

############################

if __name__ == '__main__':

	fileName = 'experimentalData.csv'

	df = pd.read_csv(fileName, encoding="SHIFT-JIS")

	### カラムのラベルをリネーム
	print(df.columns)
	numbColumns = len( df.columns )
	df.columns = [chr(i) for i in range(97, 97 + numbColumns)]

	###条件を固定
	#f : 電圧
	#e : 押し当て荷重
	#df_Coredata = df[(df['d'] == 2) & (df['f'] <= 1500) & (df['g'] <= 10)]
	#df_Coredata = df[(df['d'] == 2) & (df['i'] == 800)]
	#全変数を検討する場合
	df_Coredata = df

	pd.set_option('display.max_columns', 100)
	print(df_Coredata)
	print('サンプル数' + str( len(df_Coredata)) )

	###データの整形
	#すべて同じ値を持つ列は除外しないとエラーを引き起こす
	#hはパルス間隔
	#パルス数除外バージョン
	#df_Coredata = df_Coredata[['e','f','g','i','j']]

	df_Coredata = df_Coredata[['d','e','f','g','i','j']]
	df_Coredata = df_Coredata.replace('-', np.nan)
	df_Coredata = df_Coredata.dropna()
	df_Coredata = df_Coredata.astype(float)
	
	#多変量解析 minmax 正規化を最大最小で行う。 zscore 正規化を標準偏差で行う	
	#analysis_1(df_Coredata, 'minmax')

	#2変数による回帰直線
	#analysis_2(df_Coredata, 'g', 'j')

	#3次元プロット
	#analysis_4(df_Coredata, 'g', 'e' ,'j')

	#高等線図プロット
	#analysis_5(df_Coredata, 'g', 'i' ,'j')

	#scikitlearn学習
	#analysis_7(df_Coredata)

	#scikitlearnによる主成分分析
	analysis_8(df_Coredata)

	#arrange_0(df_Coredata, 3 , 'e')
