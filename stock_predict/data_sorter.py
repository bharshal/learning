import pandas as pd
import os
import time
from time import mktime
import matplotlib.pyplot as plt
import re
from matplotlib import style
style.use("dark_background")
from datetime import datetime

path= './intraQuarter'

def key_stats(gather=["Total Debt/Equity (mrq)",
					'Trailing P/E',
                      'Price/Sales',
                      'Price/Book',
                      'Profit Margin',
                      'Operating Margin',
                      'Return on Assets',
                      'Return on Equity',
                      'Revenue Per Share',
                      'Market Cap',
                        'Enterprise Value',
                        'Forward P/E',
                        'PEG Ratio',
                        'Enterprise Value/Revenue',
                        'Enterprise Value/EBITDA',
                        'Revenue',
                        'Gross Profit',
                        'EBITDA',
                        'Net Income Avl to Common ',
                        'Diluted EPS',
                        'Earnings Growth',
                        'Revenue Growth',
                        'Total Cash',
                        'Total Cash Per Share',
                        'Total Debt',
                        'Current Ratio',
                        'Book Value Per Share',
                        'Cash Flow',
                        'Beta',
                        'Held by Insiders',
                        'Held by Institutions',
                        'Shares Short (as of',
                        'Short Ratio',
                        'Short % of Float',
                        'Shares Short (prior ']):
	statspath = path+"/_KeyStats"
	stocklist = [x[0] for x in os.walk(statspath)]
	df=pd.DataFrame(columns=['Date',
		'Unix',
		'Ticker',
		'Price',
		'stock_p_chng',
		'SP500',
		'sp500_p_chng',
		'Diff',
		'Status',
		'DE_ratio',
		'Trailing P/E',
         'Price/Sales',
         'Price/Book',
         'Profit Margin',
         'Operating Margin',
         'Return on Assets',
         'Return on Equity',
         'Revenue Per Share',
         'Market Cap',
         'Enterprise Value',
         'Forward P/E',
         'PEG Ratio',
         'Enterprise Value/Revenue',
         'Enterprise Value/EBITDA',
         'Revenue',
         'Gross Profit',
         'EBITDA',
         'Net Income Avl to Common ',
         'Diluted EPS',
         'Earnings Growth',
         'Revenue Growth',
         'Total Cash',
         'Total Cash Per Share',
         'Total Debt',
         'Current Ratio',
         'Book Value Per Share',
         'Cash Flow',
         'Beta',
         'Held by Insiders',
         'Held by Institutions',
         'Shares Short (as of',
         'Short Ratio',
         "Short % of Float",
         'Shares Short (prior ',                                
         ##############
         'Status'])
	#print(stocklist)
	sp500=pd.read_csv("index.csv")
	print(sp500.index)
	ticker_list= []
	for each_dir in sorted(stocklist[1:50]):
		each_file=os.listdir(each_dir)
		ticker=each_dir.split("/")[-1]
		ticker_list.append(ticker)
		starting_stock_value=False	
		starting_sp500_value=False
		#print('supertop')


		if len(each_file)>0:
			for file in each_file:
				#if not (file[3]=="6"):

				#if not (file[3]=="6" or file[3]=="9"):
				date_stamp=   datetime.strptime (file,'%Y%m%d%H%M%S.html')
				unix_time=time.mktime(date_stamp.timetuple())
				#print(date_stamp,unix_time)
				full_file_path=each_dir+'/'+file
				#print(file+':'+ticker)
				source=open(full_file_path,'r').read()
				#print("top")
				#print(source)
				try:
					try:
						value=float(source.split(gather+':</td><td class="yfnc_tabledata1">')[1].split('</td>')[0])
					except Exception as e:
						value=float(source.split(gather+':</td>\n<td class="yfnc_tabledata1">')[1].split('</td>')[0])
					try:
						sp500_date=datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d')							
						row=sp500.loc[sp500['Date']==sp500_date]
						#row=sp500[(sp500.==sp500_date)]
						sp500_value=float(row["Adj Close"])
					except:
						sp500_date=datetime.fromtimestamp(unix_time-259200).strftime('%Y-%m-%d')
						row=sp500.loc[sp500['Date']==sp500_date]
						#row=sp500[(sp500.index==sp500_date)]
						sp500_value=float(row["Adj Close"])


					try:
						stock_price=float(source.split('</small><big><b>')[1].split('</b></big>')[0])
					except Exception as e:
						#pass
						try:
							stock_price=source.split('</small><big><b>')[1].split('</b></big>')[0]
							#print(stock_price)
							stock_price=stock_price.split('>')[1].split('<')[0]

							#stock_price=re.search(r'(\d(1,8)\.\d(1,8))',stock_price)
							#print('ddone',stock_price)
							stock_price=float(stock_price)
							#print("done",stock_price)
						except Exception as e:
							pass
							# stock_price=(source.split('<span class="time_rtq_ticker">')[1].split('</span>')[0])
							# print(stock_price)
							# stock_price=stock_price.split('>')[1].split('<')[0]
							# print("second",stock_price)

							#print(str(e),file,ticker)

					#print("stock_price: ",stock_price,"ticker: ",ticker)
					if not starting_stock_value:
						starting_stock_value=stock_price
					if not starting_sp500_value:
						starting_sp500_value=sp500_value
					#print("first    ",starting_sp500_value,"    ",starting_stock_value)
					#print("second   ",sp500_value,"     ",stock_price)

					stock_p_chng=((stock_price- starting_stock_value)/starting_stock_value)*100
					sp500_p_chng=((sp500_value- starting_sp500_value)/starting_sp500_value)*100
					diff =stock_p_chng - sp500_p_chng
					if diff>0:
						status='outperformed'
					else:
						status='underperformed'
					#print(stock_p_chng,"    ",sp500_p_chng)
					df=df.append({'Date':date_stamp,
						'Unix':unix_time,
						'Ticker':ticker,
						'DE_ratio':value,
						'Price':stock_price,
						'stock_p_chng':stock_p_chng,
						'SP500':sp500_value,
						'sp500_p_chng':sp500_p_chng,
						'Diff': diff,
						'Status':status,
						'Trailing P/E':value_list[1],
                        'Price/Sales':value_list[2],
                        'Price/Book':value_list[3],
                        'Profit Margin':value_list[4],
                        'Operating Margin':value_list[5],
                        'Return on Assets':value_list[6],
                        'Return on Equity':value_list[7],
                        'Revenue Per Share':value_list[8],
                        'Market Cap':value_list[9],
                         'Enterprise Value':value_list[10],
                         'Forward P/E':value_list[11],
                         'PEG Ratio':value_list[12],
                         'Enterprise Value/Revenue':value_list[13],
                         'Enterprise Value/EBITDA':value_list[14],
                         'Revenue':value_list[15],
                         'Gross Profit':value_list[16],
                         'EBITDA':value_list[17],
                         'Net Income Avl to Common ':value_list[18],
                         'Diluted EPS':value_list[19],
                         'Earnings Growth':value_list[20],
                         'Revenue Growth':value_list[21],
                         'Total Cash':value_list[22],
                         'Total Cash Per Share':value_list[23],
                         'Total Debt':value_list[24],
                         'Current Ratio':value_list[25],
                         'Book Value Per Share':value_list[26],
                         'Cash Flow':value_list[27],
                         'Beta':value_list[28],
                         'Held by Insiders':value_list[29],
                         'Held by Institutions':value_list[30],
                         'Shares Short (as of':value_list[31],
                         'Short Ratio':value_list[32],
                         'Short % of Float':value_list[33],
                         'Shares Short (prior ':value_list[34]
						},ignore_index=True)
				except Exception as e:#print(str(e))
					pass
					#print("bottom")
	for each_ticker in ticker_list:
		try:
			plot_df=df[(df['Ticker']==each_ticker)]
			plot_df=plot_df.set_index(['Date'])
			if plot_df['Status'][-1]=='underperformed':
				color='r'
			else:
				color='g'
			plot_df['Diff'].plot(label=each_ticker,color=color)
			plt.legend()
		except Exception as e:
			print(str(e),each_ticker)
			#time.sleep(10)
			#pass



		save=gather.replace(' ','').replace(')','').replace('(','').replace('/','')+('.csv') 
		#print(save)
		df.to_csv(save)
	plt.show()


key_stats()
