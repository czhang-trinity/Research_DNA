import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

#matplotlib choose xwindows by default..
#matplotlib.use('Agg')

df = pd.read_csv('after_combat.txt').dropna()
#df = df.drop(columns = ['Unnamed: 0'])
#print(len(df))

#listCols = df.iloc[:, 2:45] #45 columnns of cites

#print (listCols)

listDatasets = df['availability'].unique()

print(listDatasets)
#['GSE19711' 'GSE40279' 'GSE41037' 'GSE41169' 'GSE42861']

#sample1 = ['3.14','0.4544','1.797']
#sample2 = [0.656,0.98451, 0.737]
#sample3 = [0.1316544, 21.44,8.88]
#F, p = stats.f_oneway(sample1, sample2, sample3)
#F, p = stats.f_oneway(listCols)
#print(F)
#print(p)

#boxplot = df.boxplot('cg00374717',by='availability')
#ava = df['cg00374717'][df.availability=='GSE19711']

grps = pd.unique(df.availability.values)

def anovaByCite(citeName, grps):
    #list of entries associated with each group
    dataGrps  = [ df[citeName][df.availability==grp] for grp in grps]
    fig1, ax1 = plt.subplots()
    ax1.set_title('ANOVA Test')
    ax1.boxplot(dataGrps)
    plt.show()

lstCites = ["cg16408394", "cg25683012", "cg19761273", "cg27544190", "cg03588357","cg03286783", "cg19273182", "cg15703512", "cg01511567","cg09441152","cg02047577", "cg17338403", "cg07158339", "cg01873645", "cg05442902","cg04999691", "cg24450312", "cg04452713", "cg22613010", "cg09646392","cg17274064", "cg16984944", "cg00436603", "cg24126851", "cg14723032","cg06926735", "cg14308452", "cg00374717", "cg07455279", "cg02085507", "cg20692569", "cg04528819", "cg08370996", "cg26297688", "cg23092072","cg04084157", "cg01968178", "cg25505610", "cg06993413", "cg00864867","cg22736354", "cg06493994", "cg02479575", "cg16241714", "cg14424579"]

print(lstCites)
#anovaByCite('age', grps)

for c in lstCites:
    print(c)
    anovaByCite(c, grps)



