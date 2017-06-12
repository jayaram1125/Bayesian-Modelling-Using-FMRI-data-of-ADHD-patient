import nilearn.datasets
import input_data.NiftiMapsMasker
from pgmpy.models import LinearGaussianBayesianNetwork  as LGB
from  pgmpy.factors.continuous import LinearGaussianCPD as LGCPD
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
import sklearn.linear_model.LinearRegression
from pgmpy.sampling import HamiltonianMC as HMC, LeapFrog, GradLogPDFGaussian
from pgmpy.factors.continuous import JointGaussianDistribution
import scipy.stats as sp


atlas = datasets.fetch_atlas_msdl()
atlas_filename = atlas['maps']
labels = atlas['labels']
data = datasets.fetch_adhd(n_subjects=1)
print('First subject resting-state nifti image (4D) is located at: %s' %data.func[0])
masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,memory='nilearn_cache', verbose=5)
time_series = masker.fit_transform(data.func[0],confounds=data.confounds)
print(time_series.shape)
print(time_series[1])


meanvals = []
variancevals = []
sigmavals = []

for i in range(0,39):
   meanvals.append(np.mean(time_series[:,i]))
   variancevals.append(np.var(time_series[:,i]))
   sigmavals.append(np.std(time_series[:,i]))

   model = LGB([('var1'),
                ('var2', 'var1'), ('var2', 'var27'),
                ('var3'),
                ('var4'),
                ('var5', 'var35'), ('var5', 'var37'),
                ('var6', 'var4'), ('var6', 'var33'),
                ('var7', 'var5'), ('var7', 'var19'), ('var7', 'var33'),
                ('var8', 'var22'),
                ('var9', 'var11'),
                ('var10', 'var4'), ('var10', 'var11'), ('var10', 'var19'), ('var10', 'var25'), ('var10', 'var32'),
                ('var11'),
                ('var12', 'var10'), ('var12', 'var15'), ('var12', 'var20'),
                ('var13', 'var5'),
                ('var14', 'var1'), ('var14', 'var3'), ('var14', 'var35'),
                ('var15'),
                ('var16', 'var15'),
                ('var17', 'var11'), ('var17', 'var23'),
                ('var18', 'var19'), ('var19', 'var25'),
                ('var19'),
                ('var20', 'var3'),
                ('var21', 'var3'), ('var21', 'var22'),
                ('var22', 'var20'),
                ('var23', 'var1'),
                ('var24', 'var25'),
                ('var25'),
                ('var26', 'var4'), ('var26', 'var20'), ('var26', 'var22'), ('var26', 'var27'),
                ('var27'),
                ('var28', 'var16'), ('var28', 'var20'), ('var28', 'var23'), ('var28', 'var29'),
                ('var29'),
                ('var30', 'var3'), ('var30', 'var18'), ('var30', 'var6'), ('var30', 'var29'),
                ('var31', 'var32'),
                ('var32', 'var29'),
                ('var33'),
                ('var34', 'var29'),
                ('var35', 'var31'),
                ('var36', 'var6'), ('var36', 'var37'),
                ('var37', 'var32'), ('var37', 'var35'),
                ('var38'),
                ('var39', 'var19'), ('var39', 'var36'), ('var39', 'var38')])

df = pd.read_csv('data_MRI.csv')


def betaVectorCalculator(feature_cols, df, s):
    X = df[feature_cols]
    y = s
    lm = LinearRegression()
    lm.fit(X, y)
    beta = []
    beta.append(lm.intercept_)
    beta.extend(lm.coef_.tolist())
    return beta


'''CPD1'''
beta = betaVectorCalculator(['var2', 'var14', 'var23'], df, df.var1)
print("beta1=" + str(beta))
cpd1 = LGCPD('var1', beta, variancevals[0], ['var2', 'var14', 'var23'])

'''CPD2'''
cpd2 = LGCPD('var2', [0], variancevals[1])

'''CPD3'''
beta = betaVectorCalculator(['var14', 'var20', 'var21', 'var30'], df, df.var3)
print("beta3=" + str(beta))
cpd3 = LGCPD('var3', beta, variancevals[2], ['var14', 'var20', 'var21', 'var30'])

'''CPD4'''
beta = betaVectorCalculator(['var6', 'var10', 'var26'], df, df.var4)
print("beta4=" + str(beta))
cpd4 = LGCPD('var4', beta, variancevals[3], ['var6', 'var10', 'var26'])

'''CPD5'''
beta = betaVectorCalculator(['var7', 'var13'], df, df.var5)
print("beta5=" + str(beta))
cpd5 = LGCPD('var5', beta, variancevals[4], ['var7', 'var13'])

'''CPD6'''
beta = betaVectorCalculator(['var30', 'var36'], df, df.var6)
print("beta6=" + str(beta))
cpd6 = LGCPD('var6', beta, variancevals[5], ['var30', 'var36'])

'''CPD7'''
cpd7 = LGCPD('var7', [0], variancevals[6])

'''CPD8'''
cpd8 = LGCPD('var8', [0], variancevals[7])

'''CPD9'''
cpd9 = LGCPD('var9', [0], variancevals[8])

'''CPD10'''
beta = betaVectorCalculator(['var12'], df, df.var10)
print("beta10=" + str(beta))
cpd10 = LGCPD('var10', beta, variancevals[9], ['var12'])

'''CPD11'''
beta = betaVectorCalculator(['var9', 'var10', 'var17'], df, df.var11)
print("beta11=" + str(beta))
cpd11 = LGCPD('var11', beta, variancevals[10], ['var9', 'var10', 'var17'])

'''CPD12'''
cpd12 = LGCPD('var12', [0], variancevals[12])

'''CPD13'''
cpd13 = LGCPD('var13', [0], variancevals[13])

'''CPD14'''
cpd14 = LGCPD('var14', [0], variancevals[14])

'''CPD15'''
beta = betaVectorCalculator(['var12', 'var16'], df, df.var15)
print("beta15=" + str(beta))
cpd15 = LGCPD('var15', beta, variancevals[14], ['var12', 'var16'])

'''CPD16'''
beta = betaVectorCalculator(['var28'], df, df.var16)
print("beta16=" + str(beta))
cpd16 = LGCPD('var16', beta, variancevals[15], ['var28'])

'''CPD17'''
cpd17 = LGCPD('var17', [0], variancevals[16])

'''CPD18'''
beta = betaVectorCalculator(['var30'], df, df.var18)
print("beta18=" + str(beta))
cpd18 = LGCPD('var18', beta, variancevals[17], ['var30'])

'''CPD19'''
beta = betaVectorCalculator(['var7', 'var10', 'var18', 'var39'], df, df.var19)
print("beta19=" + str(beta))
cpd19 = LGCPD('var19', beta, variancevals[18], ['var7', 'var10', 'var18', 'var39'])

'''CPD20'''
beta = betaVectorCalculator(['var12', 'var22', 'var26', 'var28'], df, df.var20)
print("beta20=" + str(beta))
cpd20 = LGCPD('var20', beta, variancevals[19], ['var12', 'var22', 'var26', 'var28'])

'''CPD21'''
cpd21 = LGCPD('var21', [0], variancevals[20])

'''CPD22'''
beta = betaVectorCalculator(['var8', 'var21', 'var26'], df, df.var22)
print("beta22=" + str(beta))
cpd22 = LGCPD('var22', beta, variancevals[21], ['var8', 'var21', 'var26'])

'''CPD23'''
beta = betaVectorCalculator(['var17', 'var28'], df, df.var23)
print("beta23=" + str(beta))
cpd23 = LGCPD('var23', beta, variancevals[22], ['var17', 'var28'])

'''CPD24'''
cpd24 = LGCPD('var24', [0], variancevals[23])

'''CPD25'''
beta = betaVectorCalculator(['var10', 'var19', 'var24'], df, df.var25)
print("beta25=" + str(beta))
cpd25 = LGCPD('var25', beta, variancevals[24], ['var10', 'var19', 'var24'])

'''CPD26'''
cpd26 = LGCPD('var26', [0], variancevals[25])

'''CPD27'''
beta = betaVectorCalculator(['var2', 'var26'], df, df.var27)
print("beta27=" + str(beta))
cpd27 = LGCPD('var27', beta, variancevals[26], ['var2', 'var26'])

'''CPD28'''
cpd28 = LGCPD('var28', [0], variancevals[27])

'''CPD29'''
beta = betaVectorCalculator(['var28', 'var30', 'var32', 'var34'], df, df.var29)
print("beta29=" + str(beta))
cpd29 = LGCPD('var29', beta, variancevals[28], ['var28', 'var30', 'var32', 'var34'])

'''CPD30'''
cpd30 = LGCPD('var30', [0], variancevals[29])

'''CPD31'''
beta = betaVectorCalculator(['var35'], df, df.var31)
print("beta31=" + str(beta))
cpd31 = LGCPD('var31', beta, variancevals[30], ['var35'])

'''CPD32'''
beta = betaVectorCalculator(['var10', 'var31', 'var37'], df, df.var32)
print("beta32=" + str(beta))
cpd32 = LGCPD('var32', beta, variancevals[31], ['var10', 'var31', 'var37'])

'''CPD33'''
beta = betaVectorCalculator(['var6', 'var7'], df, df.var33)
print("beta33=" + str(beta))
cpd33 = LGCPD('var33', beta, variancevals[32], ['var6', 'var7'])

'''CPD34'''
cpd34 = LGCPD('var34', [0], variancevals[33])

'''CPD35'''
beta = betaVectorCalculator(['var5', 'var14', 'var37'], df, df.var35)
print("beta35=" + str(beta))
cpd35 = LGCPD('var35', beta, variancevals[34], ['var5', 'var14', 'var37'])

'''CPD36'''
beta = betaVectorCalculator(['var39'], df, df.var36)
print("beta36=" + str(beta))
cpd36 = LGCPD('var36', beta, variancevals[35], ['var39'])

'''CPD37'''
beta = betaVectorCalculator(['var5', 'var36'], df, df.var37)
print("beta37=" + str(beta))
cpd37 = LGCPD('var37', beta, variancevals[36], ['var5', 'var36'])

'''CPD38'''
beta = betaVectorCalculator(['var39'], df, df.var38)
print("beta38=" + str(beta))
cpd38 = LGCPD('var38', beta, variancevals[37], ['var39'])

'''CPD39'''
cpd39 = LGCPD('var39', [0], variancevals[38])

cpd40 = LGCPD('v', [0], 1)
cpd41 = LGCPD('a', [0, 0], 1, ['v'])

model.add_cpds(cpd1, cpd2, cpd3, cpd4, cpd5, cpd6, cpd7, cpd8, cpd9, cpd10,
               cpd11, cpd12, cpd13, cpd14, cpd15, cpd16, cpd17, cpd18, cpd19, cpd20, cpd21, cpd22
               , cpd23, cpd24, cpd25, cpd26, cpd27, cpd28, cpd29, cpd30,
               cpd31, cpd32, cpd33, cpd34, cpd35, cpd36, cpd37, cpd38, cpd39, cpd40, cpd41)


print(model.get_cpds('var1'))
print(model.get_cpds('var2'))
print(model.get_cpds('var3'))
print(model.get_cpds('var4'))
print(model.get_cpds('var5'))
print(model.get_cpds('var6'))
print(model.get_cpds('var7'))
print(model.get_cpds('var8'))
print(model.get_cpds('var9'))
print(model.get_cpds('var10'))
print(model.get_cpds('var11'))
print(model.get_cpds('var12'))
print(model.get_cpds('var13'))
print(model.get_cpds('var14'))
print(model.get_cpds('var15'))
print(model.get_cpds('var16'))
print(model.get_cpds('var17'))
print(model.get_cpds('var18'))
print(model.get_cpds('var19'))
print(model.get_cpds('var20'))
print(model.get_cpds('var21'))
print(model.get_cpds('var22'))
print(model.get_cpds('var23'))
print(model.get_cpds('var24'))
print(model.get_cpds('var25'))
print(model.get_cpds('var26'))
print(model.get_cpds('var27'))
print(model.get_cpds('var28'))
print(model.get_cpds('var29'))
print(model.get_cpds('var30'))
print(model.get_cpds('var31'))
print(model.get_cpds('var32'))
print(model.get_cpds('var33'))
print(model.get_cpds('var34'))
print(model.get_cpds('var35'))
print(model.get_cpds('var36'))
print(model.get_cpds('var37'))
print(model.get_cpds('var38'))
print(model.get_cpds('var39'))


jgd = model.to_joint_gaussian()
sampler = HMC(model=jgd, grad_log_pdf=GradLogPDFGaussian, simulate_dynamics=LeapFrog)
samples = sampler.sample(initial_pos=np.ones(41), num_samples = 176,trajectory_length=2, stepsize=0.4)
samplesdatamatrix = np.empty([176, 39])

for i in range(0,39):
      a = samples["var"+str(i+1)]
      np.append(samplesdatamatrix, a[:, None],axis=1)

'''Entropyvalues for the 39 Nodes in the bayesian network'''

def EntropyCalculator(dataarray,meanvalues,sigmavalues):
    entropyvals = []
    for i in range(0,39):
        totallogpdf = 0
        entropy = 0 ;
        for j in range(0,176):
            totallogpdf += sp.norm.logpdf(dataarray[j,i],meanvalues[i],sigmavalues[i])
            entropy = (-1*totallogpdf)/176
        entropyvals.append(entropy)
    return entropyvals

'''Relative Entropyvalues for the 39 Nodes in the bayesian network'''
def RelativeEntropyCalculator(time_series,samplesdatamatrix,timeseriesigmavals,sampledsigmavals,timeseriesmeanvals,sampledmeanvals):
    relativeentropyvals = []
    for i in range(0,39):
        totallogpdf = 0
        relativeentropy = 0
        for j in range(0,176):
            totallogpdf +=(sp.norm.logpdf(samplesdatamatrix[j,i],sampledmeanvals[i],sampledsigmavals[i])- sp.norm.logpdf(time_series[j,i],timeseriesmeanvals[i],timeseriesigmavals[i]))
            relativeentropy = (-1*totallogpdf)/176
        relativeentropyvals.append(relativeentropy)
    return relativeentropyvals

sampledmeanvals = []
sampledsigmavals =[]

for i in range(0,39):
   sampledmeanvals.append(np.mean(sampledatamatrix[:,i]))
   sampledsigmavals.append(np.std(sampledatamatrix[:,i]))



TimeSeriesEntropyVals = EntropyCalculator(time_series,meanvals,meanvals)
SampledValuesEntropyVals = EntropyCalculator(sampledatamatrix,sampledmeanvals,sampledsigmavals)
RelativeEntropy = RelativeEntropyCalculator(time_series,samplesdatamatrix,sigmavals,sampledsigmavals,meanvals,sampledmeanvals)

print("MeanforGivenDataValues")
print(str(meanvals))
print("")


print("EntropyforGivenDataValues")
print(str(TimeSeriesEntropyVals))
print("")

print("MeanforSampleddatavalues")
print(str(sampledmeanvals))
print("")

print("EntropyforSampledDataValues")
print(str(SampledValuesEntropyVals))
print("")

print("RelativeEntopyValues=")
print(str(RelativeEntropy))
print("")

'''2.2 DOMAIN SPECIFIC'''
'''Approximate Inference using Hamiltonian Monte Carlo Sampling'''


def QueryConditionalProbabilityCalculation(data1, data2, data3, data4, gaussianmeans, gaussianvar, childdata):
    data1mean = np.mean(data1)
    data2mean = np.mean(data2)
    data3mean = np.mean(data3)
    data4mean = np.mean(data4)

    t1 = np.vstack([data1, data2, data3, data4])
    covar = np.cov(t1)

    mean = np.array([data1mean, data2mean, data3mean, data4mean])
    covariance = np.array(covar)
    model = JointGaussianDistribution(['A', 'B', 'C', 'D'], mean, covariance)

    # Creating a HMC sampling instance
    sampler = HMC(model=model, grad_log_pdf=GradLogPDFGaussian, simulate_dynamics=LeapFrog)
    # Drawing samples
    samples = sampler.sample(initial_pos=np.ones(4), num_samples=100,
                             trajectory_length=2, stepsize=0.4)

    sampleddata1mean = np.mean(samples['B'].values)
    sampleddata2mean = np.mean(samples['C'].values)
    sampleddata3mean = np.mean(samples['D'].values)

    meanvar = gaussianmeans[0] * sampleddata1mean + gaussianmeans[1] * sampleddata2mean + gaussianmeans[
                                                                                              2] * sampleddata3mean + \
              gaussianmeans[3]

    '''Conditional Probabbliltiy calculation'''
    probablilityforquery = sp.norm.pdf(childdata, meanvar, np.sqrt(gaussianvar))

    return probablilityforquery


'''Inference for Query1'''
# P(var1 | var2, var14, var23) = N(0.748011140248*var2 + 0.122129899091*var14 + 0.104279336211*var23 + 0.00010139683392
# ,0.515764973359);
probablilityforquery1 = QueryConditionalProbabilityCalculation(time_series[0], time_series[1], time_series[13],
                                                               time_series[22],
                                                               [0.748011140248, 0.122129899091, 0.104279336211,
                                                                0.00010139683392], 0.515764973359, time_series[4, 0])
print("probablilityforquery1 = " + str(probablilityforquery1))

'''Inference for Query2'''
# P(var4 | var6, var10, var26) = N(0.472209861533*var6 + -0.207914792048*var10 + -0.317906750109*var26 + -0.000208583984668; 1.11899421783)
probablilityforquery2 = QueryConditionalProbabilityCalculation(time_series[3], time_series[5], time_series[9],
                                                               time_series[25],
                                                               [0.472209861533, -0.207914792048, -0.317906750109,
                                                                -0.000208583984668], 1.11899421783, time_series[4, 3])

print("probablilityforquery2 = " + str(probablilityforquery2))

# P(var11 | var9, var10, var17) = N(-0.126538532496*var9 + 0.22976436421*var10 + 0.692552775779*var17 + -0.000173862156993; 0.900560514493)
'''Inference for Query3'''
probablilityforquery3 = QueryConditionalProbabilityCalculation(time_series[10], time_series[8], time_series[9],
                                                               time_series[16],
                                                               [-0.126538532496, 0.22976436421, 0.692552775779,
                                                                0.00010139683392], 0.900560514493, time_series[4, 10])

print("probablilityforquery3 =" + str(probablilityforquery3))

'''Inference for Query4'''

# P(var25 | var10, var19, var24) = N(0.380799057427*var10 + -0.00570509488907*var19 + 0.221949580904*var24 + 0.00028431637559; 0.741570405167)
probablilityforquery4 = QueryConditionalProbabilityCalculation(time_series[24], time_series[9], time_series[18],
                                                               time_series[23],
                                                               [0.380799057427, -0.00570509488907, 0.221949580904,
                                                                0.00028431637559], 0.741570405167, time_series[4, 24])
print("probablilityforquery4 =" + str(probablilityforquery4))






