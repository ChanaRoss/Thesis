import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn as sk
import seaborn as sns
from scipy import optimize

# 2 features (sum neighbors + Sum Cell)
# def feature(mat,i,j):
#     feature1 = mat[i][j]
#     matPadded = np.pad(mat,1,mode = 'constant')
#     feature2 = np.sum(matPadded[i:i+3,j:j+3]) - matPadded[i+1,j+1]
#     return np.vstack([feature1,feature2])

# 9 features (Neighbors)
# def feature3(mat,i,j):
#     feature1 = mat[i][j]
#     matPadded = np.pad(mat,1,mode = 'constant')
#     neighbors = matPadded[i:i+3,j:j+3]
#     featureOut = neighbors.reshape(neighbors.size)
#     return featureOut

# 5 features (sum Evnes in x + sum rows + sum columns)
# def feature(mat,i,j):
#     feature1 = list(np.array([mat[i][j]]))
#     numPadding = 10
#     matPadded = np.pad(mat,numPadding,mode = 'constant')
#     # neighbors = matPadded[i:i+3,j:j+3]
#     # featureOut = neighbors.reshape(neighbors.size)
#     feature1.append(np.sum(mat[i,:]))
#     feature1.append(np.sum(mat[:,j]))
#     feature1.append(np.sum(matPadded[i+numPadding-1:i+2+numPadding,j-1+numPadding:j+2 +numPadding]) - matPadded[i+numPadding,j+numPadding])
#     feature1.append(np.sum(matPadded[i+numPadding-5:i+6+numPadding,j-5+numPadding:j+6 +numPadding]) - matPadded[i+numPadding,j+numPadding])
#     return feature1

# # 3 features (sum Evnes in x + sum rows + sum columns)
# def feature(mat,i,j):
#     feature1 = list(np.array([mat[i][j]]))
#     # matPadded = np.pad(mat,1,mode = 'constant')
#     # neighbors = matPadded[i:i+3,j:j+3]
#     # featureOut = neighbors.reshape(neighbors.size)
#     feature1.append(np.sum(mat[i,:]))
#     feature1.append(np.sum(mat[:,j]))
#     return feature1


# 11 features (Neighbors + sum rows + sum columns)
def feature(mat,i,j):
    feature1 = mat[i][j]
    matPadded = np.pad(mat,1,mode = 'constant')
    neighbors = matPadded[i:i+3,j:j+3]
    feature2 = list(neighbors.reshape(neighbors.size))
    feature2.append(np.sum(mat[i,:]))
    feature2.append(np.sum(mat[:,j]))
    featureOut = np.vstack(feature2)
    return featureOut

# def feature(mat,i,j):
#     return mat[i][j]

def features(mat,i,j):
    out = np.zeros(mat.shape)
    out[i][j] = 1
    return out.reshape(out.size,1)

def f(lmda,mat):
    z_lmda = 0
    # mat = args[0] # written this way for optimize function in scipy
    matSum = np.sum(mat.astype(np.int64))
    assert(matSum!=0)
    tempSum = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            feature_out = feature(mat,i,j)
            tempSum += mat[i][j]*np.dot(lmda,feature_out)
            z_lmda  += np.exp(np.dot(lmda,feature(mat,i,j)))
    tempSum = float(tempSum)/matSum
    result = np.log(z_lmda) - tempSum
    return  result

def pTheoretical(lmda,mat,index_x,index_y,matX,matY,z_lmda):
    result = np.exp(np.dot(lmda,feature(mat,index_x,index_y)))/z_lmda
    return result


def main():
    # 0 : feature is sum of all events in mat , 1 : features are indicators , 2: two features (sum and sum of neighbors)
    typeFeatures = 2
    matTemp = np.load('/home/chana/Documents/Thesis/Uber_Analysis/PostAnalysis/Mat.p')
    mat = matTemp[20:60][0:44]
    # mat = np.array([[1, 1, 1], [1, 4, 1], [3, 1, 10], [5, 2, 9], [1, 2, 3], [2, 2, 2],[0,0,0]]).astype(float)

    matX = mat.shape[0]
    matY = mat.shape[1]
    if typeFeatures == 0:
        lmda0 = 1
    elif typeFeatures == 1:
        lmda0 = np.ones(shape = (1,mat.size))
    elif typeFeatures==2:
        # lmda0 = np.ones(shape = (1,9))
        lmda0 = np.zeros(shape=(1, 11))+0.05
        # lmda0 = np.ones(shape=(1, 2))
    # result = optimize.fmin(f,x0=lmda0,args=(mat,),xtol=1e-3,ftol=1e-4,maxiter=2000,full_output=True,disp=True,retall=True)
    result = optimize.minimize(f,x0 = lmda0,args =(mat,),method = 'Powell',tol = 0.01,options={'maxiter' : 400})
    # lmdaOpt,fOpt,iter,funcalls,warnflg,allvecs = result
    lmdaOpt = result.x
    print(result.message)
    print(lmdaOpt)


    # lmda = np.linspace(-10,10,100)
    # fout = [f(lmdatemp,mat) for lmdatemp in lmda]
    # fig = plt.figure(1)
    # plt.plot(lmda,fout)
    # # plt.show()

    matSum = np.sum(mat)
    pNumeric = mat/matSum
    pTheory = np.zeros_like(pNumeric)
    z_lmda = 0
    for i in range(matX):
        for j in range(matY):
            z_lmda += np.exp(np.dot(lmdaOpt,feature(mat,i,j)))
    for i in range(matX):
        for j in range(matY):
            pTheory[i][j] = pTheoretical(lmdaOpt,mat,i,j,matX,matY,z_lmda)
    norm1 = np.linalg.norm(pTheory.reshape(1,pTheory.size) - pNumeric.reshape(1,pNumeric.size),1)
    print('norm1 is:' + str(norm1))
    # pTheory.dump('pTheory.p')
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(211)
    ax1 = plt.matshow(pNumeric)
    # ax2 = fig1.add_subplot(212)
    ax2 = plt.matshow(pTheory)
    plt.show()
    print('done')

if __name__ == '__main__':
    main()
    print('im done')