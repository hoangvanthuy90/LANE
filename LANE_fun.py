import numpy as np
    
def LANE_fun(Net = None,Attri = None,LabelY = None,d = None,alpha1 = None,alpha2 = None,varargin = None): 
    #Jointly embed labels and attriuted network into embedding representation H
#     H = LANE_fun(Net,Attri,LabelY,d,alpha1,alpha2,numiter);
#     H = AANE_fun(Net,Attri,d,alpha1,alpha2,numiter);
    
    #          Net   is the weighted adjacency matrix
#         Attri  is the attribute information matrix with row denotes nodes
#        LabelY  is the label information matrix
#          d     is the dimension of the embedding representation
#         alpha1 is the weight for node attribute information
#         alpha2 is the weight for label information
#        numiter is the max number of iteration
    
    #   Copyright 2017, Xiao Huang and Jundong Li.
#   $Revision: 1.0.0 $  $Date: 2017/10/18 00:00:00 $
    
    n = Net.shape[1-1]
    LG = norLap(Net)
    
    LA = norLap(Attri)
    
    UAUAT = np.zeros((n,n))
    
    opts.disp = 0
    if len(varargin)==0:
        ## Unsupervised attriuted network embedding
# Input of Parameters
        numiter = alpha2
        beta1 = d
        beta2 = alpha1
        d = LabelY
        H = np.zeros((n,d))
        for i in np.arange(1,numiter+1).reshape(-1):
            HHT = H * np.transpose(H)
            TotalLG1 = LG + beta2 * UAUAT + HHT
            UG,__ = eigs(0.5 * (TotalLG1 + np.transpose(TotalLG1)),d,'LA',opts)
            UGUGT = UG * np.transpose(UG)
            TotalLA = beta1 * LA + beta2 * UGUGT + HHT
            UA,__ = eigs(0.5 * (TotalLA + np.transpose(TotalLA)),d,'LA',opts)
            UAUAT = UA * np.transpose(UA)
            TotalLH = UAUAT + UGUGT
            H,__ = eigs(0.5 * (TotalLH + np.transpose(TotalLH)),d,'LA',opts)
    else:
        ## Supervised attriuted network embedding
        numiter = varargin[0]
        H = np.zeros((n,d))
        LY = norLap(LabelY * np.transpose(LabelY))
        UYUYT = np.zeros((n,n))
        # Iterations
        for i in np.arange(1,numiter+1).reshape(-1):
            HHT = H * np.transpose(H)
            TotalLG1 = LG + alpha1 * UAUAT + alpha2 * UYUYT + HHT
            UG,__ = eigs(0.5 * (TotalLG1 + np.transpose(TotalLG1)),d,'LA',opts)
            UGUGT = UG * np.transpose(UG)
            TotalLA = alpha1 * (LA + UGUGT) + HHT
            UA,__ = eigs(0.5 * (TotalLA + np.transpose(TotalLA)),d,'LA',opts)
            UAUAT = UA * np.transpose(UA)
            TotalLY = alpha2 * (LY + UGUGT) + HHT
            UY,__ = eigs(0.5 * (TotalLY + np.transpose(TotalLY)),d,'LA',opts)
            UYUYT = UY * np.transpose(UY)
            TotalLH = UAUAT + UGUGT + UYUYT
            H,__ = eigs(0.5 * (TotalLH + np.transpose(TotalLH)),d,'LA',opts)
    
    return H
    
    
def norLap(InpX = None): 
    # Compute the normalized graph Laplacian of InpX
    InpX = np.transpose(InpX)
    
    InpX = bsxfun(rdivide,InpX,sum(InpX ** 2) ** 0.5)
    
    InpX[np.isnan[InpX]] = 0
    SX = np.transpose(InpX) * InpX
    nX = len(SX)
    SX[np.arange[1,nX ** 2+nX + 1,nX + 1]] = 1 + 10 ** - 6
    DXInv = spdiags(full(np.sum(SX, 2-1)) ** (- 0.5),0,nX,nX)
    LapX = DXInv * SX * DXInv
    LapX = 0.5 * (LapX + np.transpose(LapX))
    return LapX
    
    return H