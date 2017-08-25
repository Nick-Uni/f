from numpy import*;t,y=load('m');f=3136
predict=lambda x:y[(x@(ones((f,6666))-t)+(ones((x.shape[0],f))-x)@t).argmin(1)]