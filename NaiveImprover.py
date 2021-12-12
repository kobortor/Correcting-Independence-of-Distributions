import random;
#If x=y for a product distribution. Then we want to improve it to independent distribution(x,y)
#Step 1:generate a dataset
n=100;
P=[];
for i in range(n):
    P.append([0]*n);
for i in range(n):
    for j in range(n):
        elem=random.randint(0,n);
        P[i][j]=[elem,elem]
#Pick r=n, just pick a sample space less than o(n^2)
r=n;
Q=[];
X=[];
Y=[];
for i in range(r):
    i=random.randint(0,n-1);
    j=random.randint(0,n-1);
    Q.append(P[i][j]);
    X.append(P[i][j][0]);
    Y.append(P[i][j][1]);
#Now pick x_i y_j independently
size=100;
#initial
Improver_out_dist=[];
for i in range(size):
    Improver_out_dist.append([0]*size);
#Pick from X and Y
for i in range(size):
    for j in range(size):
        x_pos=random.randint(0,r-1);
        y_pos=random.randint(0,r-1);
        Improver_out_dist[i][j]=[X[x_pos],Y[y_pos]];
print(Improver_out_dist)



