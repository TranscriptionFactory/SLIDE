import json, numpy as np, sys
A='.tmpfiles/2026-09-28_sampler-audit/run_5/marginal_chunk_4'
meta=json.load(open('.tmpfiles/2026-09-14_paired-convergence/results/marginal_chunk_4/inputs/metadata.json'))
names=meta['lf_names']; n,p=meta['shape']['Z']
sig=np.loadtxt(f'{A}/sigma.csv',delimiter=','); rs=np.loadtxt(f'{A}/rust_s.csv'); Rs=np.loadtxt(f'{A}/r_s.csv')
print('n,p',n,p,'sigma eig min/max',np.linalg.eigvalsh(sig)[[0,-1]])
d=np.diag(sig)
for j in range(p): print(names[j], f'var={d[j]:.4g} rust={rs[j]:.4e} R={Rs[j]:.4e} R/var={Rs[j]/d[j]:.3e}')
def sk(s): S=np.diag(s); return 2*S-S@np.linalg.solve(sig,S)
for lab,s in [('rust',rs),('R',Rs)]:
    K=sk(s); K=(K+K.T)/2; e=np.linalg.eigvalsh(K); print(lab,'Sigma_k eig min',e[:4],'maxdiag',K.diagonal().max(), 'feas 2sig-S min eig',np.linalg.eigvalsh(2*sig-np.diag(s))[0])
    for name in ['shared_sigma','shared_s']:
        F=np.fromfile(f'{A}/rust_{name}_factor.raw','<f8').reshape(p,p)
        if (name=='shared_sigma')==(lab=='rust'):
            D=F.T@F-K; print('  factor resid diag',np.round(np.diag(D),14)[:p], 'offdiag max',np.abs(D-np.diag(np.diag(D))).max())
z=rs==0; print('zero idx',[names[j] for j in np.where(z)[0]])
h=Rs.copy(); h[z]=0; print('R S zeroed at rust zeros: 2sig-S min eig',np.linalg.eigvalsh(2*sig-np.diag(h))[0])
h2=rs.copy(); h2[z]=Rs[z]; print('rust S with R at zeros: 2sig-S min eig',np.linalg.eigvalsh(2*sig-np.diag(h2))[0])
print('--- factorization checks')
def block_factor(s):
    K=sk(s); K=(K+K.T)/2; nz=np.where(s>0)[0]
    L=np.zeros((p,p)); L[np.ix_(nz,nz)]=np.linalg.cholesky(K[np.ix_(nz,nz)])
    return K,L
for lab,s in [('C rust exact',rs),('D rust R-zeros',h2),('B R',Rs)]:
    K=sk(s); K=(K+K.T)/2; nz=s>0
    print(lab,'all eig[:7]',np.linalg.eigvalsh(K)[:7]); print('  nonzero block min eig',np.linalg.eigvalsh(K[np.ix_(nz,nz)])[0])
    try: K2,L=block_factor(s); print('  block chol ok, resid',np.abs(L@L.T-K2).max(), 'min Ldiag nz',np.diag(L)[nz].min())
    except np.linalg.LinAlgError as e: print('  chol FAIL',e)
import glob
w=[json.load(open(f))['wall_ms'] for f in sorted(glob.glob('.tmpfiles/2026-10-05_s-selection/full/marginal_chunk_4/rust_s/iter_0*.json'))[:50]]
print('wall ms median',np.median(w))
