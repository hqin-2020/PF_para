import numpy as np
import scipy as sp
import seaborn as sns
sns.set()

def simulate(θ_true, T):
    
    Azo = θ_true['Azo']; Azz = θ_true['Azz']; Bz = θ_true['Bz']
    Ass = θ_true['Ass']; Bs = θ_true['Bs']
    
    Z01 = 0
    Z02 = Azo[1,0]/(1-Azz[1,1])
    S0 = sp.linalg.solve((np.eye(3) - Ass), np.zeros([3,1]))

    Z = np.zeros((2,T)) 
    S = np.zeros((3,T)) 
    Z[:,[0]] = np.array([[Z01],[Z02]])
    S[:,[0]] = S0

    np.random.seed(0)
    Wz = np.random.multivariate_normal(np.zeros(2), np.eye(2), T).T
    np.random.seed(1)
    Ws = np.random.multivariate_normal(np.zeros(3), np.eye(3), T).T

    for t in range(T-1):
        Z[:,[t+1]] = Azo + Azz @ Z[:,[t]] + Bz @ Wz[:,[t+1]]
        S[:,[t+1]] = Ass @ S[:,[t]] + Bs @ Ws[:,[t+1]]

    D = np.ones((3,1)) @ Z[[0],:] + S
    
    return D

def decompose_θ(θ):
    
    λ = θ['Azz'][1,1]; η = θ['Azo'][1,0]
    b11 = θ['Bz'][0,0]; b22 = θ['Bz'][1,1]
    As11 = θ['Ass'][0,0]; As12 = θ['Ass'][0,1]; As13 = θ['Ass'][0,2]
    As21 = θ['Ass'][1,0]; As22 = θ['Ass'][1,1]; As23 = θ['Ass'][1,2]
    As31 = θ['Ass'][2,0]; As32 = θ['Ass'][2,1]; As33 = θ['Ass'][2,2]
    Aso2 = θ['Aso'][1,0]; Aso3 = θ['Aso'][2,0]
    Bs11 = θ['Bs'][0,0];  Bs21 = θ['Bs'][1,0];  Bs22 = θ['Bs'][1,1]; Bs31 = θ['Bs'][2,0]; Bs32 = θ['Bs'][2,1];  Bs33 = θ['Bs'][2,2]
    
    (P, L, U) = sp.linalg.lu(θ['Bs']@θ['Bs'].T)
    D = np.diag(np.diag(U))   # D is just the diagonal of U
    U /= np.diag(U)[:, None]  # Normalize rows of U
    J = L
    Δ = D
    J_inv = sp.linalg.inv(J)
    j21 = J_inv[1,0]; j31 = J_inv[2,0]; j32 = J_inv[2,1]

    return λ, η, b11, b22, As11, As12, As13, As21, As22, As23, Aso2, As31, As32, As33, Aso3, Bs11, Bs21, Bs22, Bs31, Bs32, Bs33, j21, j31, j32
    
def draw_para(b, Λ, c, d):
    
    ζ = sp.stats.gamma.rvs(c/2+1, loc = 0, scale = 1/(d/2))
    σ2 = 1/ζ
    β = np.random.multivariate_normal(b.flatten(), sp.linalg.inv(ζ*Λ)).reshape(-1,1)
    return β, σ2

def update_θ(H):
    
    stability_max_iter = 200_000
    
    β_z1, σ2_z1 = draw_para(H['1st'][0], H['1st'][1], H['1st'][2], H['1st'][3])
    b11 = np.sqrt(σ2_z1)
    
    λ_stable = False
    λ_iter = 0
    while λ_stable == False:
        β_z2, σ2_z2 = draw_para(H['2nd'][0], H['2nd'][1], H['2nd'][2], H['2nd'][3])
        η = β_z2[0,0] 
        λ = β_z2[1,0] 
        b22 = np.sqrt(σ2_z2)
        λ_iter += 1
        if abs(λ) < 0.99:
            λ_stable = True
        elif λ_iter > stability_max_iter:
            print('λ Unstable')
        else:
            λ_stable = False

    Azo = np.array([[0],[η]])
    Azz = np.array([[1, 1],[0, λ]])
    Bz = np.array([[b11, 0],[0, b22]])

    Ass_stable = False
    Ass_iter = 0
    while Ass_stable == False:
        β_s1, σ2_s1 = draw_para(H['3rd'][0], H['3rd'][1], H['3rd'][2], H['3rd'][3])
        β_s2, σ2_s2 = draw_para(H['4th'][0], H['4th'][1], H['4th'][2], H['4th'][3])
        β_s3, σ2_s3 = draw_para(H['5th'][0], H['5th'][1], H['5th'][2], H['5th'][3])

        J_inv = np.array([[1,                  0,        0],\
                          [-β_s2[1,0],         1,        0],\
                          [-β_s3[1,0], -β_s3[2,0],       1]])
        Aso = sp.linalg.solve(J_inv, np.array([[0], β_s2[0], β_s3[0]]))
        Ass = sp.linalg.solve(J_inv, np.array([[β_s1[0,0], β_s1[1,0], β_s1[2,0]],\
                                               [β_s2[2,0], β_s2[3,0], β_s2[4,0]],\
                                               [β_s3[3,0], β_s3[4,0], β_s3[5,0]]]))
        Ass_iter += 1
        if np.max(abs(np.linalg.eigvals(Ass)))<0.99:
            Ass_stable = True
        elif Ass_iter > stability_max_iter:
            print('Ass Unstable')
        else:
            Ass_stable = False
    
    Bs = sp.linalg.solve(J_inv, np.diag([σ2_s1,σ2_s2,σ2_s3])**0.5)
    θ = {'Azo' : Azo, 'Azz' : Azz, 'Bz' : Bz, 'Aso' : Aso, 'Ass' : Ass, 'Bs' : Bs, 'J_inv': J_inv, 'λ_iter': λ_iter, 'Ass_iter' : Ass_iter}
    
    return θ

def init_H(θ_true, Λ_scale, cd_scale):
    
    λ, η, b11, b22, As11, As12, As13, As21, As22, As23, Aso2, As31, As32, As33, Aso3, Bs11, Bs21, Bs22, Bs31, Bs32, Bs33, j21, j31, j32 = decompose_θ(θ_true)
    
    H0 = {'1st':[np.array([[1],[1]]), np.eye(2) * Λ_scale, 1 * cd_scale, 1 * cd_scale], 
          '2nd':[np.array([[η],[λ]]), np.eye(2) * Λ_scale, 1 * cd_scale, 1 * cd_scale], 
          '3rd':[np.array([[As11], [As12], [As13]]), np.eye(3) * Λ_scale, 1 * cd_scale, 1 * cd_scale], 
          '4th':[np.array([[Aso2], [-j21], [As11*j21+As21], [As12*j21+As22], [As13*j21+As23]]), np.eye(5) * Λ_scale, 1 * cd_scale, 1 * cd_scale],
          '5th':[np.array([[Aso2*j32+Aso3], [-j31], [-j32], [As11*j31+As21*j32+As31], [As12*j31+As22*j32+As32], [As13*j31+As23*j32+As33]]), np.eye(6) * Λ_scale, 1 * cd_scale, 1 * cd_scale]}
    
    return H0

def init_X(θ, D_0):
    
    λ, η, b11, b22, As11, As12, As13, As21, As22, As23, Aso2, As31, As32, As33, Aso3, Bs11, Bs21, Bs22, Bs31, Bs32, Bs33, j21, j31, j32 = decompose_θ(θ)
    ones = np.ones([3,1])
    Ass = np.array([[As11, As12, As13],\
                    [As21, As22, As23],\
                    [As31, As32, As33]])
    Aso = np.array([[0],\
                    [Aso2],\
                    [Aso3]])
    Bs =  np.array([[Bs11, 0,    0],\
                    [Bs21, Bs22, 0],\
                    [Bs31, Bs32, Bs33]])
    
    μs = sp.linalg.solve(np.eye(3) - Ass, Aso) 
    Σs = sp.linalg.solve_discrete_lyapunov(Ass, Bs@Bs.T)
    
    β = sp.linalg.solve(np.hstack([Σs@np.array([[1,1],[0,-1],[-1,0]]), ones]).T, np.array([[0,0,1]]).T)                                     
    γ1 = np.array([[1],[0],[0]]) - sp.linalg.inv(Σs)@ones/(ones.T@sp.linalg.inv(Σs)@ones)
    γ2 = np.array([[0],[1],[0]]) - sp.linalg.inv(Σs)@ones/(ones.T@sp.linalg.inv(Σs)@ones)
    γ3 = np.array([[0],[0],[1]]) - sp.linalg.inv(Σs)@ones/(ones.T@sp.linalg.inv(Σs)@ones)
    Γ = np.hstack([γ1, γ2, γ3])
    
    Z01 = β.T@(D_0 - μs)
    Σz01 = 0
    Z02 = η/(1-λ)
    Σz02 = b22**2/(1-λ**2)
    S0 = Γ.T@(D_0 - μs) + μs
    Σs0 = (1/(ones.T@sp.linalg.inv(Σs)@ones))[0][0]
    
    μ0 = np.array([[Z01[0][0]],\
                   [Z02],\
                   [S0[0][0]],\
                   [S0[1][0]],\
                   [S0[2][0]]])
    Σ0 = np.array([[Σz01,0,    0,   0,   0],\
                   [0,   Σz02, 0,   0,   0],\
                   [0,   0,    Σs0, Σs0, Σs0],\
                   [0,   0,    Σs0, Σs0, Σs0],\
                   [0,   0,    Σs0, Σs0, Σs0]]) 
    return μ0, Σ0

def init(θ_true, D0, Λ_scale, cd_scale):
    
    H0 = init_H(θ_true, Λ_scale, cd_scale)
    Σ0_postive = False
    while Σ0_postive == False:
        θ0 = update_θ(H0)
        μ0, Σ0 = init_X(θ0, D0)
        if np.all(np.linalg.eigvals(Σ0)>=0) == True:
            Σ0_postive = True
        else: 
            Σ0_postive = False
    
    X0 = sp.stats.multivariate_normal.rvs(μ0.flatten(), Σ0).reshape(-1,1)
    ν0 = 1

    return θ0, X0, H0, ν0

def update_X(Dt_next, Xt, θt_next):
    
    Azo = θt_next['Azo']; Azz = θt_next['Azz']; Bz = θt_next['Bz']; Bz1 = Bz[[0],:]
    Aso = θt_next['Aso']; Ass = θt_next['Ass']; Bs = θt_next['Bs']
    Zt = Xt[0:2,:]; Zt1 = Zt[0,0]; Zt2 = Zt[1,0]
    St = Xt[2:5,:]
    ones = np.ones([3,1])

    Φ = sp.linalg.solve(ones@Bz1@Bz1.T@ones.T + Bs@Bs.T, ones@Bz1@Bz.T)
    Γ = sp.linalg.solve(ones@Bz1@Bz1.T@ones.T + Bs@Bs.T, Bs@Bs.T)
    
    mean = np.vstack([Azo + Azz@Zt + Φ.T@(Dt_next-ones*Zt1 - ones*Zt2 - Aso - Ass@St),\
                      Aso + Ass@St + Γ.T@(Dt_next-ones*Zt1 - ones*Zt2 - Aso - Ass@St)])
    cov = np.vstack([np.hstack([Bz@Bz.T, np.zeros([2,3])]),\
                     np.hstack([np.zeros([3,2]), Bs@Bs.T])]) -\
          np.vstack([Φ.T, Γ.T]) @ (ones@Bz1@Bz1.T@ones.T+Bs@Bs.T)@np.hstack([Φ, Γ])

    Xt_next = sp.stats.multivariate_normal.rvs(mean.flatten(), cov).reshape(-1,1)
    
    return Xt_next

def bayes_para_update(bt, Λt, ct, dt, Rt_next, Zt_next):
    
    Λt_next = Λt + Rt_next@Rt_next.T
    bt_next = sp.linalg.solve(Λt_next, Λt@bt + Rt_next*Zt_next)
    ct_next = ct + 1
    dt_next = Zt_next**2 - bt_next.T@Λt_next@bt_next + bt.T@Λt@bt + dt
    
    return bt_next, Λt_next, ct_next, dt_next

def update_H(Xt_next, Xt, Ht):
    
    Zt1 = Xt[0,0]; Zt2 = Xt[1,0]; St1 = Xt[2,0]; St2 = Xt[3,0]; St3 = Xt[4,0];   
    Zt_next_1 = Xt_next[0,0];  Zt_next_2 = Xt_next[1,0];  St_next_1 = Xt_next[2,0];  St_next_2 = Xt_next[3,0];  St_next_3 = Xt_next[4,0];  
    
    first_eq_Rt_next = np.array([[Zt1],[Zt2]])
    first_eq_Zt_next = Zt_next_1
    first_eq_bt_next = np.array([[1],[1]])
    first_eq_Λt_next = Ht['1st'][1] + first_eq_Rt_next@first_eq_Rt_next.T
    first_eq_ct_next = Ht['1st'][2] + 1
    first_eq_dt_next = (first_eq_Zt_next - first_eq_Rt_next.T@first_eq_bt_next)**2  + Ht['1st'][3]
    
    second_eq_bt_next, second_eq_Λt_next, second_eq_ct_next, second_eq_dt_next = \
    bayes_para_update(Ht['2nd'][0], Ht['2nd'][1], Ht['2nd'][2], Ht['2nd'][3], np.array([[1],[Zt2]]), Zt_next_2)
    
    third_eq_bt_next, third_eq_Λt_next, third_eq_ct_next, third_eq_dt_next = \
    bayes_para_update(Ht['3rd'][0], Ht['3rd'][1], Ht['3rd'][2], Ht['3rd'][3], np.array([[St1],[St2],[St3]]), St_next_1)
    
    fourth_eq_bt_next, fourth_eq_Λt_next, fourth_eq_ct_next, fourth_eq_dt_next = \
    bayes_para_update(Ht['4th'][0], Ht['4th'][1], Ht['4th'][2], Ht['4th'][3], np.array([[1],[St_next_1],[St1],[St2],[St3]]), St_next_2)
    
    fifth_eq_bt_next, fifth_eq_Λt_next, fifth_eq_ct_next, fifth_eq_dt_next = \
    bayes_para_update(Ht['5th'][0], Ht['5th'][1], Ht['5th'][2], Ht['5th'][3], np.array([[1],[St_next_1],[St_next_2],[St1],[St2],[St3]]), St_next_3)
    
    Ht_next = {'1st':[first_eq_bt_next, first_eq_Λt_next, first_eq_ct_next, first_eq_dt_next],
               '2nd':[second_eq_bt_next, second_eq_Λt_next, second_eq_ct_next, second_eq_dt_next],
               '3rd':[third_eq_bt_next, third_eq_Λt_next, third_eq_ct_next, third_eq_dt_next],
               '4th':[fourth_eq_bt_next, fourth_eq_Λt_next, fourth_eq_ct_next, fourth_eq_dt_next],
               '5th':[fifth_eq_bt_next, fifth_eq_Λt_next, fifth_eq_ct_next, fifth_eq_dt_next]}
                      
    return Ht_next

def update_ν(Dt_next, Xt, θt_next):
    
    ones = np.ones([3,1])
    Azo = θt_next['Azo']; Azz = θt_next['Azz']; Bz = θt_next['Bz']; Bz1 = Bz[[0],:]
    Aso = θt_next['Aso']; Ass = θt_next['Ass']; Bs = θt_next['Bs']
    Zt = Xt[0:2,:]; Zt1 = Zt[0,0]; Zt2 = Zt[1,0]
    St = Xt[2:5,:]

    mean = ones*Zt1 +ones*Zt2 + Aso + Ass@St
    cov = ones@Bz1@Bz1.T@ones.T+Bs@Bs.T
    
    density = sp.stats.multivariate_normal.pdf(Dt_next.flatten(), mean.flatten(), cov)
    
    return density

def update_θXHν(input_ti):
    
    Xti, Hti, Dt_next, seed = input_ti
    np.random.seed(seed)
    θti_next = update_θ(Hti)
    Xti_next = update_X(Dt_next, Xti, θti_next)
    Hti_next = update_H(Xti_next, Xti, Hti)
    νti_next = update_ν(Dt_next, Xti, θti_next)
    
    return [θti_next, Xti_next, Hti_next, νti_next]